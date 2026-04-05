"""
Anotação automática dos sinais de Libras com Qwen2-VL-7B (4-bit).

Pipeline:
  1. Para cada vídeo dos sinalizadores (Zenodo), roda Qwen2-VL com prompt
     estruturado para fonologia de língua de sinais
  2. Salva JSON por vídeo: CM, PA, MOV, OR, ENM
  3. Agrega por classe: gera o dicionário fônico das 20 classes

Saídas:
  data/phonetic/raw/<class>/<filename>.json  — anotação por vídeo
  data/phonetic/dictionary.json              — dicionário agregado por classe

Uso:
    cd /home/jonathangois/Documentos/Libras_2026
    source env/bin/activate
    python3 -u src/annotate_qwen.py 2>&1 | tee results/annotate_qwen.log

Nota: requer os vídeos em data/videos_zenodo/ (download_zenodo.py).
      Roda na GPU se disponível (~4GB VRAM com quantização 4-bit).
"""

import os, json, re, time
from pathlib import Path
from collections import defaultdict

import torch

# ── Configuração ────────────────────────────────────────────────────────────
VIDEO_DIR  = Path("data/videos_zenodo")
OUT_DIR    = Path("data/phonetic")
DICT_FILE  = OUT_DIR / "dictionary.json"
MODEL_ID   = "Qwen/Qwen2-VL-7B-Instruct"
USE_4BIT   = True        # quantização 4-bit → ~4GB VRAM
MAX_FRAMES = 16          # frames amostrados do vídeo (Qwen2-VL aceita até ~32)
SKIP_EXISTING = True     # não re-anota vídeos já processados

CLASSES = {
    "01":"acontecer","02":"aluno","03":"amarelo","04":"america",
    "05":"aproveitar","06":"bala","07":"banco","08":"banheiro",
    "09":"barulho","10":"cinco","11":"conhecer","12":"espelho",
    "13":"esquina","14":"filho","15":"maca","16":"medo",
    "17":"ruim","18":"sapo","19":"vacina","20":"vontade",
}

# ── Prompt fonológico ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Você é um especialista em fonologia de língua de sinais (Libras).
Analise o vídeo de um sinal isolado de Libras e descreva seus parâmetros fonológicos.
Responda APENAS com um JSON válido, sem texto adicional."""

USER_PROMPT = """Analise este vídeo de um sinal de Libras e retorne um JSON com os seguintes campos:

{
  "mao_dominante": "direita" | "esquerda" | "ambas",
  "configuracao_mao_dominante": "descrição da forma da mão (ex: punho fechado, mão aberta, dedo indicador estendido, mão-B, etc.)",
  "configuracao_mao_nao_dominante": "descrição ou 'parada/ausente'",
  "ponto_articulacao": "onde a mão está no corpo (ex: frente ao peito, altura da cabeça, espaço neutro, queixo, etc.)",
  "movimento": "descrição do movimento (ex: extensão frontal, circular para baixo, batida lateral, sem movimento, etc.)",
  "orientacao_palma": "direção da palma (ex: para frente, para baixo, para cima, para o corpo, etc.)",
  "expressao_facial": "descrição do rosto (ex: neutro, sobrancelhas levantadas, boca aberta, bochechas infladas, etc.)",
  "contato_corporal": "sim/não e onde (ex: toca o queixo, não há contato, etc.)",
  "observacoes": "qualquer detalhe relevante para identificar o sinal"
}"""

# ── Carregamento do modelo ───────────────────────────────────────────────────
def load_model():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig

    print(f"Carregando {MODEL_ID}...", flush=True)
    print(f"  4-bit quantização: {USE_4BIT}", flush=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if USE_4BIT else None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16 if not USE_4BIT else None,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "eager",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )
    print("  Modelo carregado.", flush=True)
    return model, processor


def _has_flash_attn():
    try:
        import flash_attn
        return True
    except ImportError:
        return False


# ── Extração de frames do vídeo ──────────────────────────────────────────────
def sample_video_frames(video_path: Path, n_frames: int = MAX_FRAMES) -> list:
    """Retorna lista de PIL Images amostradas uniformemente do vídeo."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


# ── Anotação de um único vídeo ───────────────────────────────────────────────
def annotate_video(video_path: Path, model, processor) -> dict | None:
    from qwen_vl_utils import process_vision_info

    frames = sample_video_frames(video_path)
    if not frames:
        return None

    # Monta mensagem multimodal: imagens + prompt
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": USER_PROMPT})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    text_out  = processor.decode(generated, skip_special_tokens=True)

    # Extrai JSON da resposta
    return parse_json_response(text_out)


def parse_json_response(text: str) -> dict | None:
    """Extrai e parseia o primeiro bloco JSON da resposta do modelo."""
    # Tenta direto
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Tenta achar bloco ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Tenta achar { ... } genérico
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return {"raw_response": text, "parse_error": True}


# ── Classe de um arquivo de vídeo ────────────────────────────────────────────
def class_from_path(video_path: Path) -> tuple[str, str] | None:
    """Retorna (class_id, class_name) a partir do caminho do vídeo."""
    fname = video_path.name
    # Formato Zenodo: 3-01Acontecer_1RGB.mp4
    m = re.match(r"^\d+-(\d{2})", fname)
    if m:
        cid = m.group(1)
        return cid, CLASSES.get(cid, "unknown")
    # Fallback: nome do diretório pai
    for part in video_path.parts:
        m = re.match(r"^(\d{2})[A-Z]", part)
        if m:
            cid = m.group(1)
            return cid, CLASSES.get(cid, "unknown")
    return None


# ── Agregação em dicionário fônico ───────────────────────────────────────────
ANNOTATION_FIELDS = [
    "mao_dominante", "configuracao_mao_dominante", "configuracao_mao_nao_dominante",
    "ponto_articulacao", "movimento", "orientacao_palma",
    "expressao_facial", "contato_corporal", "observacoes",
]

def aggregate_dictionary(raw_dir: Path) -> dict:
    """
    Para cada classe, lê todas as anotações individuais e produz
    uma entrada consensual no dicionário fônico.
    """
    dictionary = {}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        annotations = []
        for jfile in class_dir.glob("*.json"):
            with open(jfile) as f:
                ann = json.load(f)
            if not ann.get("parse_error"):
                annotations.append(ann)

        if not annotations:
            continue

        # Para cada campo: moda dos valores (mais frequente)
        entry = {"class": class_name, "n_samples": len(annotations)}
        for field in ANNOTATION_FIELDS:
            values = [a.get(field, "") for a in annotations if a.get(field)]
            if values:
                # Moda simples
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                entry[field] = most_common
                entry[f"{field}_variants"] = dict(Counter(values))

        dictionary[class_name] = entry
        print(f"  {class_name}: {len(annotations)} anotações agregadas", flush=True)

    return dictionary


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Coleta todos os vídeos disponíveis
    videos = sorted(VIDEO_DIR.rglob("*RGB.mp4"))
    print(f"{len(videos)} vídeos encontrados em {VIDEO_DIR}", flush=True)

    if not videos:
        print("ERRO: Nenhum vídeo encontrado. Execute download_zenodo.py primeiro.", flush=True)
        return

    # Carrega modelo
    model, processor = load_model()

    # Anota cada vídeo
    ok, skip, err = 0, 0, 0
    for i, vpath in enumerate(videos):
        result = class_from_path(vpath)
        if result is None:
            print(f"  SKIP (classe desconhecida): {vpath.name}", flush=True)
            skip += 1
            continue

        class_id, class_name = result
        out_dir_class = OUT_DIR / "raw" / class_name
        out_dir_class.mkdir(parents=True, exist_ok=True)
        out_file = out_dir_class / (vpath.stem + ".json")

        if SKIP_EXISTING and out_file.exists():
            skip += 1
            continue

        t0 = time.time()
        ann = annotate_video(vpath, model, processor)
        elapsed = time.time() - t0

        if ann is None:
            print(f"  [{i+1}/{len(videos)}] ERRO (sem frames): {vpath.name}", flush=True)
            err += 1
            continue

        ann["_meta"] = {
            "filename": vpath.name,
            "class_id": class_id,
            "class_name": class_name,
            "elapsed_s": round(elapsed, 1),
        }

        with open(out_file, "w") as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

        ok += 1
        if ok % 10 == 0 or ok == 1:
            print(f"  [{i+1}/{len(videos)}] {class_name} — {elapsed:.1f}s — ok={ok}", flush=True)

    print(f"\nAnotação concluída: {ok} ok | {skip} skip | {err} erros", flush=True)

    # Agrega dicionário
    print("\nAgregando dicionário fônico...", flush=True)
    dictionary = aggregate_dictionary(OUT_DIR / "raw")
    with open(DICT_FILE, "w") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"Dicionário salvo: {DICT_FILE} ({len(dictionary)} classes)", flush=True)


if __name__ == "__main__":
    main()
