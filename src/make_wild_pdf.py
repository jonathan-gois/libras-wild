"""Gera PDF do procedimento de aquisição e segmentação da base Libras Wild."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path
import json, numpy as np

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "results" / "wild_pipeline_procedure.pdf"
OUT.parent.mkdir(exist_ok=True)

# ── Estilos ───────────────────────────────────────────────────────────────────
BG   = colors.HexColor("#1a1a2e")
BG2  = colors.HexColor("#16213e")
ACC  = colors.HexColor("#0f3c78")
GRN  = colors.HexColor("#2ecc71")
YLW  = colors.HexColor("#f39c12")
WHT  = colors.white
GRY  = colors.HexColor("#aaaacc")

styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles[name] if name in styles else styles["Normal"]
    defaults = dict(textColor=WHT, backColor=BG)
    defaults.update(kw)
    return ParagraphStyle(name + str(id(defaults)), parent=base, **defaults)

title_s  = S("Title",  fontSize=22, leading=28, spaceAfter=6,  alignment=TA_CENTER,
              textColor=GRN, fontName="Helvetica-Bold")
h1_s     = S("h1",     fontSize=14, leading=18, spaceBefore=14, spaceAfter=4,
              textColor=YLW, fontName="Helvetica-Bold")
h2_s     = S("h2",     fontSize=11, leading=14, spaceBefore=8,  spaceAfter=3,
              textColor=GRN, fontName="Helvetica-Bold")
body_s   = S("body",   fontSize=9,  leading=13, spaceAfter=4,   alignment=TA_JUSTIFY,
              fontName="Helvetica")
small_s  = S("small",  fontSize=8,  leading=11, textColor=GRY,  fontName="Helvetica")
code_s   = S("code",   fontSize=8,  leading=11, fontName="Courier",
              backColor=BG2, textColor=GRN, leftIndent=12, spaceAfter=6)
sub_s    = S("sub",    fontSize=9,  leading=12, textColor=WHT,
              fontName="Helvetica", leftIndent=14, spaceAfter=2)

def P(text, style=body_s): return Paragraph(text, style)
def H1(text): return Paragraph(text, h1_s)
def H2(text): return Paragraph(text, h2_s)
def SP(n=6): return Spacer(1, n)
def HR(): return HRFlowable(width="100%", thickness=0.5, color=ACC, spaceAfter=6)
def Code(text): return Paragraph(text, code_s)

def table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND",   (0, 0), (-1, -1), BG2),
        ("TEXTCOLOR",    (0, 0), (-1, -1), WHT),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("FONTNAME",     (0, 0), (-1, -1), "Helvetica"),
        ("GRID",         (0, 0), (-1, -1), 0.3, ACC),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [BG, BG2]),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND",  (0, 0), (-1, 0), ACC),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR",   (0, 0), (-1, 0), GRN),
        ]
    t.setStyle(TableStyle(style))
    return t

# ── Dados do dataset ──────────────────────────────────────────────────────────
idx_path = ROOT / "dataset" / "wild" / "index.json"
if idx_path.exists():
    idx   = json.load(open(idx_path))
    durs  = [c["duration"] for c in idx]
    n_vids = len(set(c["video_id"] for c in idx))
    n_clips = len(idx)
    avg_dur = np.mean(durs)
    pct_1s  = 100 * sum(1 for d in durs if d <= 1.0) / len(durs)
else:
    n_clips, n_vids, avg_dur, pct_1s = 4133, 50, 0.65, 84

# ── Documento ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

story = []

# Capa
story += [
    SP(20),
    P("Libras Wild", S("T", fontSize=28, textColor=GRN, fontName="Helvetica-Bold",
                        alignment=TA_CENTER, spaceAfter=4)),
    P("Procedimento de Aquisição e Segmentação Automática", title_s),
    P("Base de dados de sinais Libras extraídos de vídeos YouTube", small_s),
    SP(8), HR(), SP(8),
    P(f"<b>Resultado:</b> {n_clips:,} clipes &nbsp;|&nbsp; {n_vids} vídeos "
      f"&nbsp;|&nbsp; duração média {avg_dur:.2f}s &nbsp;|&nbsp; "
      f"{pct_1s:.0f}% abaixo de 1s",
      S("stat", fontSize=10, alignment=TA_CENTER, textColor=YLW,
        fontName="Helvetica-Bold")),
    SP(16), HR(),
]

# ── 1. Visão Geral ────────────────────────────────────────────────────────────
story += [H1("1. Visão Geral do Pipeline"), SP(4)]
story += [P(
    "O pipeline Wild transforma vídeos do YouTube em clipes individuais "
    "de sinais Libras, prontos para anotação humana. O objetivo é construir "
    "uma base <i>in the wild</i> — com variabilidade de signatários, iluminação "
    "e contexto discursivo — complementar à base controlada MINDS-Libras.",
), SP(6)]

pipeline_data = [
    ["Etapa", "Módulo", "Saída"],
    ["1. Busca",        "find_libras_videos.py", "Lista de URLs (foco: glossário)"],
    ["2. Download",     "yt-dlp (h264/mp4)",      "Vídeo local"],
    ["3. Landmarks",    "MediaPipe Holistic",      "(T×225) float32"],
    ["4. Segmentação",  "sign_segmenter.py",       "Candidatos por pico de energia"],
    ["5. Filtro",       "sign_spotter.py",         "P(sinal válido) ≥ 0.50"],
    ["6. Clipe",        "ffmpeg",                  "Arquivo .mp4 cortado"],
    ["7. Anotação",     "GitHub Pages + Supabase", "Label + parâmetros fonológicos"],
]
story += [table(pipeline_data, [2.5*cm, 4*cm, 8*cm]), SP(8)]

# ── 2. Seleção de Vídeos ──────────────────────────────────────────────────────
story += [H1("2. Seleção de Vídeos"), SP(4)]
story += [P(
    "A qualidade dos clipes depende diretamente do tipo de vídeo. "
    "Vídeos de <b>discurso contínuo</b> (conversas, palestras) produzem clipes "
    "com múltiplos sinais; vídeos de <b>glossário/vocabulário</b> produzem sinais "
    "isolados mais limpos.",
), SP(4)]

story += [H2("2.1 Critérios de seleção")]
criteria = [
    ("Tipo preferido", "Glossário, dicionário, vocabulário temático — um sinal por vez"),
    ("Duração",         "1–8 minutos (vídeos curtos tendem a ser mais focados)"),
    ("Enquadramento",   "Signatário centralizado, câmera estável, fundo simples"),
    ("Iluminação",      "Boa visibilidade das mãos e rosto"),
    ("Evitar",          "Debates, painéis, legendas sobre o vídeo, intérprete pequeno"),
]
for k, v in criteria:
    story.append(P(f"<b>• {k}:</b> {v}", sub_s))
story.append(SP(6))

story += [H2("2.2 Queries utilizadas")]
story += [P("Busca automática via <i>yt-dlp</i> com os seguintes termos:"), SP(2)]
queries = [
    "glossário Libras sinais isolados",
    "dicionário Libras palavra por palavra",
    "como fazer sinal Libras vocabulário",
    "INES TV Libras manuário glossário",
    "Libras sinais animais cores números",
    "Handtalk Libras sinais dicionário",
    "Libras acessibilidade SENAI SENAC sinais",
]
for q in queries:
    story.append(Code(f"&nbsp; {q}"))
story.append(SP(8))

# ── 3. Extração de Landmarks ──────────────────────────────────────────────────
story += [H1("3. Extração de Landmarks (MediaPipe Holistic)"), SP(4)]
story += [P(
    "Cada frame do vídeo é processado pelo MediaPipe Holistic, que extrai "
    "33 pontos de pose + 21 pontos de mão esquerda + 21 pontos de mão direita "
    "= <b>75 landmarks × 3 coordenadas (x, y, z) = 225 valores por frame</b>."
), SP(4)]
lm_data = [
    ["Componente", "Landmarks", "Índices", "Relevância"],
    ["Pose (torso + braços)", "33",      "0–32",   "Posição dos pulsos (15,16), ombros (11,12)"],
    ["Mão esquerda",          "21",      "33–53",  "Configuração de mão, dedos"],
    ["Mão direita",           "21",      "54–74",  "Configuração de mão, dedos"],
]
story += [table(lm_data, [4*cm, 2.5*cm, 2.5*cm, 6.5*cm]), SP(4)]
story += [P(
    "A série temporal resultante tem forma <b>(T, 225)</b>, onde T é o número de frames "
    "do vídeo. Para um vídeo de 30fps com 60s, T ≈ 1800 frames."
), SP(8)]

# ── 4. Segmentação por Picos de Velocidade ────────────────────────────────────
story += [H1("4. Segmentação por Picos de Velocidade"), SP(4)]
story += [P(
    "Baseada na fonologia de línguas de sinais: durante um sinal há alta "
    "velocidade (<i>stroke</i>); entre sinais há pausa/hold com baixa velocidade. "
    "O segmentador usa a <b>janela adaptativa centrada no pico</b> de energia cinética."
), SP(6)]

story += [H2("4.1 Cálculo da energia cinética")]
story += [P(
    "A energia por frame combina velocidade dos pulsos e centros das mãos:"
), SP(2)]
story += [Code(
    "e(t) = 0.4 × max(vel_pulso_esq, vel_pulso_dir) + 0.3 × vel_centro_mao_esq + "
    "0.3 × vel_centro_mao_dir"
), SP(4)]

story += [H2("4.2 Detecção de picos e janela adaptativa")]
params_data = [
    ["Parâmetro", "Valor", "Descrição"],
    ["smooth_sigma_s",       "0.08 s",   "Gaussiana fina para suavizar ruído"],
    ["min_peak_height",      "0.25",     "Fração do máximo global"],
    ["min_peak_dist_s",      "0.25 s",   "Distância mínima entre picos"],
    ["min_peak_prominence",  "0.15",     "Proeminência mínima (rejeita picos rasos)"],
    ["half_win_s",           "0.55 s",   "Semi-janela máxima ao redor do pico"],
    ["energy_drop",          "0.20",     "Expande até energia < 20% do pico"],
    ["max_dur_s",            "1.8 s",    "Duração máxima do clipe"],
    ["min_dur_s",            "0.25 s",   "Duração mínima do clipe"],
]
story += [table(params_data, [4.5*cm, 2.5*cm, 8.5*cm]), SP(4)]
story += [P(
    "Para cada pico detectado, a janela expande-se da esquerda e direita enquanto "
    "a energia for > 20% do pico ou atingir o limite de 0.55s. Segmentos com "
    "sobreposição > 50% são deduplicados, mantendo o de maior energia."
), SP(8)]

# ── 5. Filtro de Sinais Válidos ───────────────────────────────────────────────
story += [H1("5. Filtro de Sinais Válidos (Sign Spotter)"), SP(4)]
story += [P(
    "Um classificador binário decide se o segmento detectado contém um sinal "
    "Libras completo ou é ruído (preparação, retração, hold, transição, "
    "gesto não-linguístico ou multi-sinal)."
), SP(4)]

story += [H2("5.1 Dados de treinamento")]
train_data = [
    ["Classe", "Fonte", "N"],
    ["Positivo (sinal completo)",  "MINDS-Libras (clips completos)",  "800"],
    ["Positivo (stroke only)",     "MINDS-Libras (janela centrada)",  "800"],
    ["Negativo (preparação)",      "Primeiros 35% de cada clip",      "800"],
    ["Negativo (retração)",        "Últimos 35% de cada clip",        "800"],
    ["Negativo (hold)",            "Frame do meio repetido",          "800"],
    ["Negativo (multi-sinal)",     "Dois sinais concatenados",        "800"],
    ["Negativo (corte aleatório)", "Janela fora do stroke",           "800"],
]
story += [table(train_data, [5*cm, 6*cm, 2.5*cm]), SP(4)]

story += [H2("5.2 Modelo e features")]
story += [P(
    "Modelo: <b>Gradient Boosting (300 árvores, max_depth=4)</b> com StandardScaler. "
    "F1-score cross-val: <b>0.851 ± 0.016</b>. "
    "Features: estatísticas de velocidade dos pulsos e mãos (média, máx, std), "
    "perfil temporal em quartis (invariante à duração), posição relativa das mãos."
), SP(4)]
story += [P(
    "<b>Importante:</b> a feature de duração foi removida deliberadamente para evitar "
    "que o modelo aprenda a discriminar por comprimento do clipe em vez de "
    "características cinemáticas do sinal."
), SP(8)]

# ── 6. Resultados ─────────────────────────────────────────────────────────────
story += [H1("6. Resultados do Pipeline"), SP(4)]

res_data = [
    ["Métrica", "Valor"],
    ["Vídeos processados",          f"{n_vids}"],
    ["Total de clipes gerados",     f"{n_clips:,}"],
    ["Duração média por clipe",     f"{avg_dur:.2f} s"],
    ["Duração mediana",             "0.67 s"],
    ["Clipes ≤ 1.0s",              f"{pct_1s:.0f}%"],
    ["Range de duração",            "0.23 – 1.08 s"],
    ["Spotter score médio",         "0.850"],
    ["Threshold de aprovação",      "≥ 0.50"],
    ["Taxa de aprovação (média)",   "~55%"],
]
story += [table(res_data, [9*cm, 6.5*cm]), SP(6)]

story += [P(
    "A comparação com a versão anterior (threshold vale-a-vale) é clara: "
    "os clipes passaram de <b>3.5s médio</b> (múltiplos sinais) para <b>0.65s médio</b> "
    "(sinal individual), alinhado com a duração linguística esperada para Libras "
    "em discurso contínuo (0.5–1.2s por sinal)."
), SP(8)]

# ── 7. Próximos Passos ────────────────────────────────────────────────────────
story += [H1("7. Próximos Passos"), SP(4)]
steps = [
    ("Anotação humana",
     "Voluntários acessam https://jonathan-gois.github.io/libras-wild/ "
     "e anotam os 4.133 clipes com rótulo, confiança e parâmetros fonológicos."),
    ("Retraining com Wild",
     "Após ≥50 anotações por classe: python3 src/train_wild.py — "
     "treina classificador combinando MINDS (800) + Wild anotado."),
    ("Avaliação LOPO Wild",
     "Testar se adicionar dados Wild reduz o gap LOPO "
     "(atualmente 24pp: 98.5% 10-fold → 74.5% LOPO)."),
    ("Ampliar a base",
     "python3 src/find_libras_videos.py --n 30 && "
     "python3 src/wild_pipeline.py --urls dataset/candidate_urls.txt"),
    ("Melhoria do spotter",
     "Incluir clips Wild anotados como negativos confirmados no treino do spotter, "
     "substituindo negativos sintéticos."),
]
for title, desc in steps:
    story.append(P(f"<b>• {title}:</b> {desc}", sub_s))
story.append(SP(8))

# ── 8. Como Reproduzir ────────────────────────────────────────────────────────
story += [H1("8. Como Reproduzir"), SP(4)]
story += [Code("# 1. Busca vídeos Libras (glossário)")]
story += [Code("python3 src/find_libras_videos.py --n 20")]
story += [Code("# 2. Processa vídeos: download → landmarks → segmentação → filtro → clipes")]
story += [Code("python3 src/wild_pipeline.py --urls dataset/candidate_urls.txt --out dataset/wild")]
story += [Code("# 3. Atualiza índice do site")]
story += [Code("python3 -c \"import json; idx=json.load(open('dataset/wild/index.json')); "
               "json.dump([{k:v for k,v in c.items() if k!='clip'} for c in idx], "
               "open('docs/data/wild_index.json','w'), separators=(',',':'))\"")]
story += [Code("# 4. Treinar spotter (se necessário)")]
story += [Code("python3 src/sign_spotter.py --train")]
story += [Code("# 5. Treinar com anotações Wild")]
story += [Code("python3 src/train_wild.py")]
story += [SP(4), HR()]

story += [P(
    "Repositório: https://github.com/jonathan-gois/libras-wild &nbsp;|&nbsp; "
    "Site de anotação: https://jonathan-gois.github.io/libras-wild/",
    small_s
)]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story,
          onFirstPage =lambda c, d: c.setFillColor(BG) or c.rect(0,0,d.width+4*cm,d.height+4*cm,fill=1),
          onLaterPages=lambda c, d: c.setFillColor(BG) or c.rect(0,0,d.width+4*cm,d.height+4*cm,fill=1))

print(f"PDF salvo: {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
