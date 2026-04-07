# Libras Wild — Anotação e Reconhecimento de Sinais

Ferramenta de anotação colaborativa para construir uma base de dados de Libras
capturada em vídeos reais ("in the wild").

## 🌐 Ferramenta de Anotação Online

494 segmentos de sinais detectados automaticamente aguardam validação.

---

## Fluxo de Anotação

```
Anotador abre o site
    │
    ▼
Assiste ao clipe (YouTube embed, timestamps automáticos)
    │
    ▼
Valida: é sinal válido? Qual sinal? Parâmetros fonológicos?
    │
    ▼
Salva localmente (IndexedDB) → Exporta JSON
    │
    ▼
Envia o JSON via Issues ou PR para este repositório
    │
    ▼
export_wild_dataset.py mescla anotações → processed_data_wild.pkl
```

---

## Exportar dataset após coleta

```bash
source env/bin/activate
python3 src/export_wild_dataset.py \
    --annotations results/annotations_*.json \
    --landmarks   results/youtube/-ZDkdbPqUZg/landmarks.pkl \
    --segments    results/youtube/-ZDkdbPqUZg/segments.json \
    --output      data/processed_data_wild.pkl
```

---

## Adicionar novos vídeos à fila de anotação

```bash
# 1. Roda pipeline no novo vídeo
source env/bin/activate
python3 -c "
from src.youtube_pipeline import run_pipeline
run_pipeline('https://youtu.be/VIDEO_ID')
"

# 2. Atualiza docs/data/segments.json
python3 src/prepare_annotation_data.py

# 3. Commit e push → deploy automático
git add docs/data/
git commit -m 'data: adiciona segmentos de VIDEO_ID'
git push
```

