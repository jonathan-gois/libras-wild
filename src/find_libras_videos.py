"""
find_libras_videos.py — Busca vídeos Libras no YouTube e salva URLs candidatas.

Usa yt-dlp para busca (sem API key).
Filtra por duração (2-15 min) e evita duplicatas.

Uso:
    python3 src/find_libras_videos.py --queries queries.txt --out dataset/candidate_urls.txt
    python3 src/find_libras_videos.py --n 50
"""

import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

DEFAULT_QUERIES = [
    "Libras vocabulário sinais básicos",
    "aprender Libras cotidiano",
    "Libras diálogo surdo",
    "Libras palavras do dia a dia",
    "intérprete Libras palestra",
    "INES Libras sinais",
    "Libras curso básico aula",
    "comunicação Libras surdo ouvinte",
]


def search_yt(query: str, n: int = 20) -> list[dict]:
    cmd = [
        "yt-dlp",
        f"ytsearch{n}:{query}",
        "--flat-playlist",
        "--print-json",
        "--no-warnings",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    videos = []
    for line in result.stdout.splitlines():
        try:
            info = json.loads(line)
            dur = info.get("duration") or 0
            if 90 <= dur <= 900:   # 1.5 min a 15 min
                videos.append({
                    "id":       info.get("id", ""),
                    "url":      f"https://www.youtube.com/watch?v={info.get('id','')}",
                    "title":    info.get("title", ""),
                    "duration": dur,
                    "query":    query,
                })
        except Exception:
            continue
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=str, help="Arquivo TXT com queries (uma por linha)")
    ap.add_argument("--n",       type=int, default=15, help="Resultados por query")
    ap.add_argument("--out",     type=str, default="dataset/candidate_urls.txt")
    args = ap.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [l.strip() for l in open(args.queries) if l.strip()]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Carrega URLs já conhecidas
    existing = set()
    if out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if line and not line.startswith("#"):
                existing.add(line.split()[0])

    found = []
    for q in queries:
        print(f"Buscando: {q!r}...", flush=True)
        results = search_yt(q, args.n)
        new = [r for r in results if r["url"] not in existing]
        found.extend(new)
        for r in new:
            existing.add(r["url"])
        print(f"  {len(new)} novos ({len(results)} totais)", flush=True)

    if not found:
        print("Nenhum novo vídeo encontrado.")
        return

    with open(out_path, "a") as f:
        f.write(f"\n# Busca automática — {len(found)} vídeos\n")
        for r in found:
            f.write(f"{r['url']}  # {r['duration']}s — {r['title'][:60]}\n")

    print(f"\n{len(found)} URLs adicionadas → {out_path}")
    print("Para processar:")
    print(f"  python3 src/wild_pipeline.py --urls {out_path} --out dataset/wild")


if __name__ == "__main__":
    main()
