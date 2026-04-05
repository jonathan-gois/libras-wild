"""
Gera PDF dos resultados do projeto Libras 2026.
Layout: 1 slide = 1 página A4 paisagem (297 × 210 mm).
Usa apenas primitivas ReportLab — sem sobreposição.
"""
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import os

OUT = "results/libras2026_resultados.pdf"

PW, PH = landscape(A4)   # 297 × 210 mm  → pts: ~841 × 595

# ── Paleta ───────────────────────────────────────────────────────────────────
C_BG      = colors.HexColor("#1A1A2E")
C_HEADER  = colors.HexColor("#0F3C78")
C_MID     = colors.HexColor("#16213E")
C_CARD    = colors.HexColor("#0F2D55")
C_GREEN   = colors.HexColor("#2ECC71")
C_YELLOW  = colors.HexColor("#F39C12")
C_RED     = colors.HexColor("#E74C3C")
C_WHITE   = colors.HexColor("#FFFFFF")
C_LGRAY   = colors.HexColor("#CCCCCC")
C_DGRAY   = colors.HexColor("#888899")
C_ROW_A   = colors.HexColor("#162140")
C_ROW_B   = colors.HexColor("#1E2A4A")
C_WIN     = colors.HexColor("#072A0F")
C_WARN    = colors.HexColor("#3A1A05")


# ── Helpers ──────────────────────────────────────────────────────────────────
def new_page(c):
    c.setFillColor(C_BG)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)

def header_bar(c, title, bg=C_HEADER, fg=C_WHITE, size=22):
    h = 32
    c.setFillColor(bg)
    c.rect(0, PH - h, PW, h, fill=1, stroke=0)
    c.setFillColor(fg)
    c.setFont("Helvetica-Bold", size)
    c.drawString(14, PH - h + 8, title)

def hline(c, y, x0=14, x1=None, color=C_HEADER, width=1):
    if x1 is None:
        x1 = PW - 14
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x0, y, x1, y)

def filled_rect(c, x, y, w, h, color):
    """y = baseline (bottom of rect) em pt, coordenadas ReportLab."""
    c.setFillColor(color)
    c.rect(x, y, w, h, fill=1, stroke=0)

def text(c, txt, x, y, size=11, color=C_WHITE, bold=False, italic=False,
         align="left", max_width=None):
    """Escreve uma linha de texto. y = baseline."""
    if bold and italic:
        font = "Helvetica-BoldOblique"
    elif bold:
        font = "Helvetica-Bold"
    elif italic:
        font = "Helvetica-Oblique"
    else:
        font = "Helvetica"
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "center" and max_width:
        c.drawCentredString(x + max_width / 2, y, txt)
    elif align == "right" and max_width:
        c.drawRightString(x + max_width, y, txt)
    else:
        c.drawString(x, y, txt)

def text_wrap(c, txt, x, y, max_width, size=11, color=C_WHITE,
              bold=False, italic=False, line_gap=4):
    """
    Escreve texto com quebra de linha automática.
    Retorna y final (abaixo do último texto).
    """
    font = "Helvetica-Bold" if bold else "Helvetica"
    if italic:
        font = "Helvetica-BoldOblique" if bold else "Helvetica-Oblique"
    lines = simpleSplit(txt, font, size, max_width)
    c.setFont(font, size)
    c.setFillColor(color)
    for line in lines:
        c.drawString(x, y, line)
        y -= (size + line_gap)
    return y

def table(c, data, x, y, col_widths, row_h=18,
          header_bg=C_HEADER, row_bg_a=C_ROW_A, row_bg_b=C_ROW_B,
          font_size=10, win_rows=None):
    """
    Desenha tabela. y = topo da tabela (pt, ReportLab).
    Retorna y final (abaixo da última linha).
    win_rows: set de row indices para highlight verde.
    """
    win_rows = win_rows or set()
    cy = y
    for ri, row in enumerate(data):
        bg = header_bg if ri == 0 else (row_bg_a if ri % 2 == 1 else row_bg_b)
        if ri in win_rows:
            bg = C_WIN
        cx = x
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            filled_rect(c, cx, cy - row_h, cw, row_h, bg)
            bold = (ri == 0)
            col = C_GREEN if ri in win_rows else (C_WHITE if ri == 0 else C_LGRAY)
            if "⚠" in cell:
                col = C_YELLOW
            pad = 4
            c.setFont("Helvetica-Bold" if bold else "Helvetica", font_size)
            c.setFillColor(col)
            # trunca se não couber
            available = cw - 2 * pad
            while c.stringWidth(cell, "Helvetica-Bold" if bold else "Helvetica", font_size) > available and len(cell) > 3:
                cell = cell[:-2] + "…"
            c.drawString(cx + pad, cy - row_h + 5, cell)
            cx += cw
        cy -= row_h
    return cy  # y abaixo da última linha

def bullet(c, items, x, y, size=11, gap=6, color=C_LGRAY,
           marker="▸  ", bold_marker=False, max_width=None):
    """Lista de bullets. Retorna y final."""
    for item in items:
        if max_width:
            font = "Helvetica"
            lines = simpleSplit(marker + item, font, size, max_width)
            c.setFont(font, size)
            c.setFillColor(color)
            for li, line in enumerate(lines):
                c.drawString(x if li == 0 else x + c.stringWidth(marker, font, size),
                             y, line if li == 0 else line)
                y -= (size + gap)
        else:
            text(c, marker + item, x, y, size=size, color=color)
            y -= (size + gap)
    return y

def footer(c, page_n, total=8):
    c.setFont("Helvetica", 8)
    c.setFillColor(C_DGRAY)
    c.drawString(14, 8, "Projeto Libras 2026  ·  MINDS-Libras")
    c.drawRightString(PW - 14, 8, f"{page_n} / {total}")


# ════════════════════════════════════════════════════════════════════════════
c = canvas.Canvas(OUT, pagesize=landscape(A4))
c.setTitle("Reconhecimento de Libras — Resultados 2026")

# ── Página 1: Capa ───────────────────────────────────────────────────────────
new_page(c)
filled_rect(c, 0, PH - 70, PW, 70, C_HEADER)
c.setFillColor(C_WHITE)
c.setFont("Helvetica-Bold", 28)
c.drawCentredString(PW/2, PH - 42, "Reconhecimento de Sinais em Libras")
c.setFont("Helvetica", 14)
c.setFillColor(C_LGRAY)
c.drawCentredString(PW/2, PH - 62, "Dataset MINDS-Libras  ·  800 amostras  ·  20 classes")

# 3 cards
card_data = [
    ("98.50%", "MELHOR 10-FOLD CV",     C_GREEN),
    ("74.50%", "GENERALIZAÇÃO LOPO",    C_YELLOW),
    ("+2.5 pp","SUPERA ESTADO DA ARTE", C_GREEN),
]
cw, ch = 200, 90
cx0 = (PW - 3*cw - 2*20) / 2
for i, (val, lbl, col) in enumerate(card_data):
    bx = cx0 + i * (cw + 20)
    by = PH - 70 - 20 - ch
    filled_rect(c, bx, by, cw, ch, colors.HexColor("#0A2555"))
    c.setFont("Helvetica-Bold", 36)
    c.setFillColor(col)
    c.drawCentredString(bx + cw/2, by + 44, val)
    c.setFont("Helvetica", 9)
    c.setFillColor(C_DGRAY)
    c.drawCentredString(bx + cw/2, by + 26, lbl)

# subtitulo
ty = PH - 70 - 20 - ch - 28
c.setFont("Helvetica-Oblique", 12)
c.setFillColor(C_LGRAY)
c.drawCentredString(PW/2, ty, "ExtraTrees-1000  ·  GEI + FFT Sliding Window + Kinematic Stats  ·  11.172 features")
ty -= 18
c.setFont("Helvetica", 10)
c.setFillColor(C_DGRAY)
c.drawCentredString(PW/2, ty, "Hardware: CPU  ·  Sem GPU  ·  Sem vídeo RGB  ·  Apenas skeleton MediaPipe")

footer(c, 1)
c.showPage()


# ── Página 2: Contexto & Dataset ─────────────────────────────────────────────
new_page(c)
header_bar(c, "Contexto & Dataset")
hline(c, PH - 34)

# Coluna esquerda
lx, rx = 14, PW/2 + 10
cw_col = PW/2 - 24

ty = PH - 50
text(c, "MINDS-Libras", lx, ty, size=14, bold=True, color=C_GREEN)
ty -= 18
items_ds = [
    "800 amostras (original) / 1180 com Zenodo (+4 sinalizadores)",
    "20 sinais isolados de Libras",
    "10 sinalizadores × 40 amostras/classe",
    "MediaPipe Holistic: 75 landmarks × 3 coords = 225 features/frame",
]
ty = bullet(c, items_ds, lx+6, ty, size=11, gap=5, max_width=cw_col - 10)

ty -= 12
text(c, "20 Classes", lx, ty, size=13, bold=True, color=C_YELLOW)
ty -= 16
classes = ["acontecer","aluno","amarelo","america","aproveitar",
           "bala","banco","banheiro","barulho","cinco",
           "conhecer","espelho","esquina","filho","maca",
           "medo","ruim","sapo","vacina","vontade"]
cols_n = 4
cell_w = (cw_col) / cols_n
cell_h = 18
for i, cls in enumerate(classes):
    ci = i % cols_n
    ri = i // cols_n
    bx = lx + ci * cell_w
    by = ty - (ri + 1) * (cell_h + 3)
    filled_rect(c, bx, by, cell_w - 2, cell_h, C_CARD)
    c.setFont("Helvetica", 9)
    c.setFillColor(C_LGRAY)
    c.drawCentredString(bx + (cell_w - 2)/2, by + 5, cls)

# Coluna direita: pipeline
ty2 = PH - 50
text(c, "Pipeline de Extração", rx, ty2, size=14, bold=True, color=C_GREEN)
ty2 -= 18
steps = [
    ("1", "Vídeo MP4",       "MediaPipe Holistic extrai 75 landmarks/frame"),
    ("2", "Normalização",    "Centro: quadris  |  Escala: distância ombro-ombro"),
    ("3", "GEI  64×48px",    "Esqueleto 2D médio por pixel  →  3.072 dims"),
    ("4", "Kinematic Stats", "mean/std/max/min de vel e acel  →  1.800 dims"),
    ("5", "FFT Sliding",     "janela=16, passo=8, top-5 mag  →  6.300 dims"),
    ("6", "Concat + ET-1000","Total: 11.172 dims  →  ExtraTrees-1000"),
]
for num, title, desc in steps:
    filled_rect(c, rx, ty2 - 13, 22, 18, C_HEADER)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(C_WHITE)
    c.drawCentredString(rx + 11, ty2 - 8, num)
    text(c, title, rx + 28, ty2, size=12, bold=True, color=C_WHITE)
    ty2 -= 14
    text(c, desc, rx + 28, ty2, size=10, color=C_DGRAY)
    ty2 -= 18

footer(c, 2)
c.showPage()


# ── Página 3: Comparativo Geral ──────────────────────────────────────────────
new_page(c)
header_bar(c, "Comparativo — Estado da Arte vs Nossos Modelos")
hline(c, PH - 34)

data3 = [
    ["Método",                                        "Acurácia (10-fold)",  "Δ vs SotA (96%)"],
    ["Passos et al. — GEI + SVM",                     "84.66%",              "−11.3 pp"],
    ["Rego et al. — FFT+Kin+BiLSTM",                  "92.0%",               "−4.0 pp"],
    ["Rezende et al. — CNN 3D (RGB+Depth+Skel.)",      "93.3%",               "−2.7 pp"],
    ["De Castro / Arcanjo — 3D CNN / FastDTW",         "~96.0%",              "referência"],
    ["─── Nossos modelos ───────────────────────────", "──────────────",      "──────────"],
    ["BiLSTM multimodal (sem augmentação)",            "72.50% ±5.51%",       "−23.5 pp"],
    ["Transformer + mirror aug (invalidado)",          "88.75% ±4.51%",       "−7.3 pp"],
    ["ST-GCN + MobileNetV3 (v5)",                     "91.88% ±4.12%",       "−4.1 pp"],
    ["Transformer sem mirror (v4)",                   "94.62% ±5.59%",       "−1.4 pp"],
    ["ExtraTrees-1000 (1180 amostras / Zenodo)",       "96.78% ±2.03%",       "+0.8 pp"],
    ["ExtraTrees-1000 (800 amostras)  ← VENCEDOR",    "98.50% ±0.75%",       "+2.5 pp  ✓"],
]
col_w3 = [370, 140, 110]
table(c, data3, 14, PH - 40, col_w3, row_h=20,
      font_size=10, win_rows={11})

# LOPO warning
wy = PH - 40 - len(data3)*20 - 14
filled_rect(c, 14, wy - 20, PW - 28, 24, colors.HexColor("#3A2A05"))
c.setFont("Helvetica-Bold", 11)
c.setFillColor(C_YELLOW)
c.drawString(20, wy - 13, "⚠  LOPO (generalização real): 74.50% ±11.68%  —  gap de 24 pp em relação ao 10-fold CV")

footer(c, 3)
c.showPage()


# ── Página 4: Método Vencedor ────────────────────────────────────────────────
new_page(c)
filled_rect(c, 0, PH - 32, PW, 32, C_GREEN)
c.setFont("Helvetica-Bold", 22)
c.setFillColor(C_BG)
c.drawString(14, PH - 22, "Método Vencedor — ExtraTrees-1000")
hline(c, PH - 34, color=C_GREEN)

lx, rx = 14, PW/2 + 10
cw_col = PW/2 - 24

# Parâmetros
ty = PH - 50
text(c, "Parâmetros do Classificador", lx, ty, size=13, bold=True, color=C_GREEN)
ty -= 4
params = [
    ["Parâmetro",          "Valor"],
    ["n_estimators",       "1000"],
    ["criterion",          "gini"],
    ["max_features",       "sqrt"],
    ["min_samples_leaf",   "1"],
    ["bootstrap",          "False  (diferença chave vs RF)"],
    ["n_jobs",             "−1  (todos os núcleos)"],
    ["random_state",       "42"],
]
ty = table(c, params, lx, ty, [170, 200], row_h=18, font_size=10)

ty -= 14
text(c, "Por que supera redes neurais?", lx, ty, size=13, bold=True, color=C_GREEN)
ty -= 16
reasons = [
    "800 amostras / 20 classes = apenas 40 por classe",
    "Redes neurais: ~9M parâmetros → overfitting severo nesse regime",
    "ExtraTrees: ~100K parâmetros efetivos",
    "Sem PCA: lida nativamente com 11.172 dims",
    "Thresholds aleatórios → menor variância que Random Forest",
]
ty = bullet(c, reasons, lx+6, ty, size=10, gap=5, max_width=cw_col - 10)

# Coluna direita: importância
ty2 = PH - 50
text(c, "Importância das Features (Gini)", rx, ty2, size=13, bold=True, color=C_GREEN)
ty2 -= 20

bars = [
    ("FFT Sliding Window  (6.300 dims)", 74.95, C_GREEN),
    ("Kinematic Stats  (1.800 dims)",    13.32, C_YELLOW),
    ("GEI  (3.072 dims)",               11.74, C_RED),
]
bar_max_w = cw_col - 80
for lbl, pct, col in bars:
    text(c, lbl, rx, ty2, size=11, color=C_LGRAY)
    ty2 -= 14
    bw = bar_max_w * pct / 100
    filled_rect(c, rx, ty2 - 10, bw, 14, col)
    text(c, f"{pct:.1f}%", rx + bw + 6, ty2 - 2, size=11, bold=True, color=col)
    ty2 -= 26

ty2 -= 8
hline(c, ty2 + 4, x0=rx, color=C_DGRAY)
ty2 -= 8
text(c, "FFT sozinho: 98.12%  |  Combinação: 98.50%", rx, ty2, size=10,
     color=C_DGRAY, italic=True)
ty2 -= 16
text(c, "Variância: ±1.28% (FFT só)  →  ±0.75% (3 modalidades)", rx, ty2,
     size=10, color=C_DGRAY, italic=True)

ty2 -= 20
text(c, "Ablação completa (ET-1000)", rx, ty2, size=13, bold=True, color=C_YELLOW)
ty2 -= 4
abl = [
    ["Features",        "Acurácia",     "±std"],
    ["GEI only",        "52.00%",       "±4.00%"],
    ["KIN only",        "81.62%",       "±2.44%"],
    ["FFT only",        "98.12%",       "±1.28%"],
    ["GEI + KIN",       "81.62%",       "±5.39%"],
    ["GEI + FFT",       "98.00%",       "±1.50%"],
    ["KIN + FFT",       "98.00%",       "±1.27%"],
    ["GEI + KIN + FFT", "98.50%",       "±0.75%  ✓"],
]
table(c, abl, rx, ty2, [140, 90, 90], row_h=17, font_size=10, win_rows={7})

footer(c, 4)
c.showPage()


# ── Página 5: Mirror Augmentation ───────────────────────────────────────────
new_page(c)
filled_rect(c, 0, PH - 32, PW, 32, C_RED)
c.setFont("Helvetica-Bold", 22)
c.setFillColor(C_WHITE)
c.drawString(14, PH - 22, "Achado Crítico — Mirror Augmentation é Prejudicial")
hline(c, PH - 34, color=C_RED)

ty = PH - 50
text(c, "Hipótese inicial:", 14, ty, size=12, bold=True, color=C_LGRAY)
ty -= 16
filled_rect(c, 14, ty - 14, PW - 28, 20, colors.HexColor("#2A1010"))
c.setFont("Helvetica-Oblique", 11)
c.setFillColor(C_LGRAY)
c.drawString(20, ty - 7, '"Espelhar o sinal simula sinalizadores canhotos -> dobra o dataset -> melhora a generalizacao"')
ty -= 32

hline(c, ty + 6, color=C_DGRAY)
ty -= 14
text(c, "Resultado da analise k-NN no espaco FFT:", 14, ty, size=12, bold=True, color=C_YELLOW)
ty -= 18

findings = [
    ("94.9%",   "das amostras espelhadas tem o vizinho mais proximo em OUTRA classe",   C_RED),
    ("19 / 20", "classes tem seu espelho mapeado majoritariamente para outra classe",    C_RED),
    ("Excecao:", '"acontecer" - unico sinal simetrico (palmas frente a frente)',         C_LGRAY),
]
for val, desc, col in findings:
    filled_rect(c, 14, ty - 16, 72, 20, colors.HexColor("#4A1010"))
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(col)
    c.drawCentredString(14 + 36, ty - 5, val)
    text(c, desc, 96, ty - 5, size=11, color=C_LGRAY)
    ty -= 26

ty -= 10
hline(c, ty + 4, color=C_DGRAY)
ty -= 14
text(c, "Causa fonologica:", 14, ty, size=12, bold=True, color=C_YELLOW)
ty -= 16
ty = text_wrap(c,
    "Libras usa lateralidade como traco fonologico. Espelhar nao cria uma variante valida "
    "do mesmo sinal - gera confusao com outros sinais que partilham o mesmo envelope "
    "espacial mas no hemicorpo oposto.",
    14, ty, PW - 28, size=11, color=C_LGRAY)

ty -= 16
text(c, "Impacto medido:", 14, ty, size=12, bold=True, color=C_YELLOW)
ty -= 6
mirror_data = [
    ["Modelo",         "Augmentação",   "Acurácia",        "Δ"],
    ["v3 Transformer", "COM mirror",    "88.75% ±4.51%",   "baseline"],
    ["v4 Transformer", "SEM mirror",    "94.62% ±5.59%",   "+5.87 pp  ✓"],
]
table(c, mirror_data, 14, ty, [180, 130, 160, 100], row_h=20,
      font_size=11, win_rows={2})

footer(c, 5)
c.showPage()


# ── Página 6: LOPO ───────────────────────────────────────────────────────────
new_page(c)
filled_rect(c, 0, PH - 32, PW, 32, C_YELLOW)
c.setFont("Helvetica-Bold", 22)
c.setFillColor(C_BG)
c.drawString(14, PH - 22, "Generalização Real — Leave-One-Person-Out (LOPO)")
hline(c, PH - 34, color=C_YELLOW)

# Dois cards lado a lado
card_h, card_w = 90, (PW - 42) / 2
ty_card = PH - 40 - card_h

filled_rect(c, 14, ty_card, card_w, card_h, C_WIN)
c.setFont("Helvetica", 11)
c.setFillColor(C_LGRAY)
c.drawCentredString(14 + card_w/2, ty_card + card_h - 14, "10-fold CV")
c.setFont("Helvetica-Bold", 42)
c.setFillColor(C_GREEN)
c.drawCentredString(14 + card_w/2, ty_card + 30, "98.50%")
c.setFont("Helvetica", 9)
c.setFillColor(C_DGRAY)
c.drawCentredString(14 + card_w/2, ty_card + 14, "mesmo sinalizador no treino e no teste")

filled_rect(c, 28 + card_w, ty_card, card_w, card_h, C_WARN)
c.setFont("Helvetica", 11)
c.setFillColor(C_LGRAY)
c.drawCentredString(28 + card_w + card_w/2, ty_card + card_h - 14, "LOPO")
c.setFont("Helvetica-Bold", 42)
c.setFillColor(C_YELLOW)
c.drawCentredString(28 + card_w + card_w/2, ty_card + 30, "74.50%")
c.setFont("Helvetica", 9)
c.setFillColor(C_DGRAY)
c.drawCentredString(28 + card_w + card_w/2, ty_card + 14, "sinalizador nunca visto no treino")

# seta gap
mid_x = 14 + card_w + 7
c.setFont("Helvetica-Bold", 18)
c.setFillColor(C_RED)
c.drawCentredString(mid_x, ty_card + 45, "↓24pp")

ty = ty_card - 20

lx, rx = 14, PW/2 + 10
cw_col = PW/2 - 24

text(c, "Resultado por sinalizador (ET-1000)", lx, ty, size=12, bold=True, color=C_YELLOW)
ty -= 4
sdata = [
    ["Sinalizador",    "Acurácia"],
    ["Sinalizador 01", "77.5%"],
    ["Sinalizador 02", "88.0%"],
    ["Sinalizador 05", "76.5%"],
    ["Sinalizador 06", "82.5%"],
    ["Sinalizador 08", "75.0%"],
    ["Sinalizador 10", "51.0%  ⚠"],
    ["Média",          "74.50% ±11.68%"],
]
table(c, sdata, lx, ty, [180, 120], row_h=18, font_size=11, win_rows=set())

# Direita: interpretação
ty2 = ty
text(c, "Interpretação", rx, ty2, size=12, bold=True, color=C_YELLOW)
ty2 -= 16
ty2 = text_wrap(c,
    "O modelo aprende o estilo motor de cada sinalizador, "
    "não as propriedades fonológicas universais do sinal.",
    rx, ty2, cw_col, size=11, color=C_LGRAY)
ty2 -= 14
text(c, "Soluções:", rx, ty2, size=11, bold=True, color=C_LGRAY)
ty2 -= 14
sols = [
    "Mais sinalizadores (Zenodo: +4 → 1180 amostras)",
    "Normalização adversarial por sinalizador",
    "Features de velocidade relativa (invariância ao estilo)",
    "Data augmentation estilo-agnóstica",
]
bullet(c, sols, rx+6, ty2, size=10, gap=5, max_width=cw_col - 10)

footer(c, 6)
c.showPage()


# ── Página 7: Pipeline YouTube ───────────────────────────────────────────────
new_page(c)
header_bar(c, "Pipeline YouTube — Análise de Vídeo em Libras")
hline(c, PH - 34)

lx, rx = 14, PW/2 + 10
cw_col = PW/2 - 24

# Esquerda: pipeline
ty = PH - 50
text(c, "Vídeo: https://youtu.be/-ZDkdbPqUZg  (17 min, aula de Libras)",
     lx, ty, size=10, color=C_DGRAY, italic=True)
ty -= 20

steps_yt = [
    ("yt-dlp",      "Download URL  →  MP4 + SRT legendas"),
    ("MediaPipe",   "Extração landmarks: 30.775 frames, 30fps, ~1027s"),
    ("Sign Spotter","Energia vel. punho + threshold adaptativo  →  559 segmentos"),
    ("ET-1000",     "Classificação top-5 por segmento  (11.172 features)"),
    ("Alinhamento", "Overlap ≥30% com legendas  →  437 entradas geradas"),
]
step_h = 30
for num, (comp, desc) in enumerate(steps_yt):
    by = ty - (num * (step_h + 6))
    filled_rect(c, lx, by - step_h + 6, 24, step_h - 2, C_HEADER)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(C_WHITE)
    c.drawCentredString(lx + 12, by - step_h + 14, str(num+1))
    text(c, comp, lx + 30, by - 6, size=12, bold=True, color=C_WHITE)
    text(c, desc, lx + 30, by - 20, size=10, color=C_DGRAY)

ty -= len(steps_yt) * (step_h + 6) + 20

text(c, "Amostra do alinhamento legenda → sinal:", lx, ty, size=11, bold=True, color=C_YELLOW)
ty -= 14
samples = [
    ('[6s]  "menos hoje vai eu sou a professora"',        "-> vacina"),
    ('[10s] "fazer isso como que voce vai aprender"',     "-> vacina"),
    ('[18s] "sinalizando em Libras ao final desse"',      "-> vacina"),
    ('[41s] "Entao a gente vai comecar aqui"',            "-> aproveitar"),
    ('[48s] "sinaliza ai comigo a gente vai comecar"',    "-> acontecer"),
]
for sub, sign in samples:
    c.setFont("Helvetica", 9)
    c.setFillColor(C_LGRAY)
    c.drawString(lx + 6, ty, sub)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(C_GREEN)
    c.drawString(lx + 6 + c.stringWidth(sub, "Helvetica", 9) + 4, ty, sign)
    ty -= 13

# Direita: resultados
ty2 = PH - 50
text(c, "Top sinais detectados (437 alinhamentos)", rx, ty2, size=12, bold=True, color=C_GREEN)
ty2 -= 6
top_data = [
    ["Sinal",       "Ocorrências", "%"],
    ["vacina",      "156",         "35.8%"],
    ["acontecer",   "66",          "15.1%"],
    ["medo",        "61",          "14.0%"],
    ["aproveitar",  "60",          "13.8%"],
    ["esquina",     "58",          "13.3%"],
    ["america",     "16",          "3.7%"],
    ["outros",      "20",          "4.6%"],
]
table(c, top_data, rx, ty2, [120, 90, 80], row_h=18, font_size=11)
ty2 -= len(top_data) * 18 + 18

text(c, "Limitações:", rx, ty2, size=11, bold=True, color=C_YELLOW)
ty2 -= 14
lims = [
    "Modelo conhece apenas 20 sinais — vocabulário restrito",
    "Confianças baixas (13–21% top-1): maioria dos sinais fora do vocabulário",
    "Solução: threshold de rejeição + modelo com vocabulário maior",
]
bullet(c, lims, rx+6, ty2, size=10, gap=5, max_width=cw_col - 10)

footer(c, 7)
c.showPage()


# ── Página 8: Conclusões ─────────────────────────────────────────────────────
new_page(c)
header_bar(c, "Conclusões & Próximos Passos")
hline(c, PH - 34)

lx, rx = 14, PW/2 + 10
cw_col = PW/2 - 24

ty = PH - 50
text(c, "Conclusões", lx, ty, size=13, bold=True, color=C_GREEN)
ty -= 16
concl = [
    ("ExtraTrees-1000 supera o SotA em +2.5pp sem GPU e sem RGB", C_GREEN),
    ("FFT domina: 74.95% da importância Gini, 98.12% sozinho", C_GREEN),
    ("Mirror augmentation é prejudicial: −5.87pp (traço fonológico)", C_GREEN),
    ("Gap CV/LOPO: 24pp — modelo memoriza estilo do sinalizador", C_YELLOW),
    ("ST-GCN não supera Transformer simples com 800 amostras", C_YELLOW),
    ("1180 amostras (Zenodo): CV mais realista → 96.78% (↓1.72pp)", C_LGRAY),
]
for txt, col in concl:
    marker = "✓  " if col == C_GREEN else "⚠  "
    text(c, marker + txt, lx + 6, ty, size=11, color=col)
    ty -= 16

ty -= 8
text(c, "Próximos Passos", lx, ty, size=13, bold=True, color=C_YELLOW)
ty -= 16
nexts = [
    "LOPO com 1180 amostras (10 sinalizadores)",
    "Anotação fonológica com Qwen2-VL-7B: CM, PA, MOV, OR, ENM",
    "v6 PhoneticFusionModel + InfoNCE contrastive loss (CLIP-style)",
    "Normalização adversarial por sinalizador (reduzir gap LOPO)",
    "Threshold de rejeição out-of-vocabulary no pipeline YouTube",
    "Expansão do vocabulário para além de 20 classes",
]
bullet(c, nexts, lx + 6, ty, size=11, gap=6, marker="→  ",
       color=C_LGRAY, max_width=cw_col - 10)

# Tabela resumo direita
ty2 = PH - 50
text(c, "Resumo Final", rx, ty2, size=13, bold=True, color=C_GREEN)
ty2 -= 6
summary = [
    ["Modelo",                        "Acurácia"],
    ["BiLSTM (sem aug)",              "72.50% ±5.51%"],
    ["BiLSTM (com aug)",              "79.50% ±3.46%"],
    ["Transformer + mirror (v3)",     "88.75% ±4.51%"],
    ["ST-GCN + MobileNetV3 (v5)",     "91.88% ±4.12%"],
    ["Transformer sem mirror (v4)",   "94.62% ±5.59%"],
    ["ET-1000  1180 amostras",        "96.78% ±2.03%"],
    ["ET-1000  800 amostras",         "98.50% ±0.75%  ← vencedor"],
    ["LOPO ET-1000 (real)",           "74.50% ±11.68%"],
]
table(c, summary, rx, ty2, [230, 150], row_h=18,
      font_size=10, win_rows={7})

# caixa final
box_y = 30
filled_rect(c, 14, box_y, PW - 28, 42, C_WIN)
c.setFont("Helvetica-BoldOblique", 11)
c.setFillColor(C_WHITE)
c.drawCentredString(PW/2, box_y + 26,
    "Com features clássicas (FFT + Kinematic + GEI) e ExtraTrees-1000 atingimos 98.50% ±0.75%")
c.setFont("Helvetica-Oblique", 10)
c.setFillColor(C_LGRAY)
c.drawCentredString(PW/2, box_y + 12,
    "superando todo o estado da arte em +2.5pp sem GPU, sem vídeo RGB e sem redes profundas.")

footer(c, 8)
c.showPage()


c.save()
print(f"PDF gerado: {OUT}")
