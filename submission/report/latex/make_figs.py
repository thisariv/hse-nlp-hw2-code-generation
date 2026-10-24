"""Рисунки для LaTeX-отчёта. Числа взяты из work/notes/*.md и work/results/*.json (ссылки в комментариях)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})
BLUE, ORANGE, GREEN, GREY, RED = "#3b75af", "#d0843a", "#4d9a55", "#8a8f98", "#c0504d"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def leaderboard():
    # Публичный лидерборд по посылкам: notes/final-inference.md, final-variants.md, сообщения пользователя.
    items = [("C greedy", 70.94), ("C, голосование", 73.12), ("то же,\nновый чеккер", 76.35),
             ("v6 адаптивное\nголосование", 77.71), ("v7 ансамбль\n3 моделей", 77.92),
             ("v8 + модель W", 78.54), ("v9 модели B5", 78.44), ("v10 модели B6", 78.96),
             ("v11 бюджет 1/8/8", 79.37), ("финал: модели B7", 80.10)]
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(vals))
    ax.plot(x, vals, "-o", color=BLUE, lw=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.axvline(1.5, color=GREY, ls="--", lw=1)
    ax.text(1.55, 71, "смена чеккера\nу организаторов", fontsize=8, color=GREY)
    ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("public pass@1, %")
    ax.set_ylim(70, 81.2)
    save(fig, "leaderboard")


def single_model_path():
    # Путь качества одиночной модели (greedy exec-match). №18–22: на 991 задаче; дальше — heldout 4096.
    items = [("№18\nSwiGLU, 500k пар", 15.24), ("№21\nэнкодер 4 слоя", 19.48), ("№24\nнорм. эмбеддингов", 39.11),
             ("№25\nLR, dropout 0", 40.94), ("№26b\n6,4M пар, WSD", 61.96), ("L2\nC 6+6, 16M пар", 70.41),
             ("L3\nC, 27M пар", 71.36), ("P1-B\nдлинный cooldown", 72.07), ("B6\nW / C, cooldown 11%", 72.61),
             ("B7\ncooldown 22%", 72.92)]
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    colors = [GREY] * 2 + [BLUE] * 3 + [ORANGE] * 2 + [GREEN] * 3
    ax.bar(range(len(vals)), vals, color=colors, width=.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)), [l.replace("\n", ": ") for l in labels], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("greedy exec-match, %")
    ax.set_ylim(0, 82)
    ax.text(-0.3, 26, "оценка\nна 991 задаче", fontsize=8, color=GREY)
    save(fig, "single_model_path")


def learning_curves():
    # notes/b4-results.md, b5-results.md, b6-results.md, b7-results.md, l3-results.md, p1-results.md
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    cl_cd15 = [(9.72, 67.48), (13.40, 68.14), (21.37, 69.92), (31.94, 70.02)]
    cl_cd50 = [(15.85, 70.41), (26.77, 71.36), (37.59, 71.12)]
    w_cd15 = [(3.10, 63.40), (4.16, 65.94), (5.91, 67.24), (7.66, 68.51), (11.08, 69.85), (12.70, 70.21), (16.94, 70.36)]
    w_cd50 = [(9.40, 70.39), (14.33, 70.90), (19.78, 70.95)]
    for pts, c, lab, st in [(cl_cd15, ORANGE, "C 54M: cooldown 15 мин", ":"), (cl_cd50, ORANGE, "C 54M: cooldown 50 мин", "-"),
                            (w_cd15, BLUE, "W 100M: cooldown 15 мин", ":"), (w_cd50, BLUE, "W 100M: cooldown 50 мин", "-")]:
        x, y = zip(*pts)
        ax.plot(x, y, st, marker="o", color=c, label=lab, ms=4, lw=1.5)
    ax.scatter([26.77, 37.59, 37.59], [72.07, 72.61, 72.85], marker="*", s=160, color=ORANGE, zorder=5, label="C: длинный cooldown (P1-B, B6, B7)")
    ax.scatter([19.78, 19.78], [72.41, 72.92], marker="*", s=160, color=BLUE, zorder=5, label="W: длинный cooldown (B6, B7)")
    ax.set_xlabel("увиденных пар вопрос–код, млн")
    ax.set_ylabel("greedy exec-match, % (heldout 4096)")
    ax.set_ylim(62.5, 73.5)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "learning_curves")


def cooldown():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    groups = ["W (6+6, d768)", "C-long2x (6+6, d512)"]
    vals = [[70.95, 72.41, 72.92], [71.12, 72.61, 72.85]]
    labs = ["50 мин, линейный", "1−√t, ~11% обучения (B6)", "1−√t, ~22% обучения (B7)"]
    cols = [GREY, BLUE, GREEN]
    w = .25
    for j in range(3):
        xs = np.arange(2) + (j - 1) * w
        ax.bar(xs, [v[j] for v in vals], width=w, color=cols[j], label=labs[j])
        for xx, v in zip(xs, [v[j] for v in vals]):
            ax.text(xx, v + .05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(2), groups)
    ax.set_ylim(70, 73.4)
    ax.set_ylabel("greedy exec-match, %")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=1, frameon=False)
    save(fig, "cooldown")


def voting():
    # tune2048, K8-журналы: greedy C B6 1475; K8 C B6 1526; v11-политика 1551; тройка v2 1557.
    names = ["greedy\nодной модели", "голосование\n8 сэмплов\nодной модели", "3 модели,\nадаптивно 1/8/8\n(v11)", "3 модели\nB7 + P1-B\n(финал v2)"]
    vals = [1475, 1526, 1551, 1557]
    pct = [100 * v / 2048 for v in vals]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(4), pct, color=[GREY, BLUE, BLUE, GREEN], width=.6)
    for i, p in enumerate(pct):
        ax.text(i, p + .1, f"{p:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(4), names, fontsize=8)
    ax.set_ylim(70, 77)
    ax.set_ylabel("exec-match на tune, %")
    save(fig, "voting")


def ideas():
    # Отдельная проверочная выборка: e1-ensembles.md (eval2048), tta-experiment.md (eval2048), e2-policy.md (heldout4k), b7-results.md
    rows = [("Токенный ансамбль (greedy)", -2.25, -3.32, -1.17), ("Test-time augmentation", -0.05, -0.39, 0.29),
            ("Model soup C B6 + P1-B", -0.05, -0.54, 0.39), ("Cross-LL как вес голоса", 0.15, -0.24, 0.54),
            ("Политика E2 (смесь T, 3-й этап)", 0.20, -0.24, 0.64), ("Смесь температур", 0.68, -0.10, 1.46),
            ("Бюджет 1/8/8 (принят)", 0.73, 0.10, 1.37)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (n, d, lo, hi) in enumerate(rows):
        c = GREEN if lo > 0 else (RED if hi < 0 else GREY)
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=c, capsize=4)
    ax.axvline(0, color="k", lw=.8, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("разница с текущим решением, п.п. (95% CI, отдельная выборка)")
    save(fig, "ideas")


def time_budget():
    # v11 test1919: test-1/timing.json (126,8 с); финал v2 — та же схема, 122,7 с.
    parts = [("импорт, загрузка, прогрев", 17.3), ("greedy якоря", 13.8), ("этап 1: по 1 сэмплу\nот 3 моделей", 32.6),
             ("этап 2: до 8 сэмплов\n(38% задач)", 56.1), ("голосование,\nпроверка, запись", 7.0)]
    fig, ax = plt.subplots(figsize=(8.6, 1.6))
    left = 0
    cols = [GREY, BLUE, "#6c9bd0", ORANGE, GREEN]
    for (n, v), c in zip(parts, cols):
        ax.barh(0, v, left=left, color=c, label=f"{n.replace(chr(10), ' ')} — {v:.0f} с")
        ax.text(left + v / 2, 0, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white")
        left += v
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.45), ncol=2, frameon=False)
    ax.axvline(180, color=RED, lw=2)
    ax.text(178, .45, "лимит 180 с", color=RED, ha="right", fontsize=8)
    ax.set_xlim(0, 185)
    ax.set_yticks([])
    ax.set_xlabel("секунды на одной T4 (все 1919 задач теста)")
    ax.grid(False)
    save(fig, "time_budget")


def compute():
    # session-summary.json → wall_seconds; L1 — только основное обучение (17 570 с, полная сессия длиннее);
    # «короткие» — ≈40 запусков: смоуки, инференс v5–v11 и пакеты, E1/E2, TTA, варианты (оценка).
    rows = [("S1–S3: голосование, ёмкость, представления", 3.9, BLUE), ("L1: C с нуля (основное обучение)", 4.9, GREEN),
            ("L2: C, 2 порядка данных", 9.56, GREEN), ("L3: C, продолжение", 10.56, GREEN), ("P1: длинный cooldown", 10.09, GREEN),
            ("R1: глубокий энкодер", 9.94, GREY), ("D1: фильтр шумных пар", 1.33, GREY), ("B1: W/E с нуля (NaN)", 10.48, RED),
            ("B2: W с QK-norm", 8.19, GREEN), ("B3: W, 2 LR", 10.63, GREEN), ("B4: W + truncation", 10.64, GREY),
            ("B5: W и C, +9 ч", 10.74, GREY), ("B6: cooldown ×1", 3.97, GREEN), ("B7: cooldown ×2", 7.42, GREEN),
            ("короткие запуски (≈40 шт.)", 6.0, "#b9a6d6")]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (n, h, c) in enumerate(rows):
        ax.barh(i, h, color=c)
        ax.text(h + .1, i, f"{h:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("часы на Kaggle (2×T4); всего ≈118 ч")
    ax.set_xlim(0, 12.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="дало прирост"), Patch(color=GREY, label="без прироста"),
                       Patch(color=RED, label="провал"), Patch(color=BLUE, label="разведка")],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(.35, -.1), ncol=4, frameon=False)
    save(fig, "compute")


if __name__ == "__main__":
    for f in [leaderboard, single_model_path, learning_curves, cooldown, voting, ideas, time_budget, compute]:
        f()
    print("ok", sorted(p.name for p in OUT.iterdir()))
