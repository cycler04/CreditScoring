#!/usr/bin/env python3
"""Sinh chart PNG (biến thể light + dark) cho docs/week1-report.md.

Chạy:  uv run --with matplotlib docs/assets/gen_charts.py

Mọi con số bad rate / phân phối được tính lại trực tiếp từ
datasets/raw/cs-training.csv; metric, feature importance và IV đọc từ artifact
trong outputs/ (bắt buộc phải có, script không tự sinh lại pipeline).
`selfcheck()` khẳng định mọi số khớp với số đã công bố trong week1-report.md.
Riêng bảng Pearson-vs-IV ở phần 3 và mốc top leaderboard lấy nguyên từ report.

Quy ước vẽ: một series = slot 1 (blue), điểm cần nhấn mạnh = slot 2 (orange)
kèm direct label. Không dual-axis — chart "n + bad rate" tách thành hai panel
dùng chung trục x. Mỗi chart xuất hai biến thể theo surface light/dark để
markdown dùng <picture> + prefers-color-scheme.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "datasets" / "raw" / "cs-training.csv"
ART = ROOT / "outputs"
OUT = Path(__file__).resolve().parent

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
    ),
}

# --- dữ liệu -----------------------------------------------------------------

COLS = [
    "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load():
    acc = {c: [] for c in COLS}
    with CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for c in COLS:
                v = row[c].strip()
                acc[c].append(math.nan if v in ("", "NA") else float(v))
    return {c: np.array(v) for c, v in acc.items()}


def artifact(rel):
    """Đọc một CSV artifact trong outputs/ thành list[dict]."""
    with (ART / rel).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


D = load()
Y = D["SeriousDlqin2yrs"]
BASE = float(Y.mean())
SPECIAL = np.isin(D["NumberOfTimes90DaysLate"], (96, 98))


def rate(mask):
    """(n, bad rate %) trên mask."""
    n = int(mask.sum())
    return n, 100.0 * float(Y[mask].mean()) if n else (n, math.nan)


def vn(x, d=2):
    """Số kiểu Việt Nam: dấu phẩy thập phân, dấu chấm hàng nghìn."""
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt(d):
    """Formatter trục dùng dấu phẩy thập phân."""
    return FuncFormatter(lambda v, _: vn(v, d))


# --- helper vẽ ---------------------------------------------------------------

def style(a, t, grid_axis="y"):
    a.set_facecolor(t["surface"])
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(t["axis"])
        a.spines[side].set_linewidth(0.8)
    a.tick_params(colors=t["muted"], labelsize=9, length=3, width=0.8)
    a.grid(axis=grid_axis, color=t["grid"], lw=0.8, zorder=0)
    a.set_axisbelow(True)


def figure(t, grid_axis="y", **kw):
    fig, ax = plt.subplots(**kw)
    fig.patch.set_facecolor(t["surface"])
    for a in np.atleast_1d(ax).ravel():
        style(a, t, grid_axis)
    return fig, ax


def title(fig, t, head, sub, x=0.012):
    fig.text(x, 0.985, head, color=t["ink"], fontsize=13, weight="bold",
             va="top")
    fig.text(x, 0.925, sub, color=t["ink2"], fontsize=9.5, va="top")


def baseline(a, t, horizontal=True):
    (a.axhline if horizontal else a.axvline)(
        BASE * 100, color=t["ink2"], lw=1.1, ls=(0, (4, 3)), zorder=3)


def save(fig, name, mode, t):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=150, facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


# --- 1. Pearson vs IV (hai panel cạnh nhau, không scatter chồng nhãn) -------

PEARSON_IV = [  # (nhãn, Pearson, IV) — lấy từ report, tính trên train split
    ("RevolvingUtilization", -0.002, 1.10),
    ("Late30–59", 0.126, 0.64),
    ("age", -0.115, 0.23),
    ("OpenCreditLines", -0.030, 0.08),
    ("MonthlyIncome", -0.020, 0.08),
    ("DebtRatio", -0.008, 0.08),
]


def chart_pearson_iv(t, mode):
    rows = sorted(PEARSON_IV, key=lambda r: r[2])  # IV tăng dần, cao nhất lên đầu
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.0, 4.3), sharey=True,
        gridspec_kw=dict(wspace=0.24))
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(top=0.74)
    for a in (left, right):
        style(a, t, grid_axis="x")
        a.grid(axis="y", visible=False)

    y = np.arange(len(rows))
    colors = [t["s2"] if lab == "RevolvingUtilization" else t["s1"]
              for lab, _, _ in rows]

    left.barh(y, [abs(p) for _, p, _ in rows], height=0.5, color=colors,
              zorder=4)
    left.set_xlim(0, 0.175)
    left.set_xticks([0.04, 0.08, 0.12, 0.16])
    left.xaxis.set_major_formatter(fmt(2))
    left.invert_xaxis()
    left.set_title("|Pearson| với target", loc="right", color=t["ink2"],
                   fontsize=9.5, pad=8)
    for i, (lab, p, _) in enumerate(rows):
        left.annotate(vn(p, 3), (abs(p), i), textcoords="offset points",
                      xytext=(-6, 0), va="center", ha="right", fontsize=8.5,
                      color=t["ink"] if colors[i] == t["s2"] else t["ink2"],
                      weight="bold" if colors[i] == t["s2"] else "normal")

    right.barh(y, [iv for _, _, iv in rows], height=0.5, color=colors, zorder=4)
    right.set_xlim(0, 1.42)
    right.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    right.xaxis.set_major_formatter(fmt(1))
    right.set_title("Information Value", loc="left", color=t["ink2"],
                    fontsize=9.5, pad=8)
    right.axvline(0.5, color=t["ink2"], lw=1.0, ls=(0, (4, 3)), zorder=2)
    right.annotate("IV > 0,5\nsoi kỹ", (0.5, -0.62), fontsize=8,
                   color=t["ink2"], ha="center", va="center")
    for i, (lab, _, iv) in enumerate(rows):
        right.annotate(vn(iv), (iv, i), textcoords="offset points",
                       xytext=(6, 0), va="center", fontsize=8.5,
                       color=t["ink"] if colors[i] == t["s2"] else t["ink2"],
                       weight="bold" if colors[i] == t["s2"] else "normal")

    left.set_yticks(y, [lab for lab, _, _ in rows], fontsize=9)
    left.tick_params(axis="y", pad=8)
    for lbl, c in zip(left.get_yticklabels(), colors):
        lbl.set_color(t["ink"] if c == t["s2"] else t["ink2"])
    title(fig, t, "Pearson nói dối, IV thì không",
          "Xếp hạng theo IV (phải) đảo ngược xếp hạng theo tương quan tuyến "
          "tính (trái): RevolvingUtilization mạnh nhất bộ dữ liệu\nnhưng "
          "Pearson gần bằng 0 — đuôi đến 50.708 kéo sập hệ số.")
    save(fig, "01-pearson-vs-iv", mode, t)


# --- 2 & 3. n + bad rate theo bin (hai panel, mỗi panel một trục) -----------

def two_panel(t, mode, name, head, sub, labels, ns, rates, hot, notes, xlabel):
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(8.6, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.8], hspace=0.14))
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(top=0.80)
    for a in (top, bot):
        style(a, t)
    colors = [t["s2"] if i in hot else t["s1"] for i in range(len(labels))]
    x = np.arange(len(labels))

    top.bar(x, ns, width=0.6, color=colors, zorder=4)
    top.set_yscale("log")
    top.set_ylabel("Số hồ sơ (log)", color=t["ink2"], fontsize=9)
    for i, n in enumerate(ns):
        if n < max(ns) / 50:
            top.annotate(vn(n, 0), (i, n), textcoords="offset points",
                         xytext=(0, 5), ha="center", fontsize=8,
                         color=t["ink2"])

    bot.bar(x, rates, width=0.6, color=colors, zorder=4)
    baseline(bot, t)
    bot.set_ylabel("Bad rate (%)", color=t["ink2"], fontsize=9)
    bot.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5)
    bot.set_xticks(x, labels, fontsize=9)
    bot.set_ylim(0, max(rates) * 1.34)
    bot.annotate(f"mức nền {vn(BASE * 100)}%", (-0.42, BASE * 100),
                 textcoords="offset points", xytext=(0, 6), ha="left",
                 fontsize=8.5, color=t["ink2"])
    for i in hot:
        bot.annotate(f"{vn(rates[i])}%", (i, rates[i]),
                     textcoords="offset points", xytext=(0, 7), ha="center",
                     fontsize=9.5, weight="bold", color=t["ink"])
    for i, txt in notes.items():
        bot.annotate(txt, (i, rates[i]), textcoords="offset points",
                     xytext=(0, 26), ha="center", fontsize=8.5,
                     color=t["ink2"])
    title(fig, t, head, sub)
    save(fig, name, mode, t)


UTIL_BINS = [(0, .25), (.25, .5), (.5, .75), (.75, 1), (1, 2), (2, 10),
             (10, math.inf)]
UTIL_LABELS = ["0–0,25", "0,25–0,5", "0,5–0,75", "0,75–1", "1–2", "2–10",
               "> 10"]


def util_groups():
    """Bin right-closed, bin đầu bao gồm cả 0 — đúng convention của report."""
    u = D["RevolvingUtilizationOfUnsecuredLines"]
    return [rate((u >= lo if i == 0 else u > lo) & (u <= hi))
            for i, (lo, hi) in enumerate(UTIL_BINS)]


def chart_utilization(t, mode):
    g = util_groups()
    two_panel(
        t, mode, "02-utilization-badrate",
        "Utilization: rủi ro tăng mạnh — nhưng chỉ tới ngưỡng 2",
        "Đỉnh 40,10% ở bin 1–2 rồi sập về 7,05% ở nhóm > 10, gần bằng mức nền. "
        "Đuôi cực đại là giá trị rác, không phải tín hiệu.",
        UTIL_LABELS, [n for n, _ in g], [r for _, r in g], hot={4, 6},
        notes={6: "≈ mức nền"},
        xlabel="RevolvingUtilizationOfUnsecuredLines")


LATE_LABELS = ["0", "1", "2", "3", "4", "5–17", "96/98"]


def late_groups():
    v = D["NumberOfTimes90DaysLate"]
    out = [rate(v == k) for k in (0, 1, 2, 3, 4)]
    out.append(rate((v >= 5) & (v <= 17)))
    out.append(rate(SPECIAL))
    return out


def chart_late90(t, mode):
    g = late_groups()
    two_panel(
        t, mode, "03-late90-badrate",
        "Mỗi lần quá hạn thêm là một bậc rủi ro",
        "Mã đặc biệt 96/98 có bad rate 54,65% — gấp hơn 8 lần mức nền. Đây là "
        "nhóm rủi ro cực cao, không phải rác: xoá 269 dòng này\nlà vứt đi tín "
        "hiệu mạnh nhất bộ dữ liệu.",
        LATE_LABELS, [n for n, _ in g], [r for _, r in g], hot={6},
        notes={6: "mã đặc biệt"},
        xlabel="NumberOfTimes90DaysLate")


# --- 4. bad rate theo tuổi --------------------------------------------------

AGE_BANDS = [(20, 30, "20s"), (30, 40, "30s"), (40, 50, "40s"),
             (50, 60, "50s"), (60, 70, "60s"), (70, 80, "70s"),
             (80, 200, "80s+")]


def age_groups():
    age = D["age"]
    return [rate((age >= lo) & (age < hi)) for lo, hi, _ in AGE_BANDS]


def chart_age(t, mode):
    g = age_groups()
    rates = [r for _, r in g]
    fig, a = figure(t, figsize=(8.2, 4.4))
    fig.subplots_adjust(top=0.78)
    x = np.arange(len(g))
    a.plot(x, rates, "-o", lw=2, ms=8, color=t["s1"], mec=t["surface"], mew=2,
           zorder=5)
    baseline(a, t)
    a.set_xticks(x, [lab for _, _, lab in AGE_BANDS], fontsize=9.5)
    a.set_ylim(0, max(rates) * 1.24)
    a.set_ylabel("Bad rate (%)", color=t["ink2"], fontsize=9.5)
    a.set_xlabel("Nhóm tuổi", color=t["ink2"], fontsize=9.5)
    for i in (0, len(g) - 1):
        a.annotate(f"{vn(rates[i])}%", (i, rates[i]),
                   textcoords="offset points", xytext=(0, 11), ha="center",
                   fontsize=9.5, weight="bold", color=t["ink"])
    a.annotate(f"mức nền {vn(BASE * 100)}%", (2.5, BASE * 100),
               textcoords="offset points", xytext=(0, -15), ha="center",
               fontsize=8.5, color=t["ink2"])
    title(fig, t, "Tuổi giảm rủi ro, đơn điệu và sạch",
          "Không đảo chiều ở bất kỳ nhóm nào, cắt mức nền giữa 40s và 50s — "
          "quan hệ thật, không phải nhiễu.")
    save(fig, "04-age-badrate", mode, t)


# --- 5. missing là tín hiệu tốt ---------------------------------------------

def missing_rows():
    out = []
    for col in ("MonthlyIncome", "NumberOfDependents"):
        m = np.isnan(D[col])
        out.append((col, rate(m), rate(~m)))
    return out


def chart_missing(t, mode):
    rows = missing_rows()
    fig, a = figure(t, grid_axis="x", figsize=(8.4, 4.0))
    fig.subplots_adjust(top=0.76)
    ys, ticks, colors = [], [], []
    for k, (col, miss, have) in enumerate(rows):
        for j, (grp, (n, r)) in enumerate((("có dữ liệu", have),
                                           ("thiếu", miss))):
            y = k * 2.6 + j
            c = t["s2"] if grp == "thiếu" else t["s1"]
            a.barh(y, r, height=0.52, color=c, zorder=4)
            a.annotate(f"{vn(r)}%   n = {vn(n, 0)}", (r, y),
                       textcoords="offset points", xytext=(7, 0), va="center",
                       fontsize=9, color=t["ink"],
                       weight="bold" if grp == "thiếu" else "normal")
            ys.append(y)
            ticks.append((y, f"{col}\n{grp}" if j else grp))
            colors.append(c)
    a.set_yticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=9)
    for lbl, c in zip(a.get_yticklabels(), colors):
        lbl.set_color(t["ink2"] if c == t["s1"] else t["ink"])
    baseline(a, t, horizontal=False)
    a.annotate(f"mức nền {vn(BASE * 100)}%", (BASE * 100, max(ys) + 0.55),
               textcoords="offset points", xytext=(5, 0), fontsize=8.5,
               color=t["ink2"])
    a.set_xlim(0, 11.5)
    a.set_ylim(-0.7, max(ys) + 0.9)
    a.grid(axis="y", visible=False)
    a.set_xlabel("Bad rate (%)", color=t["ink2"], fontsize=9.5)
    title(fig, t, "Missing ở đây là tín hiệu tốt, không phải tín hiệu xấu",
          "Cả hai nhóm thiếu dữ liệu đều an toàn hơn nhóm có dữ liệu — đảo "
          "ngược giả định mặc định của ngành.\nPhải đo WoE của bin missing, "
          "đừng suy đoán dấu.")
    save(fig, "05-missing-signal", mode, t)


# --- 6. độ dài đuôi phải: max / p99 -----------------------------------------

TAIL_FEATURES = [
    ("RevolvingUtilization", "RevolvingUtilizationOfUnsecuredLines"),
    ("DebtRatio", "DebtRatio"),
    ("MonthlyIncome", "MonthlyIncome"),
    ("Dependents", "NumberOfDependents"),
    ("RealEstateLoans", "NumberRealEstateLoansOrLines"),
    ("Late30–59", "NumberOfTime30-59DaysPastDueNotWorse"),
    ("Late90+", "NumberOfTimes90DaysLate"),
    ("Late60–89", "NumberOfTime60-89DaysPastDueNotWorse"),
    ("OpenCreditLines", "NumberOfOpenCreditLinesAndLoans"),
    ("age", "age"),
]


def tail_ratios():
    """max/p99 sau khi loại 269 dòng mang mã 96/98 (mã đặc biệt ≠ đuôi thật)."""
    out = []
    for short, col in TAIL_FEATURES:
        v = D[col][~SPECIAL]
        v = v[~np.isnan(v)]
        p99 = float(np.percentile(v, 99))
        out.append((short, float(v.max()) / p99, p99, float(v.max())))
    return sorted(out, key=lambda r: r[1])


def chart_tail(t, mode):
    rows = tail_ratios()
    fig, a = figure(t, grid_axis="x", figsize=(8.2, 4.6))
    fig.subplots_adjust(top=0.78)
    y = np.arange(len(rows))
    top3 = set(range(len(rows) - 3, len(rows)))
    colors = [t["s2"] if i in top3 else t["s1"] for i in y]
    a.barh(y, [r[1] for r in rows], height=0.55, color=colors, zorder=4)
    a.set_xscale("log")
    a.set_xlim(1, 3e5)
    a.set_yticks(y, [r[0] for r in rows], fontsize=9)
    for lbl, c in zip(a.get_yticklabels(), colors):
        lbl.set_color(t["ink"] if c == t["s2"] else t["ink2"])
    a.set_xlabel("max / p99 (thang log) — càng xa 1 càng lệch đuôi phải",
                 color=t["ink2"], fontsize=9.5)
    a.grid(axis="y", visible=False)
    for i in top3:
        short, ratio, p99, mx = rows[i]
        a.annotate(f"×{vn(ratio, 0)}", (ratio, i), textcoords="offset points",
                   xytext=(7, 0), va="center", fontsize=9.5, weight="bold",
                   color=t["ink"])
    title(fig, t, "Ba feature có đuôi phải dài bất thường",
          "max lệch p99 hàng chục đến hàng chục nghìn lần, phần còn lại dưới "
          "×15. Đã loại 269 dòng mã 96/98 nên cột\ndelinquency phản ánh đuôi "
          "thật, không phải mã đặc biệt.")
    save(fig, "06-tail-ratio", mode, t)


# --- artifact: metric và feature importance của pipeline local ---------------

MODELS = [("logistic_raw", "Logistic Regression\ntrên feature thô"),
          ("logistic_woe", "Logistic Regression\ntrên WoE"),
          ("lightgbm", "LightGBM")]
TOP_LEADERBOARD = 0.8696  # tham chiếu ngoài, không phải kết quả local


def metrics():
    """{(model, split): {auc, gini, ks}} từ outputs/models/metrics/metrics.csv."""
    return {(r["model"], r["split"]): {k: float(r[k])
                                       for k in ("auc", "gini", "ks")}
            for r in artifact("models/metrics/metrics.csv")}


def importance(model, features):
    """% importance của từng feature trong `features`, theo thứ tự truyền vào."""
    tbl = {r["feature"]: float(r["importance_pct"]) * 100
           for r in artifact("models/feature_importance/"
                             "feature_importance_table.csv")
           if r["model"] == model}
    return [tbl.get(f, 0.0) for f in features]


# --- 7. AUC local so với top leaderboard (dot plot: trục x bị cắt) ----------

def chart_auc(t, mode):
    m = metrics()
    rows = [(lab, m[(key, "test")]["auc"]) for key, lab in MODELS]
    fig, a = figure(t, grid_axis="x", figsize=(8.4, 3.8))
    fig.subplots_adjust(top=0.74)
    for i, (lab, v) in enumerate(rows):
        c = t["s2"] if i == 1 else t["s1"]
        a.plot([0.80, v], [i, i], color=t["grid"], lw=1.0, zorder=2)
        a.plot(v, i, "o", ms=11, color=c, mec=t["surface"], mew=2, zorder=5)
        left = v > 0.86  # tránh đè lên đường tham chiếu leaderboard
        a.annotate(vn(v, 4), (v, i), textcoords="offset points",
                   xytext=(-10 if left else 10, 0), va="center",
                   ha="right" if left else "left", fontsize=9.5,
                   weight="bold" if i == 1 else "normal",
                   color=t["ink"] if i == 1 else t["ink2"])
    a.axvline(TOP_LEADERBOARD, color=t["ink2"], lw=1.1, ls=(0, (4, 3)),
              zorder=3)
    a.annotate(f"top leaderboard {vn(TOP_LEADERBOARD, 4)}\n(tham chiếu ngoài)",
               (TOP_LEADERBOARD, -0.52), fontsize=8.5, color=t["ink2"],
               ha="right", va="center")
    a.set_xlim(0.80, 0.885)
    a.xaxis.set_major_formatter(fmt(2))
    a.set_ylim(-0.85, len(rows) - 0.4)
    a.set_yticks(np.arange(len(rows)), [lab for lab, _ in rows], fontsize=9)
    for i, lbl in enumerate(a.get_yticklabels()):
        lbl.set_color(t["ink"] if i == 1 else t["ink2"])
    a.set_xlabel("AUC trên test split (trục x bắt đầu từ 0,80)",
                 color=t["ink2"], fontsize=9.5)
    a.grid(axis="y", visible=False)
    title(fig, t, "Xử lý dữ liệu cẩn thận gần bằng mô hình phức tạp",
          "Đo trên test split, 30.000 hồ sơ. WoE thu hẹp khoảng cách LR–GBDT "
          "từ 0,050 xuống 0,019 AUC.\nDot plot chứ không bar: trục x bị cắt "
          "nên độ dài bar sẽ nói dối.")
    save(fig, "07-auc-ladder", mode, t)


# --- 8. feature importance: ba mô hình, ba thước đo, small multiples --------

FI_FEATURES = [  # thứ tự theo gain của LightGBM, cao nhất lên đầu
    "NumberOfTimes90DaysLate", "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse", "MonthlyIncome", "DebtRatio",
    "age", "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]
FI_SHORT = ["Late90+", "RevolvingUtilization", "Late30–59", "Late60–89",
            "MonthlyIncome", "DebtRatio", "age", "OpenCreditLines",
            "RealEstateLoans", "Dependents"]
FI_PANELS = [("lightgbm", "LightGBM\ngain", 34),
             ("logistic_woe", "LR trên WoE\n|hệ số|", 28),
             ("logistic_raw", "LR trên feature thô\n|hệ số|", 20)]


def chart_importance(t, mode):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.6), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(top=0.72)
    y = np.arange(len(FI_FEATURES))
    for a, (model, lab, xmax) in zip(axes, FI_PANELS):
        style(a, t, grid_axis="x")
        a.grid(axis="y", visible=False)
        vals = importance(model, FI_FEATURES)
        top = max(vals)
        colors = [t["s2"] if v >= top * 0.95 else t["s1"] for v in vals]
        a.barh(y, vals, height=0.55, color=colors, zorder=4)
        a.set_xlim(0, xmax)
        a.set_title(lab, loc="left", color=t["ink2"], fontsize=9.5, pad=8)
        a.set_xlabel("% trong mô hình", color=t["ink2"], fontsize=9)
        for i, v in enumerate(vals):
            a.annotate(vn(v, 1), (v, i), textcoords="offset points",
                       xytext=(5, 0), va="center", fontsize=8,
                       color=t["ink"] if colors[i] == t["s2"] else t["ink2"])
    axes[0].set_yticks(y, FI_SHORT, fontsize=9)
    axes[0].invert_yaxis()
    for lbl in axes[0].get_yticklabels():
        lbl.set_color(t["ink2"])
    title(fig, t, "Ba mô hình, ba thước đo, ba thứ hạng khác nhau",
          "Cùng thứ tự feature ở cả ba panel (xếp theo gain của LightGBM). "
          "Ba biến delinquency và RevolvingUtilization\nchiếm phần lớn gain, "
          "nhưng LR trên feature thô đẩy RevolvingUtilization và DebtRatio về "
          "hệ số 0.")
    save(fig, "08-feature-importance", mode, t)


# --- 9. AUC / Gini / KS, valid so với test ----------------------------------

METRIC_PANELS = [("auc", "AUC", (0.80, 0.885)),
                 ("gini", "Gini", (0.60, 0.77)),
                 ("ks", "KS", (0.44, 0.63))]


def chart_metrics(t, mode):
    m = metrics()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(top=0.70)
    y = np.arange(len(MODELS))
    for a, (key, lab, xlim) in zip(axes, METRIC_PANELS):
        style(a, t, grid_axis="x")
        a.grid(axis="y", visible=False)
        for i, (model, _) in enumerate(MODELS):
            v_valid = m[(model, "valid")][key]
            v_test = m[(model, "test")][key]
            a.plot([v_valid, v_test], [i, i], color=t["grid"], lw=1.4,
                   zorder=2)
            a.plot(v_valid, i, "o", ms=9, color=t["s2"], mec=t["surface"],
                   mew=2, zorder=5,
                   label="valid" if (i == 0 and key == "auc") else None)
            a.plot(v_test, i, "o", ms=9, color=t["s1"], mec=t["surface"],
                   mew=2, zorder=5,
                   label="test" if (i == 0 and key == "auc") else None)
            a.annotate(vn(v_test, 4), (v_test, i), textcoords="offset points",
                       xytext=(0, 11), ha="center", fontsize=8.5,
                       color=t["ink"])
        a.set_xlim(*xlim)
        a.xaxis.set_major_formatter(fmt(2))
        a.set_title(lab, loc="left", color=t["ink2"], fontsize=10, pad=8)
    axes[0].set_yticks(y, [lab.replace("\n", " ") for _, lab in MODELS],
                       fontsize=9)
    axes[0].set_ylim(-0.6, len(MODELS) - 0.3)
    for lbl in axes[0].get_yticklabels():
        lbl.set_color(t["ink2"])
    leg = fig.legend(loc="upper right", bbox_to_anchor=(0.995, 0.99),
                     frameon=False, fontsize=9, handletextpad=0.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    title(fig, t, "Ba metric xếp hạng, valid và test khớp nhau",
          "Nhãn số là giá trị test. Ba metric cho cùng một thứ hạng mô hình. "
          "Khoảng cách valid–test của LightGBM gần bằng 0;\nLR trên WoE tụt "
          "0,005 AUC — không có dấu hiệu overfit ở cả ba.")
    save(fig, "09-metrics-valid-test", mode, t)


# --- 10. dải điểm mỗi feature đóng góp vào scorecard ------------------------

SC_SHORT = {
    "NumberOfTime30-59DaysPastDueNotWorse": "Late30–59",
    "NumberOfTimes90DaysLate": "Late90+",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late60–89",
    "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilization",
    "NumberOfOpenCreditLinesAndLoans": "OpenCreditLines",
    "NumberRealEstateLoansOrLines": "RealEstateLoans",
    "NumberOfDependents": "Dependents",
}
NEUTRAL_POINTS = 61  # bin WoE = 0 (missing rỗng trên train) của mọi feature


def score_ranges():
    """[(nhãn, điểm nhỏ nhất, điểm lớn nhất)] xếp theo dải điểm tăng dần."""
    rows = {}
    for r in artifact("scorecard/scorecard.csv"):
        p = int(r["points"])
        lo, hi = rows.get(r["feature"], (p, p))
        rows[r["feature"]] = (min(lo, p), max(hi, p))
    out = [(SC_SHORT.get(f, f), lo, hi) for f, (lo, hi) in rows.items()]
    return sorted(out, key=lambda r: r[2] - r[1])


def chart_score_range(t, mode):
    rows = score_ranges()
    fig, a = figure(t, grid_axis="x", figsize=(8.6, 4.8))
    fig.subplots_adjust(top=0.76)
    y = np.arange(len(rows))
    for i, (lab, lo, hi) in enumerate(rows):
        broken = hi > 120  # hệ số sai dấu: bin rủi ro nhất lại được điểm cao nhất
        a.plot([lo, hi], [i, i], color=t["s2"] if broken else t["grid"],
               lw=2.0 if broken else 1.6, zorder=3)
        a.plot(lo, i, "o", ms=9, color=t["s2"], mec=t["surface"], mew=2,
               zorder=5, label="điểm thấp nhất" if i == 0 else None)
        a.plot(hi, i, "o", ms=9, color=t["s1"], mec=t["surface"], mew=2,
               zorder=5, label="điểm cao nhất" if i == 0 else None)
        a.annotate(str(lo), (lo, i), textcoords="offset points",
                   xytext=(-7, 0), ha="right", va="center", fontsize=8.5,
                   color=t["ink2"])
        a.annotate(str(hi), (hi, i), textcoords="offset points",
                   xytext=(7, 0), va="center", fontsize=8.5,
                   color=t["ink"] if broken else t["ink2"],
                   weight="bold" if broken else "normal")
    a.axvline(NEUTRAL_POINTS, color=t["ink2"], lw=1.1, ls=(0, (4, 3)), zorder=2)
    a.annotate(f"{NEUTRAL_POINTS} điểm = bin WoE 0", (NEUTRAL_POINTS, -0.72),
               fontsize=8.5, color=t["ink2"], ha="center", va="center")
    a.set_yticks(y, [lab for lab, _, _ in rows], fontsize=9)
    for lbl, (_, _, hi) in zip(a.get_yticklabels(), rows):
        lbl.set_color(t["ink"] if hi > 120 else t["ink2"])
    a.set_xlim(-60, 175)
    a.set_ylim(-1.0, len(rows) - 0.4)
    a.set_xlabel("Điểm cộng vào scorecard 300–850", color=t["ink2"],
                 fontsize=9.5)
    a.grid(axis="y", visible=False)
    leg = a.legend(loc="lower right", frameon=False, fontsize=9,
                   handletextpad=0.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    title(fig, t, "Bốn feature quyết định gần hết điểm",
          "Ba biến delinquency và RevolvingUtilization có dải điểm 94–122; sáu "
          "biến còn lại cộng lại chưa tới 105.\nLate60–89 lệch hẳn: bin rủi ro "
          "nhất của nó lại nhận điểm cao nhất bảng (154) — hệ số sai dấu.")
    save(fig, "10-score-range", mode, t)


CHARTS = [chart_pearson_iv, chart_utilization, chart_late90, chart_age,
          chart_missing, chart_tail, chart_auc, chart_importance,
          chart_metrics, chart_score_range]


def selfcheck():
    """Đối chiếu số tính lại với số đã công bố trong week1-report.md."""
    assert len(Y) == 150_000, len(Y)
    assert int(Y.sum()) == 10_026 and round(BASE * 100, 3) == 6.684
    assert int(SPECIAL.sum()) == 269
    for col in ("NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse"):
        assert np.array_equal(SPECIAL, np.isin(D[col], (96, 98))), col
    assert int(np.isnan(D["MonthlyIncome"]).sum()) == 29_731
    assert int(np.isnan(D["NumberOfDependents"]).sum()) == 3_924
    assert [n for n, _ in util_groups()] == [87_657, 21_055, 13_764, 24_203,
                                            2_950, 130, 241]
    got = [round(r, 2) for _, r in util_groups()]
    assert got == [2.14, 5.29, 10.13, 18.21, 40.10, 28.46, 7.05], got
    assert [n for n, _ in late_groups()] == [141_662, 5_243, 1_555, 667, 291,
                                             313, 269]
    got = [round(r, 2) for _, r in late_groups()]
    assert got == [4.63, 33.66, 49.90, 57.72, 67.01, 65.18, 54.65], got
    got = [round(r, 2) for _, r in age_groups()]
    # report ghi nhóm 80s+ là "~2,0%" — tính lại được 2,05%
    assert got == [11.73, 10.07, 8.37, 6.45, 3.63, 2.43, 2.05], got
    got = [(round(r, 2), n) for _, (n, r), _ in missing_rows()]
    assert got == [(5.61, 29_731), (4.56, 3_924)], got
    got = [(round(r, 2), n) for _, _, (n, r) in missing_rows()]
    assert got == [(6.95, 120_269), (6.74, 146_076)], got

    # artifact trong outputs/ — số dùng trong phần 4, 5 và 6
    m = metrics()
    got = {k: round(v["auc"], 4) for k, v in m.items() if k[1] == "test"}
    assert got == {("logistic_raw", "test"): 0.8165,
                   ("logistic_woe", "test"): 0.8473,
                   ("lightgbm", "test"): 0.8664}, got
    assert round(m[("lightgbm", "valid")]["auc"], 4) == 0.8664
    assert round(m[("logistic_woe", "test")]["ks"], 4) == 0.5361
    for model in ("lightgbm", "logistic_woe", "logistic_raw"):
        assert len(importance(model, FI_FEATURES)) == 10
    iv = {r["feature"]: round(float(r["iv"]), 4)
          for r in artifact("scorecard/iv_summary.csv")}
    assert iv["RevolvingUtilizationOfUnsecuredLines"] == 1.0970
    assert iv["NumberOfTimes90DaysLate"] == 0.8504
    assert all(r["monotonic_woe"] == "True"
               for r in artifact("scorecard/iv_summary.csv"))

    # phần 7: hai hệ số WoE dương ngược chiều kỳ vọng, và hệ quả trên điểm
    card = artifact("scorecard/scorecard.csv")
    coef = {r["feature"]: float(r["coefficient"]) for r in card}
    positive = sorted(f for f, c in coef.items() if c > 0)
    assert positive == ["NumberOfTime60-89DaysPastDueNotWorse",
                        "NumberOfOpenCreditLinesAndLoans"][::-1], positive
    late6089 = {r["bin"]: int(r["points"]) for r in card
                if r["feature"] == "NumberOfTime60-89DaysPastDueNotWorse"}
    assert late6089 == {"(-inf, inf]": 60, "MISSING": 154}, late6089
    assert {lab: hi - lo for lab, lo, hi in score_ranges()}[
        "RevolvingUtilization"] == 112
    assert all(int(r["points"]) == NEUTRAL_POINTS for r in card
               if float(r["woe"]) == 0.0)
    # Late60–89 mất quyền cắt bin vì thiếu <100 dòng so với min_samples_leaf 5%
    v = D["NumberOfTime60-89DaysPastDueNotWorse"]
    nonzero_train = round(int(((v >= 1) & (v <= 17)).sum()) * 0.6)
    assert 4_400 <= nonzero_train < 4_500, nonzero_train
    print("selfcheck OK — mọi số khớp week1-report.md và outputs/")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"]})
    selfcheck()
    for mode, t in THEMES.items():
        for fn in CHARTS:
            fn(t, mode)
    print(f"{len(CHARTS) * len(THEMES)} PNG -> {OUT}")
