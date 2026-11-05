"""Build graphs/rop_dataset/ — the ROP corpus, as figures.

Every figure reads data/rop_classified/MANIFEST.csv (and, for figure 5, the Ostrava metadata
join). Nothing is hardcoded, so re-running scripts/build_rop_classified.py and re-running this
keeps the picture and the number in step — the same contract as scripts/make_graphs.py.

    .venv/bin/python scripts/make_dataset_graphs.py

Palette: six categorical hues, one per source dataset. Validated on the dark surface with the
dataviz six-check validator — all six PASS, worst adjacent CVD ΔE 8.4 (protan). Gold↔aqua sit
at tritan ΔE 4.0, so every segment is ALSO direct-labelled and separated by a 2px surface gap;
identity never depends on colour alone.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data" / "rop_classified" / "MANIFEST.csv"
OUT = ROOT / "graphs" / "rop_dataset"

SURF, INK, INK2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#33322f"
SRC_COLOR = {                      # fixed order, never cycled
    "ostrava":   "#d95926",
    "ropvl":     "#3987e5",
    "multiview": "#199e70",
    "shenzhen":  "#c98500",
    "ropstages": "#a06be0",
    "hvdropdb":  "#e05a8a",
}
SRC_LABEL = {"ostrava": "Ostrava (CZ)", "ropvl": "ROP-VL (CN)", "multiview": "Multi-View (CN)",
             "shenzhen": "Shenzhen (CN)", "ropstages": "ropstages", "hvdropdb": "HVDROPDB (IN)"}
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

STAGE_ORDER = ["normal", "stage_0", "stage_1", "stage_2", "stage_3", "stage_4", "stage_5",
               "arop", "treated"]
NICE = {"normal": "Normal", "stage_0": "Stage 0", "stage_1": "Stage 1", "stage_2": "Stage 2",
        "stage_3": "Stage 3", "stage_4": "Stage 4", "stage_5": "Stage 5",
        "arop": "AP-ROP", "treated": "Treated / laser"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 10.5,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"  -> graphs/rop_dataset/{name}")


def load():
    if not MAN.exists():
        print(f"missing {MAN} — run scripts/build_rop_classified.py first", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(MAN)
    return df


def n_infants(s):
    """Distinct infants in a patient_id series. HVDROPDB ships NO patient IDs, so its rows
    carry an empty id — counting "" as a patient would inflate every class it touches by 1."""
    return s.replace("", pd.NA).dropna().nunique()


def bare(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1: the table
def fig_table(df):
    st = df[df.cls.isin(STAGE_ORDER)]
    srcs = [s for s in SRC_COLOR if s in set(st.source)]
    M = pd.crosstab(st.cls, st.source).reindex(index=STAGE_ORDER, columns=srcs, fill_value=0)
    tot_r = M.sum(axis=1)
    inf = st.groupby("cls").patient_id.apply(n_infants).reindex(STAGE_ORDER, fill_value=0)

    nr, nc = len(STAGE_ORDER), len(srcs)
    fig, ax = plt.subplots(figsize=(2.15 + 1.02 * nc + 2.5, 0.52 * nr + 2.0))
    ax.set_axis_off()
    vmax = float(M.values.max())

    x0, w, y0, h = 0.0, 1.0, 0.0, 1.0
    # header
    for j, s in enumerate(srcs):
        ax.text(x0 + 1.6 + j * w + w / 2, nr + 0.55, SRC_LABEL[s].replace(" (", "\n("),
                ha="center", va="bottom", fontsize=9.5, color=SRC_COLOR[s], fontweight="bold")
    ax.text(x0 + 1.6 + nc * w + 0.62, nr + 0.55, "TOTAL\nimages", ha="center", va="bottom",
            fontsize=9.5, color=INK, fontweight="bold")
    ax.text(x0 + 1.6 + nc * w + 1.85, nr + 0.55, "infants", ha="center", va="bottom",
            fontsize=9.5, color=INK2, fontweight="bold")

    for i, cls in enumerate(STAGE_ORDER):
        y = nr - 1 - i
        ax.text(x0 + 1.5, y + 0.5, NICE[cls], ha="right", va="center", fontsize=11,
                color=INK, fontweight="bold" if cls.startswith("stage") else "normal")
        for j, s in enumerate(srcs):
            v = int(M.iloc[i, j])
            cx = x0 + 1.6 + j * w
            if v:
                frac = (np.log1p(v) / np.log1p(vmax)) if vmax else 0
                col = BLUES[min(len(BLUES) - 1, int(frac * (len(BLUES) - 1)))]
                ax.add_patch(plt.Rectangle((cx + 0.03, y + 0.06), w - 0.06, h - 0.12,
                                           facecolor=col, edgecolor=SURF, lw=1.4))
                ax.text(cx + w / 2, y + 0.5, f"{v:,}", ha="center", va="center", fontsize=10,
                        color="#0d1b2a" if frac > 0.45 else "#12243a", fontweight="bold")
            else:
                ax.text(cx + w / 2, y + 0.5, "·", ha="center", va="center",
                        fontsize=12, color=GRID)
        ax.text(x0 + 1.6 + nc * w + 0.62, y + 0.5, f"{int(tot_r.iloc[i]):,}", ha="center",
                va="center", fontsize=11.5, color=INK, fontweight="bold")
        ax.text(x0 + 1.6 + nc * w + 1.85, y + 0.5, f"{int(inf.iloc[i]):,}", ha="center",
                va="center", fontsize=10.5, color=INK2)

    ax.plot([x0 + 1.55, x0 + 1.6 + nc * w + 2.4], [nr + 0.42] * 2, color=GRID, lw=1)
    ax.plot([x0 + 1.55, x0 + 1.6 + nc * w + 2.4], [0 - 0.08] * 2, color=GRID, lw=1)
    ax.text(x0 + 1.5, -0.55, "TOTAL", ha="right", va="center", fontsize=11, fontweight="bold")
    for j, s in enumerate(srcs):
        ax.text(x0 + 1.6 + j * w + w / 2, -0.55, f"{int(M[s].sum()):,}", ha="center",
                va="center", fontsize=10, color=SRC_COLOR[s], fontweight="bold")
    ax.text(x0 + 1.6 + nc * w + 0.62, -0.55, f"{int(tot_r.sum()):,}", ha="center", va="center",
            fontsize=12, color=INK, fontweight="bold")
    ax.text(x0 + 1.6 + nc * w + 1.85, -0.55, f"{n_infants(st.patient_id):,}", ha="center",
            va="center", fontsize=11, color=INK2, fontweight="bold")

    ax.set_xlim(-0.1, x0 + 1.6 + nc * w + 2.5)
    ax.set_ylim(-1.1, nr + 1.5)
    ax.set_title("ROP corpus after classification — images per ICROP class, by source dataset",
                 pad=26, loc="left", x=-0.0)
    fig.text(0.5, -0.015,
             "Cell shade = image count (log-scaled, one hue). '·' = that source has no images "
             "of that class.\nInfant counts are distinct patients per class and do not sum — "
             "an infant can appear in several classes.\n"
             "HVDROPDB ships no patient IDs, so its 100 Normal images contribute 0 known infants.",
             ha="center", fontsize=9, color=INK2)
    save(fig, "01_classification_table.png")


# ------------------------------------------- fig 2: class composition (proportional)
def fig_distribution(df):
    """Composition, NOT magnitude. Normal is 4,358 of 9,323 images, so an absolute stacked bar
    squashes Stage 0/4/5 into invisible slivers. The question this figure answers is 'where did
    each class come from', which is a proportion — magnitude lives in figures 1 and 6."""
    st = df[df.cls.isin(STAGE_ORDER)]
    srcs = [s for s in SRC_COLOR if s in set(st.source)]
    M = pd.crosstab(st.cls, st.source).reindex(index=STAGE_ORDER, columns=srcs, fill_value=0)
    tot = M.sum(axis=1)
    P = M.div(tot.replace(0, np.nan), axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    y = np.arange(len(STAGE_ORDER))[::-1]
    left = np.zeros(len(STAGE_ORDER))
    for s in srcs:
        v = P[s].values
        ax.barh(y, v, left=left, height=0.62, color=SRC_COLOR[s], label=SRC_LABEL[s],
                edgecolor=SURF, linewidth=1.8)
        for yi, (l, val) in enumerate(zip(left, v)):
            if val >= 7:                     # direct label only where it fits
                ax.text(l + val / 2, y[yi], f"{val:.0f}%", ha="center", va="center",
                        fontsize=9.5, color="#0f0f0e", fontweight="bold")
        left += v
    for yi, cls in enumerate(STAGE_ORDER):
        ax.text(102, y[yi], f"{int(tot.iloc[yi]):,}", ha="left", va="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.text(102, len(STAGE_ORDER) - 0.15, "images", ha="left", va="bottom", fontsize=9.5,
            color=INK2, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([NICE[c] for c in STAGE_ORDER], fontsize=11)
    ax.set_xlabel("share of the class, by source dataset (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    bare(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9.5,
              ncol=6, labelcolor=INK2)
    ax.set_title("What each ICROP class is made of", pad=14, loc="left")
    fig.text(0.125, -0.145,
             "Proportional so the small classes stay readable; absolute counts are on the "
             "right and in figure 1.\nStage 4 and Stage 5 exist only because of ROP-VL — and "
             "come from a single hospital.", ha="left", fontsize=9.5, color=INK2)
    save(fig, "02_class_composition.png")


# ------------------------------------------------------ fig 3: infants, the real constraint
def fig_infants(df):
    st = df[df.cls.isin(STAGE_ORDER)]
    inf_all = st.groupby("cls").patient_id.apply(n_infants).reindex(STAGE_ORDER, fill_value=0)
    inf_ost = (st[st.source == "ostrava"].groupby("cls").patient_id.apply(n_infants)
               .reindex(STAGE_ORDER, fill_value=0))

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    y = np.arange(len(STAGE_ORDER))[::-1]
    ax.barh(y, inf_all.values, height=0.62, color="#3987e5", edgecolor=SURF, lw=1.6,
            label="all sources pooled")
    ax.barh(y, inf_ost.values, height=0.62, color="#d95926", edgecolor=SURF, lw=1.6,
            label="Ostrava alone (what the model trained on)")
    for yi, (a, o) in enumerate(zip(inf_all.values, inf_ost.values)):
        ax.text(a + 6, y[yi], f"{a:,}", ha="left", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
        if o:
            ax.text(o - 4 if o > 40 else o + 6, y[yi], f"{o}", ha="right" if o > 40 else "left",
                    va="center", fontsize=9.5,
                    color="#0f0f0e" if o > 40 else "#d95926", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[c] for c in STAGE_ORDER], fontsize=11)
    ax.set_xlabel("distinct infants")
    ax.set_xlim(0, inf_all.max() * 1.12)
    bare(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
    ax.set_title("Infants per class — the constraint that actually binds", pad=14, loc="left")
    fig.text(0.125, -0.025,
             "Stage 3 went from 2 infants to "
             f"{int(inf_all['stage_3'])}. AP-ROP rested on a single infant supplying 470 images.",
             ha="left", fontsize=9.5, color=INK2)
    save(fig, "03_infants_per_class.png")


# ------------------------------------------- fig 4: what is held out of the stage classes
def fig_excluded(df):
    ex = df[~df.cls.isin(STAGE_ORDER)].copy()
    ex["group"] = ex.cls.str.split("/").str[0]
    g = ex.groupby("group").size().sort_values()
    NICEX = {"_other_pathology": "Ostrava: NOT ROP\n(hemorrhage, toxoplasma,\nhamartoms, "
                                 "hypoplasia)",
             "_farfum_plus_disease": "FARFUM-RoP\n(plus-disease axis,\nno usable stage)",
             "_no_class_label": "Shantou ×2 + ORVS\n(segmentation only,\nno class labels)",
             "_rop_unstaged": "HVDROPDB ROP\n(binary only,\nno stage)"}
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(g))
    ax.barh(y, g.values, height=0.6, color="#8a8a80", edgecolor=SURF, lw=1.6)
    for i, v in enumerate(g.values):
        ax.text(v + 18, y[i], f"{v:,}", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([NICEX.get(k, k) for k in g.index], fontsize=9.5)
    ax.set_xlabel("images")
    ax.set_xlim(0, g.max() * 1.2)
    bare(ax)
    ax.set_title("Held OUT of the stage classes — and why", pad=14, loc="left")
    fig.text(0.125, -0.06,
             "None of these can be given an ICROP stage without inventing one. "
             f"{int(g.sum()):,} images in total.",
             ha="left", fontsize=9.5, color=INK2)
    save(fig, "04_held_separate.png")


# --------------------------------- fig 5: what the current binary ROP label really contains
def fig_binary(df):
    ost = df[df.source == "ostrava"].copy()
    lab = ost.native_label.str.extract(r"DG(\d+)")[0].astype(int)
    MEAN = {0: "physiological", 1: "ROP stage 0", 2: "ROP stage 1", 3: "ROP stage 2",
            4: "ROP stage 3", 8: "AP-ROP", 9: "treated (st.p. ROP)", 10: "hamartoms",
            11: "hemorrhage", 12: "hypoplasia n.II", 13: "toxoplasma"}
    pos = lab[lab != 0]
    c = pos.value_counts().sort_values(ascending=True)
    is_rop = {k: (1 <= k <= 9) for k in c.index}
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(c))
    cols = ["#d95926" if is_rop[k] else "#8a8a80" for k in c.index]
    ax.barh(y, c.values, height=0.62, color=cols, edgecolor=SURF, lw=1.6)
    for i, (k, v) in enumerate(zip(c.index, c.values)):
        ax.text(v + 14, y[i], f"{v:,}", va="center", fontsize=10.5, color=INK,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([MEAN.get(k, str(k)) for k in c.index], fontsize=10.5)
    ax.set_xlabel("images inside the single class labelled “ROP Detected”")
    ax.set_xlim(0, c.max() * 1.18)
    bare(ax)
    n_rop = int(sum(v for k, v in zip(c.index, c.values) if is_rop[k]))
    n_not = int(sum(v for k, v in zip(c.index, c.values) if not is_rop[k]))
    ax.text(0.985, 0.06,
            f"orange = genuine ROP  {n_rop:,}\ngrey = NOT ROP  {n_not:,}  "
            f"({100 * n_not / (n_rop + n_not):.0f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=INK2)
    ax.set_title("The current binary label is not a ROP label", pad=14, loc="left")
    fig.text(0.125, -0.035,
             f"{n_not:,} of {n_rop + n_not:,} images in the positive class carry a diagnosis "
             "that is not retinopathy of prematurity.\nRecovered by joining the Ostrava "
             "metadata on (infant ID, serie, device).", ha="left", fontsize=9.5, color=INK2)
    save(fig, "05_binary_label_composition.png")


# ----------------------------------------------------------- fig 6: corpus growth
def fig_growth(df):
    """Normal is excluded: at 2,980 -> 4,358 it is four times the largest disease class and
    flattens everything the figure is about. Its growth is stated in the caption instead."""
    st = df[df.cls.isin(STAGE_ORDER)]
    order = [c for c in STAGE_ORDER if c != "normal"]
    old_all = st[st.source == "ostrava"].groupby("cls").size()
    new_all = st.groupby("cls").size()
    old = old_all.reindex(order, fill_value=0)
    new = new_all.reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(order))
    ax.bar(x - 0.2, old.values, width=0.38, color="#d95926", edgecolor=SURF, lw=1.6,
           label="Ostrava alone (before)")
    ax.bar(x + 0.2, new.values, width=0.38, color="#3987e5", edgecolor=SURF, lw=1.6,
           label="all sources pooled (after)")
    for i in x:
        if new.values[i]:
            ax.text(i + 0.2, new.values[i] + 55, f"{int(new.values[i]):,}", ha="center",
                    fontsize=9, color=INK, fontweight="bold")
        if old.values[i]:
            ax.text(i - 0.2, old.values[i] + 55, f"{int(old.values[i]):,}", ha="center",
                    fontsize=9, color="#d95926")
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in order], fontsize=10.5, rotation=18, ha="right")
    ax.set_ylabel("images")
    ax.set_ylim(0, new.values.max() * 1.16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK2)
    ax.set_title("Corpus growth per ICROP class — disease classes only", pad=14, loc="left")
    fig.text(0.125, -0.08,
             f"Normal is omitted so the disease classes stay legible; it grew "
             f"{int(old_all.get('normal', 0)):,} → {int(new_all.get('normal', 0)):,}.\n"
             f"Disease classes: {int(old.sum()):,} → {int(new.sum()):,} images. "
             "Stage 4, Stage 5 and most of Stage 3 did not exist before.",
             ha="left", fontsize=9.5, color=INK2)
    save(fig, "06_corpus_growth.png")


def main():
    df = load()
    print(f"{len(df):,} rows from {MAN.relative_to(ROOT)}")
    for fn in (fig_table, fig_distribution, fig_infants, fig_excluded, fig_binary, fig_growth):
        try:
            fn(df)
        except Exception as e:  # noqa: BLE001
            print(f"  {fn.__name__}: FAILED {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
