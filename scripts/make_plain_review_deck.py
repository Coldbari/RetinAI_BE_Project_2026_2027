"""A plain-language version of the post-proposal review deck.

    .venv/bin/python scripts/make_plain_review_deck.py

Writes RetinAI_ROP_Plain_Review.pptx. This is a SEPARATE deck from
RetinAI_Post_Proposal_Review.pptx (scripts/generate_review_ppt_full.js) — it does not
share a file, a palette or a slide with it.

Two rules held while writing it:
  * every number is copied from graphs/rop_metrics (scripts/make_metrics_table.py), which
    reads the committed artifacts. Where the older deck quoted a single checkpoint, this
    one quotes the 5-checkpoint mean, so 0.546/0.493 become 0.520/0.523.
  * short sentences, ordinary words. A slide is a thing someone says out loud, not a page.

The figures at the end are the dark-theme PNGs already in graphs/, so they sit on the
slide background without a seam.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "RetinAI_ROP_Plain_Review.pptx"

# palette — identical to scripts/figtheme.py dark, so the graphs blend into the slide
BG = RGBColor(0x1A, 0x1A, 0x19)
CARD = RGBColor(0x26, 0x25, 0x23)
INK = RGBColor(0xFF, 0xFF, 0xFF)
INK2 = RGBColor(0xC3, 0xC2, 0xB7)
MUTE = RGBColor(0x8A, 0x8A, 0x80)
GREEN = RGBColor(0x19, 0x9E, 0x70)
RED = RGBColor(0xD9, 0x4F, 0x4F)
GOLD = RGBColor(0xC9, 0x85, 0x00)
BLUE = RGBColor(0x39, 0x87, 0xE5)
RULE = RGBColor(0x33, 0x32, 0x2F)

HEAD = "Georgia"
BODY = "Arial"

W, H = 13.333, 7.5
M = 0.62

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


# ── helpers ──────────────────────────────────────────────────────────────────────────
def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def text(s, body, x, y, w, h, size=14, color=INK2, bold=False, font=BODY,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, space=6, spacing=1.0):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    raw = body if isinstance(body, (list, tuple)) else [body]
    lines = []
    for item in raw:                       # a plain "\n" in a string means a new paragraph
        if isinstance(item, str):
            lines.extend(item.split("\n"))
        else:
            lines.append(item)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        p.line_spacing = spacing
        runs = line if isinstance(line, (list, tuple)) else [(line, {})]
        for txt, opt in runs:
            r = p.add_run()
            r.text = txt
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = opt.get("color", color)
    return box


def card(s, x, y, w, h, fill=CARD, radius=0.04):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                            Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def bar(s, x, y, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def head(s, kicker, title, sub=None):
    bar(s, M, 0.52, 0.5, 0.045, GREEN)
    text(s, kicker.upper(), M + 0.66, 0.44, 8.0, 0.3, size=10.5, color=GREEN,
         bold=True, font=BODY)
    text(s, title, M, 0.82, W - 2 * M, 0.66, size=26, color=INK, bold=True, font=HEAD)
    if sub:
        text(s, sub, M, 1.52, W - 2 * M - 1.2, 0.4, size=13.5, color=MUTE, font=BODY)


def bullets(s, items, x, y, w, size=14, gap=0.46, color=INK2, dot=GREEN):
    """items: str, or (lead, rest) where lead is bolded white.

    Spacing is measured from the wrapped length of each item rather than fixed, so a
    one-line bullet does not leave a hole under it and a four-line one does not collide
    with the next. 0.55 em per character is deliberately pessimistic for Arial.
    """
    cy = y
    for it in items:
        body = it[0] + "  " + it[1] if isinstance(it, tuple) else it
        bar(s, x, cy + 0.085, 0.10, 0.10, dot)
        if isinstance(it, tuple):
            runs = [(it[0] + "  ", {"bold": True, "color": INK}), (it[1], {})]
        else:
            runs = [(it, {})]
        chars_per_line = max(8, (w - 0.28) * 72 / (size * 0.55))
        rows = max(1, -(-len(body) // int(chars_per_line)))
        h = rows * size * 1.2 / 72
        text(s, [runs], x + 0.28, cy - 0.02, w - 0.28, h, size=size, color=color,
             spacing=1.14)
        cy += max(gap, h + 0.2)
    return cy


def stat(s, x, y, w, number, label, color=GREEN, nsize=34, lsize=11):
    drop = nsize * 1.3 / 72 + 0.05        # the number's own line height, so the label clears it
    text(s, number, x, y, w, drop, size=nsize, color=color, bold=True, font=HEAD)
    text(s, label, x, y + drop, w, 0.5, size=lsize, color=MUTE, font=BODY, spacing=1.1)


def figure(s, path, x, y, w, h):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    fw, fh = iw * scale, ih * scale
    s.shapes.add_picture(str(path), Inches(x + (w - fw) / 2), Inches(y + (h - fh) / 2),
                         Inches(fw), Inches(fh))


def figure_slide(kicker, title, path, caption):
    s = slide()
    bar(s, M, 0.52, 0.5, 0.045, GREEN)
    text(s, kicker.upper(), M + 0.66, 0.44, 8.0, 0.3, size=10.5, color=GREEN, bold=True)
    text(s, title, M, 0.78, W - 2 * M, 0.5, size=23, color=INK, bold=True, font=HEAD)
    figure(s, ROOT / path, M, 1.38, W - 2 * M, 4.92)
    text(s, caption, M, 6.44, W - 2 * M, 0.7, size=12, color=MUTE, spacing=1.15)
    return s


def footer(s, left, right="RetinAI · ROP staging"):
    text(s, left, M, H - 0.52, 7.5, 0.3, size=9.5, color=MUTE)
    text(s, right, W - M - 4.5, H - 0.52, 4.5, 0.3, size=9.5, color=MUTE,
         align=PP_ALIGN.RIGHT)


# ═════════════════════════════════════════════════════════════════════════════════════
# 1 — title
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
bar(s, 0, 0, W, 0.09, GREEN)
text(s, "POST-PROPOSAL REVIEW", M, 1.28, 8, 0.3, size=11.5, color=GREEN, bold=True)
text(s, "RetinAI", M, 1.72, 10, 1.05, size=54, color=INK, bold=True, font=HEAD)
text(s, "Grading ROP from a single baby eye photo", M, 2.78, 11, 0.6, size=23, color=INK2,
     font=HEAD)
text(s, "What works, what broke, and the one idea we are testing next.",
     M, 3.42, 10.5, 0.5, size=14, color=MUTE)
bar(s, M, 4.22, W - 2 * M, 0.02, RULE)
for i, (n, l, c) in enumerate([
        ("13,037", "photos collected", INK),
        ("4", "hospitals", INK),
        ("661", "photos held back,\nopened once", GREEN),
        ("6", "grades:\nNormal, 1, 2, 3, 4/5, AP-ROP", INK)]):
    stat(s, M + i * 3.07, 4.55, 2.9, n, l, color=c, nsize=32)
footer(s, "Slides read top to bottom. Numbers come from the metric sheet at the end.")

# ═════════════════════════════════════════════════════════════════════════════════════
# 2 — why it matters
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 1 · the problem", "A blindness we already know how to stop")
card(s, M, 2.0, 7.4, 4.3)
bullets(s, [
    ("The disease.", "Babies born too early can grow bad blood vessels in the retina. "
     "Left alone, the retina peels off and the child is blind for life."),
    ("The good news.", "Caught in time, it is treatable."),
    ("The catch.", "Someone has to look. There are not enough trained eye doctors, and "
     "two doctors often give the same eye different grades."),
    ("The AI catch.", "Models that score 98% in a paper usually fall apart at a hospital "
     "they have not seen."),
], M + 0.4, 2.35, 6.6, size=14, gap=0.92)

card(s, M + 7.8, 2.0, W - 2 * M - 7.8, 4.3)
text(s, "Who this is for", M + 8.2, 2.32, 3.4, 0.3, size=13, color=GREEN, bold=True)
bullets(s, [
    "NICU teams who need a quick second look at the cot side.",
    "Eye doctors who want a grade, not just a red flag.",
    "Other researchers, who need our splits to check us.",
], M + 8.2, 2.78, 3.5, size=12.5, gap=0.78, dot=INK2)
text(s, "It is a research prototype. It is not a device, and it does not decide anything "
        "on its own.", M + 8.2, 5.35, 3.5, 0.8, size=12, color=GOLD, italic=True,
     spacing=1.15)
footer(s, "Part 1 of 6")

# ═════════════════════════════════════════════════════════════════════════════════════
# 3 — what the thing does
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 2 · the system", "One photo in, one grade out")
steps = ["photo", "crop the circle", "pad to a square", "boost contrast", "384 px",
         "EfficientNetV2-S", "grade"]
x = M
for i, st in enumerate(steps):
    w = 1.62 if i else 1.1
    card(s, x, 2.0, w, 0.62, fill=CARD if i < 6 else RGBColor(0x12, 0x30, 0x1F))
    text(s, st, x, 2.0, w, 0.62, size=11.5, color=INK if i < 6 else GREEN,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, bold=(i == 6))
    x += w + 0.1
    if i < 6:
        text(s, "›", x - 0.12, 2.06, 0.2, 0.5, size=16, color=MUTE)

cols = [
    ("Three heads, one backbone", GREEN, [
        "A stage head that knows 1 → 2 → 3 is an order, so being three stages off costs "
        "far more than being one off.",
        "A separate yes/no head for AP-ROP, because AP-ROP is not a rung on that ladder.",
        "A training-only branch that is meant to punish the model for recognising the "
        "hospital. See slide 8 — it did not work.",
    ]),
    ("What it costs to run", BLUE, [
        "112 ms per photo on a laptop CPU. No GPU at the cot side.",
        "70 minutes per fold on a free Kaggle P100.",
        "20.5 million weights. Small enough to ship, big enough to matter.",
    ]),
    ("What we would not bend", GOLD, [
        "A doctor signs every case. The score never files itself.",
        "Split by baby, never by photo.",
        "The fourth hospital stays sealed until the rules are written down.",
    ]),
]
cw = (W - 2 * M - 0.6) / 3
for i, (t, c, items) in enumerate(cols):
    cx = M + i * (cw + 0.3)
    card(s, cx, 3.0, cw, 3.3)
    text(s, t, cx + 0.3, 3.28, cw - 0.6, 0.3, size=13, color=c, bold=True)
    bullets(s, items, cx + 0.3, 3.72, cw - 0.6, size=11.5, gap=0.82, dot=c)
footer(s, "Part 2 of 6")

# ═════════════════════════════════════════════════════════════════════════════════════
# 4 — leakage / split by baby
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 3 · what we found", "Why published scores are so high",
     "Most papers split their photos at random. That puts the same baby on both sides of the line.")
card(s, M, 2.25, 6.3, 3.5)
text(s, "The same model, scored three ways", M + 0.4, 2.55, 5.5, 0.3, size=13,
     color=INK, bold=True)
rows = [("98–100%", "what papers report when they split by photo", MUTE),
        ("78.6%", "our model, split by baby, at the three hospitals it trained on", INK),
        ("52.0%", "the same model at a fourth hospital it had never seen", RED)]
for i, (n, l, c) in enumerate(rows):
    y = 3.05 + i * 0.85
    text(s, n, M + 0.4, y, 2.1, 0.5, size=25, color=c, bold=True, font=HEAD)
    text(s, l, M + 2.6, y + 0.04, 3.3, 0.7, size=11.5, color=INK2, spacing=1.12)

card(s, M + 6.7, 2.25, W - 2 * M - 6.7, 3.5)
text(s, "Why that happens", M + 7.1, 2.55, 4.8, 0.3, size=13, color=GREEN, bold=True)
bullets(s, [
    "One baby can contribute up to 470 photos. Split at random and the model sees that "
    "baby's eye in training, then gets marked on it again in the test.",
    "It stops being a test of the disease and becomes a memory test.",
    "So every number we report is split by baby, and the confidence ranges resample "
    "babies, not photos.",
], M + 7.1, 3.0, 4.7, size=12, gap=0.95, dot=INK2)
text(s, "The fourth hospital was opened once, on 18 Aug 2026, under rules committed to "
        "git beforehand.", M + 7.1, 5.95, 5.0, 0.4, size=11.5, color=GOLD, italic=True)
footer(s, "Part 3 of 6 · finding 1")

# ═════════════════════════════════════════════════════════════════════════════════════
# 5 — the unit finding
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 3 · what we found", "The grade belongs to the eye. We stamped it on every photo.",
     "This is the main finding of the project, and it is not about the network at all.")
card(s, M, 2.2, 6.0, 2.2)
text(s, "Same model. Same predictions. Only the thing being scored changes.",
     M + 0.35, 2.45, 5.3, 0.4, size=12, color=MUTE)
stat(s, M + 0.35, 2.95, 2.6, "0.52", "agreement when you score\nphoto by photo", color=RED,
     nsize=40)
stat(s, M + 3.2, 2.95, 2.6, "0.80", "agreement when you pool\na whole eye first",
     color=GREEN, nsize=40)

card(s, M, 4.6, 6.0, 1.75)
bullets(s, [
    "A doctor grades the eye. That single grade is copied onto all four photos of it.",
    "ROP grows at the outer edge. A photo of the middle of the eye cannot show it — "
    "but it still carries the label.",
], M + 0.35, 4.85, 5.4, size=12, gap=0.62, dot=GOLD)

card(s, M + 6.4, 2.2, W - 2 * M - 6.4, 4.15)
text(s, "How often a diseased eye is called Normal, by where the camera pointed",
     M + 6.8, 2.5, 4.9, 0.55, size=13, color=INK, bold=True, spacing=1.12)
views = [("Temporal", "looks straight at the ridge", 12.7, GREEN),
         ("Nasal", "looks to the nose side", 63.4, GOLD),
         ("Posterior pole", "looks at the middle, away from it", 71.7, RED)]
for i, (name, note, pct, c) in enumerate(views):
    y = 3.28 + i * 1.0
    text(s, name, M + 6.8, y, 2.4, 0.3, size=13, color=INK, bold=True)
    text(s, note, M + 6.8, y + 0.28, 3.1, 0.3, size=11, color=MUTE)
    text(s, f"{pct}%", M + 10.0, y - 0.04, 1.7, 0.4, size=20, color=c, bold=True,
         font=HEAD, align=PP_ALIGN.RIGHT)
    bar(s, M + 6.8, y + 0.62, 4.9 * pct / 100, 0.1, c)
text(s, "The mistakes point somewhere. That is what tells you it is a labelling problem "
        "and not a blind model.", M + 6.8, 6.4, 5.0, 0.4, size=11, color=MUTE, italic=True)
footer(s, "Part 3 of 6 · finding 2 · quadratic weighted kappa on the sealed hospital")

# ═════════════════════════════════════════════════════════════════════════════════════
# 6 — confident misses
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 3 · what we found", "It is surest exactly when it is most wrong")
card(s, M, 2.2, 5.6, 4.1)
text(s, "Diseased eyes called \u201cNormal\u201d", M + 0.4, 2.5, 4.8, 0.3, size=13, color=INK,
     bold=True)
stat(s, M + 0.4, 2.95, 2.4, "186", "missed cases", color=RED, nsize=38)
stat(s, M + 3.0, 2.95, 2.4, "141", "of them above\n90% confidence", color=RED, nsize=38)
text(s, "Average confidence on a miss: 0.93 — higher than when it is right.",
     M + 0.4, 4.5, 4.9, 0.4, size=12.5, color=INK2)
bar(s, M + 0.4, 5.05, 4.8, 0.02, RULE)
text(s, "So \u201conly act on confident answers\u201d throws away almost nothing. You cannot "
        "threshold this away.", M + 0.4, 5.25, 4.8, 0.8, size=12.5, color=GOLD,
     spacing=1.15)

card(s, M + 6.0, 2.2, W - 2 * M - 6.0, 4.1)
text(s, "What we are doing about it", M + 6.4, 2.5, 5.0, 0.3, size=13, color=GREEN,
     bold=True)
bullets(s, [
    ("Pool the eye.", "Grade every photo of an eye and take the worst. On the sealed "
     "hospital that alone cuts missed eyes to 17.8%."),
    ("Fit the threshold on purpose.", "Pick the highest cut-off that still catches 90% "
     "of disease at both sites, not the one that looks best at home."),
    ("Say what the score is.", "It is a ranking, not a probability. The app labels it "
     "that way until temperature scaling is fitted."),
], M + 6.4, 2.95, 5.0, size=12, gap=1.0)
footer(s, "Part 3 of 6 · finding 3 · counts from the 6,700-photo training pool")

# ═════════════════════════════════════════════════════════════════════════════════════
# 7 — two fixes that did not work
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 3 · what we found", "Two things we thought we had fixed, and had not",
     "Both are reported as failures. They cost us weeks, so they belong on a slide.")
left = [("The clever stage head", RED, [
    "Our ordinal head and a plain softmax score the same: 0.8964 against 0.8962, "
    "p = 0.98.",
    "The plain one is better calibrated — 0.035 against our 0.109.",
    "Which one wins on F1 flips when you only change the random seed. That is noise, "
    "not a result.",
    "We keep ours for one honest reason: it keeps AP-ROP off the stage ladder.",
])]
right = [("The anti-hospital branch", RED, [
    "A simple probe says the hospital fingerprint is gone. It looked like a win.",
    "A fairer probe — one that holds the disease mix fixed — still reads the hospital "
    "at 0.87, against 0.85 with no defence at all.",
    "Nothing was removed. The first probe was just measuring the disease.",
    "A weak test that agrees with you is worse than no test.",
])]
for i, group in enumerate([left, right]):
    for t, c, items in group:
        cw2 = (W - 2 * M - 0.3) / 2
        cx = M + i * (cw2 + 0.3)
        card(s, cx, 2.2, cw2, 4.1)
        bar(s, cx, 2.2, 0.06, 4.1, c)
        text(s, t, cx + 0.4, 2.5, cw2 - 0.8, 0.3, size=14, color=INK, bold=True)
        bullets(s, items, cx + 0.4, 2.95, cw2 - 0.8, size=12, gap=0.72, dot=c)
footer(s, "Part 3 of 6 · finding 4")

# ═════════════════════════════════════════════════════════════════════════════════════
# 8 — green channel
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 4 · the next test", "The green-channel idea, and how we plan to kill it",
     "The one new thing we are trying. We would rather it fail on our bench than in review.")
card(s, M, 2.2, 6.4, 4.1)
text(s, "The claim", M + 0.4, 2.5, 5.6, 0.3, size=13, color=GREEN, bold=True)
text(s, "Blood vessels show up with the most contrast in the green channel of a fundus "
        "photo. Lots of papers say so. It may well be true.",
     M + 0.4, 2.9, 5.6, 0.8, size=13, color=INK2, spacing=1.15)
text(s, "Why the obvious version breaks", M + 0.4, 3.8, 5.6, 0.3, size=13, color=GOLD,
     bold=True)
text(s, "Feeding one channel to a pretrained network throws away two thirds of its first "
        "layer and gives it colour statistics it has never seen.",
     M + 0.4, 4.2, 5.6, 0.8, size=13, color=INK2, spacing=1.15)
text(s, "What we will actually run", M + 0.4, 5.1, 5.6, 0.3, size=13, color=BLUE,
     bold=True)
text(s, "Take green, boost its contrast, copy it into all three channels. Everything "
        "else — crop, padding, size, seed, folds — stays identical.",
     M + 0.4, 5.5, 5.6, 0.8, size=13, color=INK2, spacing=1.15)

card(s, M + 6.9, 2.2, W - 2 * M - 6.9, 4.1, fill=RGBColor(0x30, 0x1C, 0x05))
text(s, "We drop the idea if either of these happens", M + 7.3, 2.5, 4.9, 0.3, size=13,
     color=GOLD, bold=True)
bullets(s, [
    "It makes the hospital easier to guess, not harder.",
    "It loses to the contrast setup we already use, scored on temporal photos only — "
    "the ones that can actually see the disease.",
], M + 7.3, 2.95, 4.9, size=12.5, gap=1.0, dot=GOLD)
bar(s, M + 7.3, 5.0, 4.8, 0.02, RULE)
text(s, "Written down before the run, so we cannot move the line afterwards. "
        "Three seeds, folds 1–4, one week of GPU.",
     M + 7.3, 5.25, 4.8, 1.0, size=12, color=INK2, spacing=1.15)
footer(s, "Part 4 of 6")

# ═════════════════════════════════════════════════════════════════════════════════════
# 9 — targets and rules
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 5 · the bar", "What counts as done, decided in advance")
tiles = [("≥ 0.85", "agreement across our\nown five folds", GREEN),
         ("≥ 0.80", "agreement per eye at a\nhospital we never trained on", GREEN),
         ("≤ 0.05", "calibration error, after\ntemperature scaling", GREEN)]
for i, (n, l, c) in enumerate(tiles):
    cx = M + i * 4.1
    card(s, cx, 2.15, 3.8, 1.5)
    text(s, n, cx + 0.35, 2.35, 3.1, 0.55, size=30, color=c, bold=True, font=HEAD)
    text(s, l, cx + 0.35, 2.95, 3.1, 0.6, size=11.5, color=MUTE, spacing=1.1)

card(s, M, 3.85, 6.2, 2.55)
text(s, "Rules we cannot bend later", M + 0.4, 4.1, 5.4, 0.3, size=13, color=INK,
     bold=True)
bullets(s, [
    "No baby appears in two splits. No near-duplicate photos across them.",
    "The model must beat a cheat baseline that reads only file metadata by 2.5×.",
    "The sealed hospital is opened once. It has been, and it is not reopening.",
], M + 0.4, 4.52, 5.4, size=12, gap=0.58, dot=GREEN)

card(s, M + 6.6, 3.85, W - 2 * M - 6.6, 2.55)
text(s, "What we are not claiming", M + 7.0, 4.1, 4.9, 0.3, size=13, color=RED,
     bold=True)
bullets(s, [
    "Stage 4/5 and AP-ROP at the new hospital: 4 and 12 photos. Too few to say anything.",
    "Heatmaps are not evidence yet — attention lands on the retina barely more than "
    "chance would put it there.",
    "Eyes already treated, and non-ROP damage, are out of scope.",
], M + 7.0, 4.52, 4.9, size=12, gap=0.58, dot=RED)
footer(s, "Part 5 of 6")

# ═════════════════════════════════════════════════════════════════════════════════════
# 10 — team and plan
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
head(s, "part 6 · who and when", "Who owns what, and the next four weeks")
owners = [("Modelling", "backbone, heads, calibration"),
          ("Data and audits", "splits, leakage checks, de-identification"),
          ("Clinical analysis", "per-view breakdown, eye pooling, heatmaps"),
          ("App and deployment", "Flask inference, PDF reports, CI gates")]
for i, (t, d) in enumerate(owners):
    cx = M + (i % 2) * 3.15
    cy = 2.2 + (i // 2) * 1.25
    card(s, cx, cy, 2.95, 1.05)
    text(s, t, cx + 0.28, cy + 0.2, 2.5, 0.3, size=12.5, color=GREEN, bold=True)
    text(s, d, cx + 0.28, cy + 0.55, 2.5, 0.4, size=10.5, color=MUTE, spacing=1.1)
text(s, "Four lanes. Everyone can defend their own numbers.",
     M, 4.62, 6.1, 0.3, size=11.5, color=MUTE, italic=True)
card(s, M, 5.0, 6.1, 1.3)
text(s, "Already banked", M + 0.28, 5.2, 3.0, 0.3, size=12.5, color=GREEN, bold=True)
bullets(s, [
    "13,037 photos filtered down to 8,024 usable ones across 4 hospitals.",
    "Five backbones benchmarked; the sealed test opened once; 275 audit files in the repo.",
], M + 0.28, 5.58, 5.55, size=11, gap=0.34, dot=INK2)

plan = [("This week", "laptop only", [
    "Scrub the last names out of the repo history.",
    "Fix the sampler in the structured config.",
    "Fit temperature scaling so the score stops lying."], GREEN),
    ("Weeks 1–2", "GPU", [
        "Run the green-channel test against what we use now.",
        "Three seeds, folds 1–4, nothing else changed."], BLUE),
    ("Weeks 3–4", "clinic", [
        "Ask Narayana Nethralaya to re-grade about 300 of our images.",
        "Compare their grades against ours, per eye."], GOLD)]
for i, (when, where, items, c) in enumerate(plan):
    cy = 2.2 + i * 1.42
    card(s, M + 6.5, cy, W - 2 * M - 6.5, 1.28)
    bar(s, M + 6.5, cy, 0.06, 1.28, c)
    text(s, when, M + 6.85, cy + 0.18, 1.7, 0.3, size=13, color=c, bold=True)
    text(s, where, M + 8.5, cy + 0.22, 1.4, 0.3, size=10.5, color=MUTE)
    text(s, ["· " + t for t in items], M + 6.85, cy + 0.55, 5.0, 0.7, size=11,
         color=INK2, space=2, spacing=1.1)
footer(s, "Part 6 of 6 · target: ISBI 2027")

# ═════════════════════════════════════════════════════════════════════════════════════
# 11 — divider before the figures
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
bar(s, M, 3.0, 0.5, 0.045, GREEN)
text(s, "THE EVIDENCE", M + 0.66, 2.92, 8, 0.3, size=11.5, color=GREEN, bold=True)
text(s, "Six figures, then every number on one page",
     M, 3.3, 11.6, 0.8, size=32, color=INK, bold=True, font=HEAD)
text(s, "Each one is drawn straight from a committed result file. Nothing here was "
        "redrawn by hand for the slide.", M, 4.35, 9.5, 0.5, size=14, color=MUTE)
footer(s, "Figures 1–7")

# ═════════════════════════════════════════════════════════════════════════════════════
# 12+ — figures
# ═════════════════════════════════════════════════════════════════════════════════════
figure_slide("figure 1", "The same answers, scored two ways",
             "graphs/paper/10_evaluation_unit.png",
             "Nothing about the model changes between the red and the green bar. Only what "
             "gets scored: a photo, or the eye it came from. The gap holds for both heads "
             "and both seeds, which is why we call it a measurement result and not a lucky run.")

figure_slide("figure 2", "The misses have a direction",
             "graphs/paper/11_camera_view_misses.png",
             "Photos pointed at the outer edge, where the ridge grows, are missed 13% of the "
             "time. Photos pointed at the middle of the eye are missed 72% of the time. A "
             "model that simply could not see disease would miss both equally.")

figure_slide("figure 3", "Where the mistakes land",
             "graphs/paper/12_confusion_locked.png",
             "The damage is in the Normal column, not around the diagonal. Most stage-1 "
             "frames get called Normal while the Normal row itself is 99% right — which is "
             "exactly what you would expect if some diseased frames do not contain the disease.")

figure_slide("figure 4", "Our clever head does not beat a plain one",
             "graphs/paper/14_head_equivalence.png",
             "Identical folds, sampling and preprocessing; only the head differs. The "
             "difference changes sign when the seed changes. An earlier version of this "
             "project reported one seed and claimed a win.")

figure_slide("figure 5", "What survives the trip to a new hospital",
             "graphs/paper/05_roc_transfer.png",
             "Both models scored on the same 663 photos from a hospital neither had seen. "
             "Telling a sick eye from a healthy one travels reasonably well. The old head we "
             "retired lands near a coin toss.")

figure_slide("figure 6", "The hospital fingerprint is still there",
             "graphs/paper/15_site_probe.png",
             "A negative result, reported as one. At every strength of the anti-hospital "
             "branch, a probe that controls for disease still identifies the hospital. The "
             "naive probe scores lower — which is the trap: a weak test understates the leak.")

figure_slide("figure 7", "Every measured number, on one page",
             "graphs/rop_metrics/rop_metric_sheet.png",
             "Each table names the file it was copied from. Red marks the rows that are "
             "dangerous rather than merely disappointing.")

# ═════════════════════════════════════════════════════════════════════════════════════
# closing
# ═════════════════════════════════════════════════════════════════════════════════════
s = slide()
bar(s, M, 1.5, 0.5, 0.045, GREEN)
text(s, "IN ONE LINE", M + 0.66, 1.42, 8, 0.3, size=11.5, color=GREEN, bold=True)
text(s, ["\u201cIs anything wrong with this eye?\u201d travels to a new hospital.",
         "\u201cWhich stage is it?\u201d does not \u2014 yet."],
     M, 1.95, 11.9, 1.6, size=29, color=INK, bold=True, font=HEAD, spacing=1.12, space=10)
card(s, M, 3.9, W - 2 * M, 1.9)
pairs = [("0.970 → 0.835", "any disease vs none, at home then away", GREEN),
         ("0.897 → 0.523", "which stage, scored photo by photo", RED),
         ("0.523 → 0.801", "the same photos, pooled per eye", GREEN)]
for i, (n, l, c) in enumerate(pairs):
    cx = M + 0.45 + i * 3.9
    text(s, n, cx, 4.2, 3.6, 0.5, size=26, color=c, bold=True, font=HEAD)
    text(s, l, cx, 4.78, 3.6, 0.6, size=12, color=MUTE, spacing=1.1)
text(s, "The next four weeks are about closing the second line, and we know which "
        "experiment does it.", M, 6.1, 11.5, 0.5, size=14, color=INK2)
footer(s, "Thank you — questions welcome", "RetinAI · ROP staging · Sept 2026")

prs.save(OUT)
print("wrote", OUT)
