const pptxgen = require("pptxgenjs");

const BLUE = "1F3864";   // headings
const TXT = "262626";    // body
const GREY = "595959";   // muted
const LINE = "BFBFBF";
const LGREY = "F2F2F2";
const RED = "9C2A20";
const GREEN = "2E6B3E";
const F = "Calibri";

const W = 13.333, H = 7.5, M = 0.6, CW = W - 2 * M;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "RetinAI - ROP Staging";

function S() { const s = pres.addSlide(); s.background = { color: "FFFFFF" }; return s; }

function head(s, t) {
  s.addText(t, {
    x: M, y: 0.38, w: CW, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 26, bold: true, color: BLUE, valign: "middle",
  });
}

function sub(s, x, y, w, t) {
  s.addText(t, {
    x, y, w, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, bold: true, color: BLUE, valign: "middle",
  });
}

function bullets(s, x, y, w, h, items, size) {
  const runs = items.map((t, i) => {
    if (typeof t === "string") {
      return { text: t, options: { bullet: true, breakLine: i < items.length - 1 } };
    }
    return { text: t.text, options: Object.assign({ bullet: true, breakLine: i < items.length - 1 }, t.o || {}) };
  });
  s.addText(runs, {
    x, y, w, h, isTextBox: true, margin: 0,
    fontFace: F, fontSize: size || 13, color: TXT, paraSpaceAfter: 6, valign: "top",
  });
}

function hdr(a) {
  return a.map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: BLUE }, fontSize: 11.5 } }));
}

function table(s, rows, o) {
  s.addTable(rows, Object.assign({
    x: M, w: CW, fontFace: F, fontSize: 11.5, color: TXT, valign: "middle",
    border: { type: "solid", color: LINE, pt: 0.75 }, autoPage: false,
  }, o));
}

// ---------------------------------------------------------------- 1 title
{
  const s = S();
  s.addText("RetinAI", {
    x: M, y: 2.15, w: CW, h: 0.8, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 46, bold: true, color: BLUE,
  });
  s.addText("Deep learning for ICROP staging of Retinopathy of Prematurity", {
    x: M, y: 2.95, w: CW, h: 0.45, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 20, color: TXT,
  });
  s.addShape(pres.ShapeType.line, {
    x: M, y: 3.62, w: 4.2, h: 0, line: { color: LINE, width: 1 },
  });
  s.addText([
    { text: "B.E. Project — Semester VII Review", options: { bold: true, breakLine: true } },
    { text: "Team: [Member 1] · [Member 2] · [Member 3] · [Member 4]", options: { breakLine: true } },
    { text: "Guide: [Guide name]        Team number: ____", options: { breakLine: true } },
    { text: "Department of Computer Engineering, VESIT", options: { color: GREY } },
  ], { x: M, y: 3.85, w: CW, h: 1.4, isTextBox: true, margin: 0, fontFace: F, fontSize: 14, color: TXT, lineSpacing: 22 });

  s.addText("Research prototype. Not a medical device and not for clinical use.", {
    x: M, y: 6.6, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11, italic: true, color: GREY,
  });
  s.addNotes("Introduce the team and the disease in one line. Say up front it is a research prototype.");
}

// ---------------------------------------------------------------- 2 problem
{
  const s = S();
  head(s, "The problem");

  sub(s, M, 1.15, 6.0, "What ROP is");
  bullets(s, M, 1.5, 6.0, 1.9, [
    "Abnormal retinal blood-vessel growth in premature babies.",
    "A leading preventable cause of childhood blindness.",
    "Treatable, but only inside a short window of a few weeks.",
    "Graded on the ICROP scale: Normal, Stage 1, 2, 3, 4/5, and AP-ROP.",
  ]);

  sub(s, M, 3.55, 6.0, "Why it is still a problem");
  bullets(s, M, 3.9, 6.0, 1.6, [
    "The shortage is of trained graders, not of cameras.",
    "Grading is genuinely hard: expert agreement is only κ = 0.24.",
    "Stages 1 and 2 differ by whether a thin line has become a ridge.",
  ]);

  sub(s, M + 6.6, 1.15, CW - 6.6, "Who would use it");
  bullets(s, M + 6.6, 1.5, CW - 6.6, 1.2, [
    "NICU screening teams — triage, which eyes to review first.",
    "Paediatric ophthalmologists — a consistent second reader.",
  ]);

  sub(s, M + 6.6, 2.9, CW - 6.6, "Constraints we worked under");
  const rows = [
    ["Data", "No ethics approval or hospital tie-up available. Public datasets only."],
    ["Labels", "Used as released. No clinician re-graded them for us."],
    ["Classes", "Only 89 Stage 4/5 images exist publicly. Imbalance 12.5 : 1."],
    ["Compute", "Free Kaggle GPU. Deployment had to be CPU-only."],
    ["Use", "Not a medical device. No prospective testing."],
  ];
  table(s, rows, {
    x: M + 6.6, y: 3.25, w: CW - 6.6, colW: [1.0, 4.63], rowH: 0.42, fontSize: 11,
  });
  s.addNotes("Key line: the bottleneck is readers, not cameras. And experts agree at 0.24, which matters later.");
}

// ---------------------------------------------------------------- 3 existing work + gap
{
  const s = S();
  head(s, "Existing work, and what is missing");

  const rows = [
    hdr(["Work", "What it reports", "Limitation"]),
    ["Vahidmoghadam et al., 2026\n(same Ostrava data as us)", "0.98 stage accuracy", "Split at image level on data with ~32 frames per infant. No external test."],
    ["Wang et al., JAMA Netw Open, 2021", "F1 0.956 image / 0.915 eye / 0.898 patient", "Reports all three units, but treats the difference as small."],
    ["Chen et al., Ophthalmol Retina, 2021", "Stage accuracy across cameras and populations", "Corpus is not public, so the audit cannot be reproduced."],
    ["Gschliesser et al., 2015", "Expert-to-expert agreement κ = 0.24", "The human ceiling. Models reporting > 0.95 need a second look."],
  ];
  table(s, rows, {
    y: 1.15, colW: [2.9, 3.4, 5.83], rowH: [0.34, 0.62, 0.52, 0.52, 0.52], fontSize: 11,
  });

  sub(s, M, 3.85, CW, "The gap");
  s.addText("There is no public ROP staging benchmark that is grouped by patient, audited for shortcuts, and tested at a hospital it never trained on.", {
    x: M, y: 4.16, w: CW, h: 0.46, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 13, color: TXT,
  });

  sub(s, M, 4.68, CW, "We checked this on our own data before training anything");
  const rows2 = [
    ["AUC 0.911", "A classifier using only image width and height — no pixels — nearly solves the task."],
    ["0.997", "Accuracy of guessing the hospital from file metadata (chance is 0.333)."],
    ["43%", "Of one corpus's \"ROP-positive\" images are not ROP, but other pathology."],
    ["663 / 663", "Our own earlier screening model flagged every image at a new hospital, healthy ones included. It was retired."],
  ];
  table(s, rows2, {
    y: 5.02, colW: [1.35, 10.78], rowH: 0.42, fontSize: 11.5,
  });
  s.addNotes("The point is not that these papers are bad. It is that four of them share one blind spot, and the fifth explains why it matters.");
}

// ---------------------------------------------------------------- 4 objectives + scope
{
  const s = S();
  head(s, "Objectives and scope");

  const rows = [
    hdr(["#", "Objective", "Target set in advance", "Status"]),
    ["1", "Build an audited 6-class ICROP dataset from public sources", "Every exclusion counted; zero patient or image leakage", { text: "Done", options: { color: GREEN, bold: true } }],
    ["2", "Train a model that learns from pixels, not from shortcuts", "Macro-F1 above the metadata-only floor of 0.187", { text: "Done — 0.692", options: { color: GREEN, bold: true } }],
    ["3", "Test at a hospital never used in training", "Sealed half opened once, under a protocol fixed beforehand", { text: "Done — 18 Aug", options: { color: GREEN, bold: true } }],
    ["4", "Check whether the scoring unit matters more than the architecture", "Effect must hold across 2 architectures and 2 seeds", { text: "Done — 4 of 4", options: { color: GREEN, bold: true } }],
    ["5", "Build a CPU-only prototype with an explanation and a report", "Under 1 second per 6-frame examination on a laptop", { text: "Done — 0.8 s", options: { color: GREEN, bold: true } }],
    ["6", "Submit the result as a paper", "ISBI 2027, four pages", { text: "Due 26 Oct", options: { color: RED, bold: true } }],
  ];
  table(s, rows, {
    y: 1.15, colW: [0.4, 4.4, 5.13, 2.2], rowH: [0.32, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44], fontSize: 11.5,
  });

  sub(s, M, 4.35, 5.9, "In scope");
  bullets(s, M, 4.68, 5.9, 1.6, [
    "ROP only — 6-class ICROP staging plus a binary screening score",
    "Retrospective, public, licensed, de-identified images",
    "Single photograph in; eye-level aggregation analysed",
  ], 12.5);

  sub(s, M + 6.2, 4.35, CW - 6.2, "Out of scope, and why");
  bullets(s, M + 6.2, 4.68, CW - 6.2, 1.9, [
    "Glaucoma — dropped. The data double-counted eyes and inflated the AUC.",
    "Diabetic retinopathy — built and parked. Not this year's focus.",
    "Zone and plus disease — no dataset we can get has these labels.",
    "Any clinical or commercial use.",
  ], 12.5);
  s.addNotes("Mention that glaucoma was dropped rather than reported with a broken number.");
}

// ---------------------------------------------------------------- 5 dataset
{
  const s = S();
  head(s, "The dataset");

  sub(s, M, 1.15, CW, "From 11 datasets down to what we could actually train on");
  const steps = [
    ["13,037", "images acquired\nfrom 11 datasets"],
    ["10,427", "have an ICROP\nstage label"],
    ["8,024", "eligible after\nclinical exclusions"],
    ["3,112", "used for training\nafter frame caps"],
  ];
  steps.forEach((st, i) => {
    const x = M + i * 2.45;
    s.addShape(pres.ShapeType.rect, {
      x, y: 1.52, w: 2.15, h: 0.95, fill: { color: LGREY }, line: { color: LINE, width: 0.75 },
    });
    s.addText(st[0], {
      x: x + 0.1, y: 1.6, w: 1.95, h: 0.35, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 19, bold: true, color: BLUE, align: "center",
    });
    s.addText(st[1], {
      x: x + 0.1, y: 1.95, w: 1.95, h: 0.46, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 10, color: GREY, align: "center",
    });
    if (i < 3) {
      s.addText("→", {
        x: x + 2.15, y: 1.52, w: 0.3, h: 0.95, isTextBox: true, margin: 0,
        fontFace: F, fontSize: 16, color: GREY, align: "center", valign: "middle",
      });
    }
  });
  s.addText("Removed: 1,295 images of other pathology, 1,063 treated eyes, and a cap of 2 normal frames per baby.", {
    x: M, y: 2.58, w: 9.6, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11, italic: true, color: GREY,
  });

  sub(s, M, 3.05, 6.4, "Images per class, after caps");
  const cls = [
    hdr(["Class", "Train", "Sealed test"]),
    ["Normal", "1,109", "232"],
    ["Stage 1", "515", "105"],
    ["Stage 2", "696", "164"],
    ["Stage 3", "535", "144"],
    ["Stage 4/5", "89", { text: "4 — too few", options: { color: RED } }],
    ["AP-ROP", "168", { text: "12 — too few", options: { color: RED } }],
  ];
  table(s, cls, { y: 3.4, w: 5.0, colW: [1.8, 1.6, 1.6], rowH: 0.31, fontSize: 11 });

  sub(s, M + 5.6, 3.05, CW - 5.6, "How it was split");
  const sp = [
    hdr(["Group", "Images", "Babies", "Purpose"]),
    ["Training", "3,112", "1,528", "3 hospitals, model learns from these"],
    ["Cross-validation", "6,700", "1,528", "5 folds, grouped by baby"],
    ["External dev", "663", "116", "4th hospital — audits and threshold"],
    ["Sealed test", "661", "116", "4th hospital — opened exactly once"],
  ];
  table(s, sp, { x: M + 5.6, y: 3.4, w: CW - 5.6, colW: [1.5, 0.85, 0.8, 3.38], rowH: 0.36, fontSize: 11 });

  bullets(s, M + 5.6, 5.4, CW - 5.6, 1.3, [
    "Split by baby, never by photograph — one baby has up to 470 frames.",
    "Zero shared babies or images between any two groups, verified.",
    "Three hospitals train, a fourth is held out completely.",
  ], 12);
  s.addNotes("Explain the cap: Ostrava photographs one baby ~29 times, so a few babies would otherwise dominate training.");
}

// ---------------------------------------------------------------- 6 method
{
  const s = S();
  head(s, "Method");

  sub(s, M, 1.15, CW, "Preprocessing");
  const pp = ["Circle crop", "Letterbox to square", "CLAHE contrast", "Resize 384 × 384", "Normalise"];
  pp.forEach((p, i) => {
    const x = M + i * 2.45;
    s.addShape(pres.ShapeType.rect, {
      x, y: 1.48, w: 2.15, h: 0.45, fill: { color: LGREY }, line: { color: LINE, width: 0.75 },
    });
    s.addText(p, {
      x: x + 0.08, y: 1.48, w: 1.99, h: 0.45, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 11.5, color: TXT, align: "center", valign: "middle",
    });
    if (i < 4) s.addText("→", { x: x + 2.15, y: 1.48, w: 0.3, h: 0.45, isTextBox: true, margin: 0, fontFace: F, fontSize: 14, color: GREY, align: "center", valign: "middle" });
  });

  sub(s, M, 2.15, 6.0, "Model");
  const md = [
    ["Backbone", "EfficientNetV2-S, ImageNet-pretrained, 20.5 M parameters"],
    ["Head 1", "CORN ordinal head over the stage ladder"],
    ["Head 2", "Separate binary branch for AP-ROP, which is not a stage"],
    ["Head 3", "Hospital adversary (DANN) — used in training only"],
    ["Training", "AdamW, lr 3e-4, OneCycle, batch 16, 20 epochs, seed 42"],
    ["Output", "Predicted stage, 6 class scores, and an \"any disease\" score"],
  ];
  table(s, md, { y: 2.5, w: 6.0, colW: [1.15, 4.85], rowH: 0.38, fontSize: 11 });

  sub(s, M + 6.4, 2.15, CW - 6.4, "Choices we had to defend");
  const ch = [
    hdr(["Choice", "Why"]),
    ["Split by baby", "An image-level split lets the model recognise the baby. That is how papers reach 98–100%."],
    ["Natural class prevalence", "We tried a balanced sampler. It collapsed the model — QWK fell to 0.121."],
    ["Hold out a whole hospital", "Cross-validation shares cameras and graders, so it cannot test transfer."],
    ["Split the held-out site in half", "One half to look at, one half sealed. Looking at a test set makes it training data."],
    ["Last-epoch weights", "No checkpoint picked using validation data. Costs accuracy, keeps the number honest."],
  ];
  table(s, ch, { x: M + 6.4, y: 2.5, w: CW - 6.4, colW: [1.9, 3.83], rowH: [0.3, 0.6, 0.5, 0.5, 0.6, 0.6], fontSize: 10.5 });

  s.addNotes("The ordinal head is for taxonomy, not accuracy — we measured it as no better than a plain softmax and we say so.");
}

// ---------------------------------------------------------------- 7 results
{
  const s = S();
  head(s, "Results");

  const rows = [
    hdr(["", "Cross-validation\n3 training hospitals", "Sealed test\nnew hospital, opened once"]),
    ["Accuracy", "0.786", { text: "0.546", options: { color: RED } }],
    ["Grader agreement (QWK)", "0.897", "0.813 per eye / 0.493 per image"],
    ["\"Any disease\" AUC", "0.970", { text: "0.835", options: { color: GREEN } }],
    ["Normal recall", "0.951", { text: "0.996", options: { color: GREEN } }],
    ["Macro-F1 (shortcut floor 0.187)", "0.692", "—"],
    ["Stage 1 / 2 / 3 recall", "0.50 / 0.63 / 0.71", { text: "0.28 / 0.31 / 0.29", options: { color: RED } }],
  ];
  table(s, rows, {
    y: 1.15, w: 7.1, colW: [2.7, 2.1, 2.3], rowH: [0.55, 0.38, 0.46, 0.38, 0.38, 0.38, 0.38], fontSize: 11,
  });

  s.addChart(pres.ChartType.bar, [
    { name: "Cross-validation", labels: ["Accuracy", "QWK (eye)", "Any-disease AUC", "Normal recall"], values: [0.786, 0.897, 0.970, 0.951] },
    { name: "Sealed test", labels: ["Accuracy", "QWK (eye)", "Any-disease AUC", "Normal recall"], values: [0.546, 0.813, 0.835, 0.996] },
  ], {
    x: M + 7.4, y: 1.1, w: CW - 7.4, h: 3.3,
    barDir: "col", chartColors: ["1F3864", "A6A6A6"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 9, dataLabelColor: TXT,
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: TXT,
    catAxisLabelColor: TXT, catAxisLabelFontSize: 9.5, valAxisLabelColor: GREY, valAxisLabelFontSize: 9,
    valAxisMaxVal: 1.1, valAxisMinVal: 0, valGridLine: { color: "E6E6E6", size: 0.75 }, catGridLine: { style: "none" },
    barGapWidthPct: 55,
  });

  sub(s, M, 4.65, CW, "What this says");
  bullets(s, M, 5.0, CW, 1.6, [
    "Screening transfers. At a new hospital it still finds 99.6% of healthy eyes, AUC 0.835.",
    "Staging does not transfer. Per-stage recall falls to about one correct call in four.",
    "The model clearly reads retinas, not metadata — 0.692 against a shortcut floor of 0.187.",
    "Most errors are near-misses: 603 of 1,433 are one stage off, only 27 are far off.",
  ]);
  s.addNotes("Lead with the 0.546 yourself. The next slide explains how much of it is the model and how much is the labels.");
}

// ---------------------------------------------------------------- 8 finding + limitations
{
  const s = S();
  head(s, "Main finding, and what it does not fix");

  sub(s, M, 1.15, 6.3, "ROP grows at the outer edge of the retina");
  bullets(s, M, 1.5, 6.3, 1.3, [
    "One session photographs the eye in several directions.",
    "The eye's single diagnosis is copied onto every frame.",
    "Some of those frames cannot physically show the disease.",
  ]);

  s.addChart(pres.ChartType.bar, [{
    name: "QWK",
    labels: ["Outer edge\n(temporal)", "Nose side\n(nasal)", "Back of eye\n(posterior)"],
    values: [0.841, 0.352, 0.229],
  }], {
    x: M, y: 2.85, w: 6.3, h: 3.0,
    barDir: "col", chartColors: ["1F3864"],
    showTitle: true, title: "Same predictions, split by where the camera pointed", titleFontSize: 12, titleColor: TXT, titleFontFace: F,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: TXT,
    showLegend: false,
    catAxisLabelColor: TXT, catAxisLabelFontSize: 10, valAxisLabelColor: GREY, valAxisLabelFontSize: 9,
    valAxisMaxVal: 1.0, valAxisMinVal: 0, valGridLine: { color: "E6E6E6", size: 0.75 }, catGridLine: { style: "none" },
    barGapWidthPct: 60,
  });
  s.addText("No aggregation — single images, scored one at a time, against the same labels.", {
    x: M, y: 5.88, w: 6.3, h: 0.28, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 10.5, italic: true, color: GREY,
  });
  s.addText("So ROP models should be scored per eye, not per photograph — and papers should say which unit their labels describe. The effect holds across 2 architectures and 2 seeds.", {
    x: M, y: 6.2, w: 6.3, h: 0.7, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12, bold: true, color: BLUE,
  });

  sub(s, M + 6.75, 1.15, CW - 6.75, "What we still cannot claim");
  const lim = [
    ["Calibration", "The percentages are not probabilities. Error 0.109 internally, 0.334 at a new hospital."],
    ["Worst error", "The model is most confident exactly when it calls a diseased eye Normal — mean 0.93."],
    ["Explanation", "Grad-CAM attention on the retina is at chance, so the heat map cannot verify its reasoning."],
    ["Severe classes", "Stage 4/5 rests on 89 images from one source. Declared not externally validated."],
    ["Generalisation", "One external hospital only. We cannot say the drop is general."],
    ["Labels", "No independent clinician has re-read them. This is the biggest open item."],
  ];
  table(s, lim, {
    x: M + 6.75, y: 1.5, w: CW - 6.75, colW: [1.35, 4.38], rowH: 0.6, fontSize: 11,
  });
  s.addText("Even scored per eye, it still misses about 18 diseased eyes in every 100.", {
    x: M + 6.75, y: 5.28, w: CW - 6.75, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 12, bold: true, color: RED,
  });
  s.addNotes("Spend time here. This is the actual contribution. Finish on the limitation so it does not sound oversold.");
}

// ---------------------------------------------------------------- 9 prototype
{
  const s = S();
  head(s, "Working prototype");

  sub(s, M, 1.15, CW, "What it does");
  const flow = [
    ["1. Upload", "A retinal photograph. Non-infant images are refused."],
    ["2. Preprocess", "Circle crop, CLAHE, resize. One forward pass."],
    ["3. Predict", "Stage, all six class scores, and an \"any disease\" score."],
    ["4. Explain", "Grad-CAM heat map and a downloadable PDF report."],
  ];
  flow.forEach((f, i) => {
    const x = M + i * 3.07;
    s.addShape(pres.ShapeType.rect, {
      x, y: 1.5, w: 2.85, h: 1.15, fill: { color: LGREY }, line: { color: LINE, width: 0.75 },
    });
    s.addText(f[0], {
      x: x + 0.15, y: 1.6, w: 2.55, h: 0.3, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 13, bold: true, color: BLUE, valign: "middle",
    });
    s.addText(f[1], {
      x: x + 0.15, y: 1.9, w: 2.55, h: 0.68, isTextBox: true, margin: 0,
      fontFace: F, fontSize: 11, color: TXT, valign: "top",
    });
  });

  sub(s, M, 2.95, 5.9, "Runs on an ordinary laptop");
  const perf = [
    hdr(["", "CPU", "GPU (MPS)"]),
    ["Preprocessing", "27 ms", "—"],
    ["Forward pass", "85–115 ms", "32 ms"],
    ["One image, end to end", "112 ms", "59 ms"],
    ["One 6-frame examination", "0.7 – 0.8 s", "0.4 s"],
    ["Model size", "83 MB", "83 MB"],
  ];
  table(s, perf, { y: 3.3, w: 5.9, colW: [2.6, 1.7, 1.6], rowH: 0.36, fontSize: 11 });

  sub(s, M + 6.3, 2.95, CW - 6.3, "Why the demo is trustworthy");
  bullets(s, M + 6.3, 3.3, CW - 6.3, 2.4, [
    "The served weights are identified by SHA-256, not assumed — we re-ran them over 663 images and matched the checkpoint exactly.",
    "Preprocessing is verified identical in training, evaluation and the app.",
    "Raw per-image predictions are committed, so every number in this deck can be recomputed.",
    "The app labels its screening score a ranking, not a probability of disease.",
    "It refuses adult retinas, after an audit found it flagged 41 of 45 as ROP-positive.",
  ], 12);

  s.addText("One command from a clean clone starts the app. 34 demo images are committed with it.", {
    x: M, y: 5.72, w: 5.9, h: 0.46, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11.5, italic: true, color: GREY,
  });
  s.addNotes("Run the demo live here if there is time. Otherwise show the PDF report it produces.");
}

// ---------------------------------------------------------------- 10 team + progress + next
{
  const s = S();
  head(s, "Team, progress and next steps");

  sub(s, M, 1.15, 6.6, "Who did what");
  const team = [
    hdr(["Member", "Area", "Owns"]),
    ["[Member 1]", "Data and ethics", "Dataset sourcing, licences, de-identification, inclusion cascade"],
    ["[Member 2]", "Modelling", "Backbone, the three heads, training runs, ablations"],
    ["[Member 3]", "Evaluation", "Shortcut audits, confidence intervals, sealed protocol, the unit study"],
    ["[Member 4]", "Application", "Web app, Grad-CAM, PDF report, demo, repository guards"],
  ];
  table(s, team, { y: 1.5, w: 6.6, colW: [1.4, 1.5, 3.7], rowH: [0.3, 0.52, 0.42, 0.52, 0.52], fontSize: 11 });

  sub(s, M, 4.0, 6.6, "Progress so far");
  bullets(s, M, 4.35, 6.6, 2.4, [
    "Jun–Jul: 11 datasets acquired, de-identified and audited.",
    "Aug: sealed external test opened once; earlier screening model retired after it flagged 663 of 663.",
    "Aug: glaucoma dropped from scope; literature survey completed.",
    "Sep: architecture shown equivalent to a plain baseline; the scoring-unit result replicated.",
    "Sep: full evidence package rebuilt from raw predictions — 275 files.",
  ], 12);

  sub(s, M + 7.0, 1.15, CW - 7.0, "Next steps");
  const nxt = [
    hdr(["What", "Owner", "By"]),
    ["Remove patient identifiers from the repository and make the guard test blocking", "M4", "23 Sep"],
    ["Temperature-scale the probabilities to fix calibration", "M3", "26 Sep"],
    ["Fix the config so the repo reproduces its own model", "M2", "26 Sep"],
    ["Re-derive the screening threshold on unseen data", "M3", "30 Sep"],
    ["Re-run ablations on all folds and both seeds", "M2", "12 Oct"],
    [{ text: "ISBI 2027 paper submission", options: { bold: true } }, { text: "All", options: { bold: true } }, { text: "26 Oct", options: { bold: true } }],
    ["Get a paediatric ophthalmologist to re-grade ~300 images", "M1", "in progress"],
    ["Find a second external hospital", "M1", "later"],
  ];
  table(s, nxt, { x: M + 7.0, y: 1.5, w: CW - 7.0, colW: [3.63, 0.75, 0.75], rowH: [0.3, 0.6, 0.42, 0.42, 0.42, 0.42, 0.36, 0.5, 0.36], fontSize: 10.5 });

  s.addText("The clinician re-read is the one item compute cannot solve, and it is why the outreach has already started.", {
    x: M + 7.0, y: 5.6, w: CW - 7.0, h: 0.5, isTextBox: true, margin: 0,
    fontFace: F, fontSize: 11, italic: true, color: GREY,
  });
  s.addNotes("Raise the repository privacy fix yourself — our own test caught it, and it is item one with a date.");
}

pres.writeFile({ fileName: "RetinAI_Review.pptx" }).then(f => console.log("WROTE " + f));
