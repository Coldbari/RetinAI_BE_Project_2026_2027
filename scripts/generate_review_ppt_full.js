const pptxgen = require("pptxgenjs");

// ---------- palette: fundus amber on clinical navy ----------
const NAVY = "10243A";
const NAVY2 = "1B3A5C";
const AMBER = "E8833A";
const TEAL = "1C7293";
const GREY = "5A6B7B";
const LIGHT = "F3F5F7";
const TINT = "FBF1E8"; // amber tint
const WHITE = "FFFFFF";
const GREEN = "2C7A4B";
const RED = "A8322D";

const HF = "Cambria";  // headers
const BF = "Calibri";  // body

const W = 13.333, H = 7.5;
const M = 0.55;              // page margin
const CW = W - 2 * M;        // content width

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "RetinAI team";
pres.title = "RetinAI - ROP Staging - BE Post Proposal Review";

// ---------- helpers ----------
function chip(s, code) {
  // the deck's motif: a rubric-criterion chip, top-right of every content slide
  if (!code) return;
  s.addShape(pres.ShapeType.roundRect, {
    x: W - M - 1.5, y: 0.34, w: 1.5, h: 0.42, rectRadius: 0.2,
    fill: { color: AMBER }, line: { color: AMBER },
  });
  s.addText(code, {
    x: W - M - 1.5, y: 0.34, w: 1.5, h: 0.42, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle",
  });
}

function title(s, t, sub, code, dark) {
  s.addText(t, {
    x: M, y: 0.34, w: CW - 1.8, h: 0.62, isTextBox: true, margin: 0,
    fontFace: HF, fontSize: 27, bold: true, color: dark ? WHITE : NAVY, valign: "middle",
  });
  if (sub) {
    s.addText(sub, {
      x: M, y: 0.98, w: CW - 1.8, h: 0.40, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12.5, italic: true, color: dark ? AMBER : GREY, valign: "middle",
    });
  }
  chip(s, code);
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || LIGHT }, line: { color: fill || LIGHT },
  });
}

function tbl(s, rows, opts) {
  s.addTable(rows, Object.assign({
    x: M, w: CW, border: { type: "solid", color: "DDE3E8", pt: 0.75 },
    fontFace: BF, fontSize: 11.5, color: NAVY, valign: "middle",
    autoPage: false,
  }, opts));
}

function hdr(txts) {
  return txts.map(t => ({
    text: t,
    options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11.5, fontFace: BF },
  }));
}

function newSlide(bg) {
  const s = pres.addSlide();
  s.background = { color: bg || WHITE };
  return s;
}

// =====================================================================
// 1 — TITLE
// =====================================================================
{
  const s = newSlide(NAVY);
  s.addShape(pres.ShapeType.roundRect, {
    x: -1.2, y: -1.6, w: 5.6, h: 5.6, rectRadius: 0.5,
    fill: { color: NAVY2 }, line: { color: NAVY2 },
  });
  s.addText("RetinAI", {
    x: M, y: 1.35, w: 9.2, h: 0.95, isTextBox: true, margin: 0,
    fontFace: HF, fontSize: 54, bold: true, color: WHITE,
  });
  s.addText("Audited deep-learning ICROP staging of Retinopathy of Prematurity", {
    x: M, y: 2.3, w: 10.4, h: 0.5, isTextBox: true, margin: 0,
    fontFace: HF, fontSize: 21, color: AMBER,
  });
  s.addText(
    "A six-class staging model built from four hospitals, tested once at a fifth-party site it had never seen — and the measurement finding that came out of it.",
    { x: M, y: 2.85, w: 10.6, h: 0.6, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 13.5, color: "C2CEDA", italic: true });

  const stats = [
    ["13,037", "photographs acquired"],
    ["11 → 4", "datasets screened → used"],
    ["3 + 1", "training + held-out hospitals"],
    ["661", "sealed test images, opened once"],
  ];
  stats.forEach((st, i) => {
    const x = M + i * 3.07;
    s.addText(st[0], {
      x, y: 3.72, w: 2.9, h: 0.55, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 30, bold: true, color: AMBER,
    });
    s.addText(st[1], {
      x, y: 4.26, w: 2.9, h: 0.46, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11, color: "C2CEDA",
    });
  });

  s.addText([
    { text: "SEM VII · B.E. Project Post-Proposal Review", options: { bold: true, color: WHITE } },
    { text: "\nProject title: ", options: { color: "9FB2C4" } },
    { text: "RetinAI — Audited ICROP Staging of Retinopathy of Prematurity", options: { color: WHITE } },
    { text: "        Team number: ______", options: { color: "9FB2C4" } },
    { text: "\nTeam: [Member 1] · [Member 2] · [Member 3] · [Member 4]        Guide: [Guide name]", options: { color: "9FB2C4" } },
    { text: "\nVivekanand Education Society's Institute of Technology · Repository commit f0d4508 · 21 Sep 2026", options: { color: "7D93A8", fontSize: 10.5 } },
  ], { x: M, y: 5.15, w: 11.0, h: 1.25, isTextBox: true, margin: 0, fontFace: BF, fontSize: 12, lineSpacing: 17 });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 6.6, w: 12.2, h: 0.46, rectRadius: 0.06,
    fill: { color: "3A1A18" }, line: { color: RED },
  });
  s.addText("Not a medical device. A final-year research prototype — no regulatory clearance, no prospective validation, no clinical use. Every finding belongs to a qualified clinician.", {
    x: M + 0.15, y: 6.6, w: 11.9, h: 0.46, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 10.5, color: "F0C4C0", valign: "middle",
  });
  s.addNotes("Open by naming the rubric: this deck is ordered so that each of your six criteria has its own slides, marked C1 to C6 in the top-right corner. Slide 2 is the map.");
}

// =====================================================================
// 2 — RUBRIC MAP
// =====================================================================
{
  const s = newSlide();
  title(s, "How to read this deck", "Every slide is tagged with the review criterion it answers. Six criteria, six sections, in your order.", null);

  const rows = [
    ["C1", "Problem understanding and need", "3 – 4", "A preventable childhood blindness whose bottleneck is readers, not cameras — with five named constraints we could not design away."],
    ["C2", "Existing solutions and the gap", "5 – 6", "Five comparator systems, their reported numbers, and the measurement defect each one shares. The gap is precise and we measured it on our own data."],
    ["C3", "Objectives, scope, success criteria", "7 – 8", "Six measurable objectives, explicit in/out of scope, and a pass-fail criteria table written in advance — including the two we failed."],
    ["C4", "Proposed approach and feasibility", "9 – 11", "End-to-end architecture, every key choice with the evidence that rejected its alternative, resource budget and a six-entry risk register."],
    ["C5", "Basic working / proof of concept", "12 – 15", "The system runs end to end on a laptop CPU; internal and external test evidence; and the behaviour explained rather than excused."],
    ["C6", "Team understanding and next plan", "16 – 18", "Four owned workstreams with what each member can defend, a dated progress trail, and a ten-item plan to ISBI 2027."],
  ];

  let y = 1.5;
  rows.forEach((r) => {
    card(s, M, y, CW, 0.83, LIGHT);
    s.addShape(pres.ShapeType.roundRect, {
      x: M + 0.14, y: y + 0.19, w: 0.58, h: 0.45, rectRadius: 0.1,
      fill: { color: AMBER }, line: { color: AMBER },
    });
    s.addText(r[0], {
      x: M + 0.14, y: y + 0.19, w: 0.58, h: 0.45, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 13, bold: true, color: WHITE, align: "center", valign: "middle",
    });
    s.addText(r[1], {
      x: M + 0.86, y: y + 0.09, w: 3.62, h: 0.33, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: NAVY, valign: "middle",
    });
    s.addText("slides " + r[2], {
      x: M + 0.86, y: y + 0.42, w: 3.62, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10.5, color: TEAL, valign: "middle",
    });
    s.addText(r[3], {
      x: M + 4.62, y: y + 0.08, w: CW - 4.8, h: 0.67, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, color: GREY, valign: "middle",
    });
    y += 0.9;
  });
  s.addNotes("Say: we built this deck against your rubric. If you want the evidence for any criterion, the chip in the corner tells you which slide it is on.");
}

// =====================================================================
// 3 — C1 PROBLEM
// =====================================================================
{
  const s = newSlide();
  title(s, "The problem: a blindness we know how to prevent", "Retinopathy of prematurity — abnormal retinal blood-vessel growth in babies born too early.", "C1");

  const big = [
    ["Leading", "preventable cause of childhood blindness worldwide", AMBER],
    ["Weeks", "the window in which treatment still works. Miss it and the loss is irreversible.", TEAL],
    ["κ = 0.24", "inter-expert agreement between ophthalmologists on this task (Gschliesser 2015)", RED],
  ];
  big.forEach((b, i) => {
    const x = M + i * 4.16;
    card(s, x, 1.52, 3.95, 1.5, i === 2 ? TINT : LIGHT);
    s.addText(b[0], {
      x: x + 0.2, y: 1.64, w: 3.55, h: 0.55, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 28, bold: true, color: b[2], valign: "middle",
    });
    s.addText(b[1], {
      x: x + 0.2, y: 2.2, w: 3.55, h: 0.72, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, color: NAVY, valign: "top",
    });
  });

  s.addText("Why it is unsolved in practice", {
    x: M, y: 3.22, w: 6.1, h: 0.34, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 15, bold: true, color: NAVY,
  });
  s.addText([
    { text: "The bottleneck is not imaging capacity. ", options: { bold: true } },
    { text: "Neonatal units can take the photographs. What they run out of is clinicians able to grade them inside the treatment window.", options: {} },
    { text: "\n\nGrading is genuinely hard, not merely tedious. ", options: { bold: true, breakLine: false } },
    { text: "In one public corpus, five ophthalmologists graded 1,533 images independently and ", options: {} },
    { text: "not one image received a unanimous grade.", options: { bold: true, color: RED } },
    { text: "\n\nThe stages are subtle and ordered. ", options: { bold: true } },
    { text: "Stage 1 is a faint flat white line; Stage 2 is that line raised into a ridge; Stage 3 adds abnormal vessels on top of it. AP-ROP is an aggressive form that is not a rung on that ladder at all.", options: {} },
  ], { x: M, y: 3.58, w: 6.1, h: 2.6, isTextBox: true, margin: 0, fontFace: BF, fontSize: 12, color: NAVY, lineSpacing: 16.5 });

  card(s, M + 6.5, 3.22, CW - 6.5, 3.0, LIGHT);
  s.addText("What an examination actually looks like", {
    x: M + 6.72, y: 3.34, w: 5.4, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  s.addText([
    { text: "One screening session photographs one eye several times, pointing the camera in different directions — outer edge, centre, nose side. In this corpus a session averages about six frames.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "The clinician records ", options: {} },
    { text: "one diagnosis for the eye", options: { bold: true } },
    { text: ". That single diagnosis is then inherited by every frame in the session — including the frames that physically cannot show the disease.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "Hold on to that sentence. It is where this project's main research result comes from (slide 14).", options: { italic: true, color: AMBER } },
  ], { x: M + 6.72, y: 3.74, w: CW - 6.94, h: 2.4, isTextBox: true, margin: 0, fontFace: BF, fontSize: 12, color: NAVY, lineSpacing: 16.5 });

  s.addNotes("Key line: the bottleneck is readers, not cameras. And the task is hard enough that humans agree at kappa 0.24 — which is why any paper claiming 0.98 deserves a second look.");
}

// =====================================================================
// 4 — C1 USERS, IMPACT, CONSTRAINTS
// =====================================================================
{
  const s = newSlide();
  title(s, "Who it is for, and what we were not free to change", "Users and context on the left; the five constraints that shaped every design decision on the right.", "C1");

  s.addText("Users and context", {
    x: M, y: 1.5, w: 5.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 15, bold: true, color: NAVY,
  });
  const users = [
    ["Primary — NICU screening team", "Triage support at the point the photographs are taken: which eyes need a specialist's attention first. Decision support, never a diagnosis."],
    ["Primary — paediatric ophthalmologist", "A second reader that never tires, and a written, reproducible record of what was seen."],
    ["Secondary — ROP methods researchers", "An audited public benchmark and a measurement result they can apply to their own numbers."],
  ];
  let uy = 1.9;
  users.forEach((u) => {
    card(s, M, uy, 5.6, 1.16, LIGHT);
    s.addText(u[0], {
      x: M + 0.18, y: uy + 0.1, w: 5.25, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12.5, bold: true, color: TEAL, valign: "middle",
    });
    s.addText(u[1], {
      x: M + 0.18, y: uy + 0.4, w: 5.25, h: 0.66, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11, color: NAVY, valign: "top",
    });
    uy += 1.26;
  });

  card(s, M, 5.68, 5.6, 1.12, TINT);
  s.addText("Impact, if it worked", {
    x: M + 0.18, y: 5.76, w: 5.25, h: 0.28, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 12.5, bold: true, color: AMBER, valign: "middle",
  });
  s.addText("A six-frame examination scored in about 0.8 seconds on an ordinary laptop CPU — no GPU, no cloud, no per-image cost. Compute was never the constraint; everything below was.", {
    x: M + 0.18, y: 6.04, w: 5.25, h: 0.68, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 11, color: NAVY, valign: "top",
  });

  s.addText("The five binding constraints", {
    x: M + 5.95, y: 1.5, w: 6.5, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 15, bold: true, color: NAVY,
  });
  const rows = [
    hdr(["#", "Constraint", "What it forced"]),
    ["1", "No primary data collection", "Ethics approval, a hospital partnership and parental consent are not available to a student project. Public, licensed, de-identified research data only."],
    ["2", "No independent grader", "Labels are used as the source institutions released them. 43% of one corpus's \"ROP-positive\" class is not ROP — found, documented, excluded."],
    ["3", "The severe classes barely exist", "Only 89 Stage 4/5 images exist in public data anywhere, all from one dataset. Training imbalance 12.5 : 1."],
    ["4", "Free compute only", "Kaggle P100/T4, ~3.5 min/epoch. Deployment had to be CPU-only from the start."],
    ["5", "Licence and safety", "The held-out hospital's data is CC BY-NC — no commercial use, no re-mirroring. Not a medical device, and nothing here changes that."],
  ];
  tbl(s, rows, {
    x: M + 5.95, y: 1.9, w: CW - 5.95, colW: [0.34, 1.95, 4.54],
    rowH: [0.3, 0.72, 0.85, 0.62, 0.62, 0.72], fontSize: 10.5,
  });
  s.addNotes("If asked 'why not collect your own data' — constraint 1. It is also why constraint 2 is the biggest open item: a clinician re-read is the long pole and it needs a person, not compute.");
}

// =====================================================================
// 5 — C2 EXISTING SOLUTIONS
// =====================================================================
{
  const s = newSlide();
  title(s, "What already exists — and what each result rests on", "Twelve papers surveyed; five set the terms. The right-hand column is what we checked ourselves.", "C2");

  const rows = [
    hdr(["System", "What it reports", "What the number actually rests on"]),
    [
      { text: "Vahidmoghadam et al.\nSci. Rep., 2026\nEfficientNet-B4", options: { bold: true } },
      { text: "0.98 stage accuracy\n1.000 plus-disease test accuracy", options: { color: GREEN, bold: true } },
      "Same Ostrava corpus we use, split 80/16/4 at the image level — on data averaging ~32 frames per infant. No external validation; three camera systems, no device-stratified analysis. Its own limitations section names the risk. We treat it as the control condition, not a competitor.",
    ],
    [
      { text: "Wang et al.\nJAMA Netw. Open, 2021\nExplainable ROP platform", options: { bold: true } },
      { text: "F1 0.956 image\n0.915 eye\n0.898 patient", options: { color: GREEN, bold: true } },
      "The only comparator that reports all three units — so the question is on the table. But the effect is small and points downward. We find the opposite sign and roughly three times the magnitude, and we show the anatomical reason why.",
    ],
    [
      { text: "Chen et al.\nOphthalmol. Retina, 2021\nStage diagnosis", options: { bold: true } },
      { text: "Accuracy and generalisability across populations and cameras", options: { color: NAVY } },
      "The closest thing to an honest generalisation study in this literature. Its corpus is not public, so nobody can reproduce the audit or reuse the splits.",
    ],
    [
      { text: "Ganin et al.\nJMLR, 2016\nDANN — the standard site-invariance fix", options: { bold: true } },
      { text: "Domain-adversarial training", options: { color: NAVY } },
      "The remedy everyone reaches for, including us. Its standard evidence is a linear site probe whose accuracy falls — but that probe is confounded by disease prevalence, which also varies by site. We tested it disease-controlled. It never improved.",
    ],
    [
      { text: "Gschliesser et al.\nAm. J. Ophthalmol., 2015\nThe human ceiling", options: { bold: true } },
      { text: "Inter-expert Fleiss κ = 0.24", options: { color: RED, bold: true } },
      "This is the number that reframes the whole table. Experts agree with each other at 0.24. A model reporting agreement above 0.95 against those same labels warrants scrutiny, not celebration.",
    ],
  ];
  tbl(s, rows, {
    y: 1.62, colW: [2.5, 2.35, 7.38],
    rowH: [0.3, 1.16, 1.02, 0.76, 1.0, 0.86], fontSize: 10.5,
  });
  s.addNotes("Do not rubbish these papers. The point is that four of the five share one blind spot, and the fifth tells you why that blind spot matters.");
}

// =====================================================================
// 6 — C2 THE GAP
// =====================================================================
{
  const s = newSlide(NAVY);
  title(s, "The gap, stated precisely", "The gap is not that ROP staging is undone. It is that the numbers are decided before an architecture is chosen.", "C2", true);

  card(s, M, 1.55, CW, 0.82, NAVY2);
  s.addText([
    { text: "Our claim:  ", options: { bold: true, color: AMBER } },
    { text: "three measurement choices — ", options: { color: WHITE } },
    { text: "the unit the labels describe, the probe used to claim site-invariance, and the split granularity", options: { color: WHITE, bold: true } },
    { text: " — each move a reported number further than the architectural contribution in the same paper. None of the three is routinely ablated.", options: { color: WHITE } },
  ], { x: M + 0.2, y: 1.55, w: CW - 0.4, h: 0.82, isTextBox: true, margin: 0, fontFace: BF, fontSize: 13, valign: "middle", lineSpacing: 17 });

  s.addText("We did not assume the gap. We measured it — on our own data, before training anything.", {
    x: M, y: 2.55, w: CW, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: AMBER,
  });

  const ev = [
    ["AUC 0.911", "from image dimensions alone", "A classifier trained on nothing but image width and height — no pixels at all — scores 0.911 on the binary task. Each hospital uses a different camera and has a different disease mix, so \"which camera?\" nearly answers \"which diagnosis?\""],
    ["0.997", "site decodability from metadata", "The hospital can be identified from trivial file metadata at 0.997 balanced accuracy against a 0.333 chance rate. The shortcut is not hypothetical; it is sitting in the data."],
    ["43%", "of \"ROP-positive\" is not ROP", "In the largest corpus, 1,295 images labelled positive record haemorrhage, toxoplasmosis, hamartoma or optic-nerve hypoplasia. A model trained on those labels is an abnormal-retina detector reported as an ROP result."],
    ["0.47 → 0.81", "same predictions, different unit", "Change nothing but whether you score per photograph or per eye, and quadratic-weighted kappa moves from 0.473 to 0.813. That is larger than any architecture effect we could find."],
    ["663 / 663", "what happens without a held-out site", "Our own earlier screening model, trained on essentially one hospital, flagged every one of 663 images from a new hospital as diseased — including every healthy eye. It was retired. It is why this project is built the way it is."],
  ];
  let y = 2.98;
  ev.forEach((e) => {
    card(s, M, y, CW, 0.76, NAVY2);
    s.addText(e[0], {
      x: M + 0.18, y: y + 0.06, w: 1.72, h: 0.38, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 19, bold: true, color: AMBER, valign: "middle",
    });
    s.addText(e[1], {
      x: M + 0.18, y: y + 0.43, w: 1.9, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 9, color: "9FB2C4", valign: "middle",
    });
    s.addText(e[2], {
      x: M + 2.2, y: y + 0.05, w: CW - 2.4, h: 0.67, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11, color: WHITE, valign: "middle",
    });
    y += 0.84;
  });

  s.addNotes("The gap in one sentence: there is no public ICROP staging benchmark that is patient-grouped, shortcut-audited, externally validated and scored exactly once. That is the hole we are filling.");
}

// =====================================================================
// 7 — C3 OBJECTIVES
// =====================================================================
{
  const s = newSlide();
  title(s, "Objectives — each one with the number that closes it", "Written before the work, not after. Five are met; the sixth is in progress with a dated deadline.", "C3");

  const rows = [
    hdr(["#", "Objective", "Measurable target, set in advance", "Outcome"]),
    ["O1", { text: "Build an audited 6-class ICROP corpus from public sources", options: { bold: true } },
      "Every exclusion counted and reconciled; the inclusion flow must balance to the image. Zero patient or image leakage across any split.",
      { text: "MET\n8,024 eligible · 0 leakage", options: { color: GREEN, bold: true } }],
    ["O2", { text: "Train a model that demonstrably learns from pixels, not shortcuts", options: { bold: true } },
      "Cross-validated macro-F1 must exceed the metadata-only floor of 0.187 established in the audit.",
      { text: "MET\n0.692 ± 0.086", options: { color: GREEN, bold: true } }],
    ["O3", { text: "Measure external transfer honestly", options: { bold: true } },
      "A whole hospital held out from training; its sealed half opened exactly once, under a protocol committed to version control beforehand.",
      { text: "MET\nopened once, 18 Aug 2026", options: { color: GREEN, bold: true } }],
    ["O4", { text: "Test whether the evaluation unit, not the architecture, drives reported agreement", options: { bold: true } },
      "The effect must replicate across at least two architectures and two random seeds, in the same direction.",
      { text: "MET\n4 of 4 cells agree", options: { color: GREEN, bold: true } }],
    ["O5", { text: "Ship a CPU-only prototype that reports its own uncertainty", options: { bold: true } },
      "Stage + probabilities + heat map + PDF report, under one second per six-frame examination on a laptop CPU, with an out-of-domain guard.",
      { text: "MET\n~0.7–0.8 s / exam", options: { color: GREEN, bold: true } }],
    ["O6", { text: "Publish the measurement result", options: { bold: true } },
      "ISBI 2027 submission, four pages, with the locked-set evaluation as its evidence base.",
      { text: "IN PROGRESS\ndue 26 Oct 2026", options: { color: AMBER, bold: true } }],
  ];
  tbl(s, rows, {
    y: 1.58, colW: [0.42, 3.05, 6.66, 2.1],
    rowH: [0.3, 0.78, 0.7, 0.78, 0.82, 0.86, 0.7], fontSize: 10.5,
  });
  s.addNotes("Every target here is a number or a yes/no, and every one was fixed before the result was known. O6 is the only open one and it has a date.");
}

// =====================================================================
// 8 — C3 SCOPE + SUCCESS CRITERIA
// =====================================================================
{
  const s = newSlide();
  title(s, "Scope, boundaries and success criteria", "Including the two criteria we failed — reported here rather than quietly dropped.", "C3");

  // in / out of scope
  card(s, M, 1.5, 6.05, 2.42, LIGHT);
  s.addText("In scope", {
    x: M + 0.18, y: 1.58, w: 5.7, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 13.5, bold: true, color: GREEN, valign: "middle",
  });
  s.addText([
    { text: "Retinopathy of prematurity only", options: { bullet: true, breakLine: true } },
    { text: "6-class ICROP-3 staging (Normal · S1 · S2 · S3 · S4/5 · AP-ROP) plus a binary \"any disease\" screening score", options: { bullet: true, breakLine: true } },
    { text: "Retrospective, public, licensed, de-identified data", options: { bullet: true, breakLine: true } },
    { text: "Single-photograph input; eye-level aggregation analysed", options: { bullet: true, breakLine: true } },
    { text: "A research prototype, an explainability audit, and a reproducible evidence package", options: { bullet: true } },
  ], { x: M + 0.22, y: 1.9, w: 5.66, h: 1.95, isTextBox: true, margin: 0, fontFace: BF, fontSize: 11, color: NAVY, paraSpaceAfter: 4 });

  card(s, M + 6.18, 1.5, CW - 6.18, 2.42, TINT);
  s.addText("Out of scope — and why", {
    x: M + 6.36, y: 1.58, w: 5.7, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 13.5, bold: true, color: RED, valign: "middle",
  });
  s.addText([
    { text: "Glaucoma — withdrawn. ", options: { bold: true } },
    { text: "The G1020 arm double-mounts per eye and contaminated the AUC. We dropped the disease rather than publish a broken 0.973.", options: { breakLine: true } },
    { text: "Diabetic retinopathy — parked. ", options: { bold: true } },
    { text: "The referable-DR arm is complete (AUC 0.906, external 0.837) but is not this year's research front.", options: { breakLine: true } },
    { text: "Zone and plus disease — unavailable. ", options: { bold: true } },
    { text: "No corpus we can obtain carries these annotations, and they are the criteria treatment decisions actually use. Named as a limitation, not hidden.", options: { breakLine: true } },
    { text: "Clinical use, regulatory clearance, commercial deployment.", options: { bold: true } },
  ], { x: M + 6.4, y: 1.9, w: CW - 6.6, h: 1.95, isTextBox: true, margin: 0, fontFace: BF, fontSize: 10.5, color: NAVY, lineSpacing: 13.5 });

  // success criteria table
  s.addText("Success criteria — thresholds fixed in advance, results as they fell", {
    x: M, y: 4.02, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const ok = { color: GREEN, bold: true }, no = { color: RED, bold: true };
  const rows = [
    hdr(["Success criterion", "Target", "Achieved", "", "Success criterion", "Target", "Achieved", ""]),
    ["No patient or image leakage", "0", "0, every comparison", { text: "PASS", options: ok },
     "External screening AUC", "≥ 0.80", "0.835", { text: "PASS", options: ok }],
    ["Beat the metadata-only floor", "> 0.187", "0.692 macro-F1", { text: "PASS", options: ok },
     "External per-stage recall", "≥ 0.60", "≈ 0.28", { text: "FAIL", options: no }],
    ["Internal ordered agreement", "QWK ≥ 0.75", "0.897 [0.869–0.923]", { text: "PASS", options: ok },
     "Calibrated probabilities", "ECE ≤ 0.05", "0.109 int / 0.334 ext", { text: "FAIL", options: no }],
    ["Internal any-disease AUC", "≥ 0.90", "0.970", { text: "PASS", options: ok },
     "Latency per 6-frame exam", "< 1 s CPU", "0.7 – 0.8 s", { text: "PASS", options: ok }],
    ["Healthy-eye recall externally", "≥ 0.90", "0.996", { text: "PASS", options: ok },
     "Sealed test opened", "exactly once", "once, 18 Aug 2026", { text: "PASS", options: ok }],
  ];
  tbl(s, rows, {
    y: 4.38, colW: [2.35, 1.0, 1.95, 0.82, 2.35, 1.0, 1.95, 0.82],
    rowH: 0.36, fontSize: 10,
  });
  s.addText("The two failures are the honest content of slides 14 and 18: they are the open problems, and both have a dated fix.", {
    x: M, y: 6.72, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 11.5, italic: true, color: GREY,
  });
  s.addNotes("Draw attention to the two FAILs yourself, before the panel finds them. That is the difference between a project that knows its own limits and one that does not.");
}

// =====================================================================
// 9 — C4 ARCHITECTURE
// =====================================================================
{
  const s = newSlide();
  title(s, "The system, end to end", "Five stages. Nothing reaches the model until the audit gates have passed.", "C4");

  const steps = [
    ["1", "ACQUIRE", "11 public sources\n13,037 images\nDOI, licence and\ncaveats logged for\nevery one"],
    ["2", "AUDIT + INCLUDE", "10,427 stage-labelled\n→ 8,024 eligible\n→ 3,112 trained on\nEvery exclusion counted\nand reconciled"],
    ["3", "GATE", "G1 patient leakage\nG2 duplicate leakage\nG3 site decodability\nG4 metadata-only floor\nG5 disease-controlled probe"],
    ["4", "PREPROCESS", "circle-crop →\nletterbox square →\nCLAHE →\n384 × 384 →\nImageNet normalise"],
    ["5", "PREDICT + SERVE", "EfficientNetV2-S\n3 heads → pred6, prob6\nFlask app: stage,\nGrad-CAM, PDF report\n1 forward pass, no TTA"],
  ];
  const bw = 2.38, gap = 0.12;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, 1.52, bw, 2.2, i === 2 ? TINT : LIGHT);
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.14, y: 1.64, w: 0.36, h: 0.36,
      fill: { color: i === 2 ? AMBER : NAVY }, line: { color: i === 2 ? AMBER : NAVY },
    });
    s.addText(st[0], {
      x: x + 0.14, y: 1.64, w: 0.36, h: 0.36, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle",
    });
    s.addText(st[1], {
      x: x + 0.56, y: 1.64, w: bw - 0.7, h: 0.36, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, bold: true, color: i === 2 ? AMBER : TEAL, valign: "middle",
    });
    s.addText(st[2], {
      x: x + 0.16, y: 2.08, w: bw - 0.32, h: 1.55, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10, color: NAVY, valign: "top", lineSpacing: 13,
    });
    if (i < 4) {
      s.addShape(pres.ShapeType.rightArrow, {
        x: x + bw + 0.002, y: 2.55, w: 0.115, h: 0.14,
        fill: { color: "A9B6C2" }, line: { color: "A9B6C2" },
      });
    }
  });

  // model detail
  s.addText("Inside stage 5 — one backbone, three heads", {
    x: M, y: 3.9, w: 6.2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  card(s, M, 4.25, 2.6, 2.15, NAVY);
  s.addText("EfficientNetV2-S", {
    x: M + 0.14, y: 4.42, w: 2.32, h: 0.34, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 12.5, bold: true, color: WHITE, align: "center",
  });
  s.addText("ImageNet-pretrained\nclassifier stripped\n\n1280-d embedding\nDropout 0.3\n\n20,177,488 params\n(98.4% of the model)", {
    x: M + 0.14, y: 4.78, w: 2.32, h: 1.5, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 9.5, color: "C2CEDA", align: "center", lineSpacing: 12,
  });

  const heads = [
    ["CORN ordinal stage head", "5,124 params", "Predicts the ordered ladder Normal < S1 < S2 < S3 < S4/5. AP-ROP images are masked out of this loss — ICROP-3 gives that form no stage.", TEAL],
    ["AP-ROP binary branch", "1,281 params", "An independent sigmoid, because aggressive posterior ROP is a distinct form, not a rung. BCE with auto square-root positive weighting.", TEAL],
    ["Site adversary (DANN)", "328,707 params", "Behind a gradient-reversal layer, λ ramped 0 → 1. Training-time machinery only; inert at inference. Measured honestly on slide 15: it never improved.", GREY],
  ];
  heads.forEach((hd, i) => {
    const x = M + 2.78 + i * 3.25;
    card(s, x, 4.25, 3.1, 2.15, LIGHT);
    s.addText(hd[0], {
      x: x + 0.16, y: 4.36, w: 2.78, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, bold: true, color: hd[3], valign: "middle",
    });
    s.addText(hd[1], {
      x: x + 0.16, y: 4.64, w: 2.78, h: 0.24, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 9.5, color: AMBER, valign: "middle",
    });
    s.addText(hd[2], {
      x: x + 0.16, y: 4.92, w: 2.78, h: 1.36, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10, color: NAVY, valign: "top", lineSpacing: 13,
    });
  });

  s.addText([
    { text: "Interface contract:  ", options: { bold: true, color: AMBER } },
    { text: "in — 384 × 384 × 3 RGB, ImageNet-normalised.   out — pred6 ∈ {0..5}, prob6 (a 6-vector summing to 1), any-ROP score = prob6[1:].sum().   Preprocessing is verified byte-identical across training, evaluation and the served web app.", options: { color: NAVY } },
  ], { x: M, y: 6.56, w: CW, h: 0.4, isTextBox: true, margin: 0, fontFace: BF, fontSize: 10.5, valign: "middle" });

  s.addNotes("The gate stage is highlighted because it is the part most projects skip. Nothing trains until G1-G5 pass.");
}

// =====================================================================
// 10 — C4 DESIGN CHOICES JUSTIFIED
// =====================================================================
{
  const s = newSlide();
  title(s, "Every key choice, and what killed its alternative", "We did not pick these from a tutorial. Each row is a decision we can defend with a number from our own runs.", "C4");

  const rows = [
    hdr(["Decision", "Alternative rejected", "The evidence that decided it"]),
    [{ text: "Split by baby, never by photograph", options: { bold: true } }, "Image-level random split",
      "One baby contributes up to 470 frames. An image-level split lets the model recognise the infant instead of the disease — which is exactly how a 2026 paper on our own Ostrava data reached 98–100% accuracy."],
    [{ text: "Train at natural prevalence", options: { bold: true } }, "Class-balanced weighted sampler",
      "We tried it. CORN plus a balanced sampler causes prior-shift collapse: QWK fell to 0.121. That arm is retired and documented. The sampling effect was five times the apparent architecture effect."],
    [{ text: "Hold an entire hospital out of training", options: { bold: true } }, "k-fold on the pooled data alone",
      "Cross-validation shares cameras, graders and colour habits, so it cannot test transfer. Internally 0.786; at the held-out hospital 0.546. Without the holdout we would have shipped 0.786 believing it."],
    [{ text: "Halve the held-out hospital: 663 dev + 661 sealed", options: { bold: true } }, "Use all 1,324 images as the test set",
      "Every time you look at a test set and then change something, it becomes training data. One half had to be lookable-at so the other half stayed genuinely clean. The sealed script refuses to run twice."],
    [{ text: "CORN ordinal head + separate AP-ROP branch", options: { bold: true } }, "A flat 6-way softmax",
      "Taxonomy fidelity — AP-ROP is not a point on the 1-2-3 ladder. Explicitly NOT accuracy: we measured the two as equivalent (p = 0.589) and the flat head is better calibrated. We report that against our own design."],
    [{ text: "Last-epoch weights", options: { bold: true } }, "Best-validation checkpoint",
      "No selection on validation data, by protocol. It costs some accuracy — best-minus-final val macro-F1 spans 0.000–0.120 — and we take the cost rather than the optimistic number."],
    [{ text: "EfficientNetV2-S at 384²", options: { bold: true } }, "Larger or newer backbones",
      "A five-backbone sweep spread 0.028 in macro-F1 — the same size as the noise across the last five epochs of a single run. Architecture is not the binding constraint here, so we stopped spending compute on it."],
  ];
  tbl(s, rows, {
    y: 1.6, colW: [2.65, 2.35, 7.23],
    rowH: [0.28, 0.72, 0.72, 0.72, 0.72, 0.78, 0.68, 0.7], fontSize: 10.5,
  });
  s.addNotes("If the panel asks one design question, it will be about the ordinal head. The answer: we chose it for taxonomy fidelity, we measured it as no better than a softmax, and we published that.");
}

// =====================================================================
// 11 — C4 FEASIBILITY + RISK
// =====================================================================
{
  const s = newSlide();
  title(s, "Feasibility: cost, resources and risk", "The project is already built on the resources available to us. The risk register is what keeps it standing.", "C4");

  s.addText("Resource budget — actual, not estimated", {
    x: M, y: 1.5, w: 5.3, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const res = [
    ["Training compute", "Kaggle free P100 / T4 · ~3.5 min per epoch · 20 epochs × 6 runs ≈ 7 GPU-hours per full 5-fold experiment", "₹0"],
    ["Data", "13,037 images from 11 public sources, all free and licensed. No collection, no annotation cost.", "₹0"],
    ["Model artefact", "82.96 MB checkpoint · 20.5 M params, of which 20.18 M run at inference", "83 MB"],
    ["Serving", "CPU-only. 26.7 ms preprocessing + 85–115 ms forward pass per image; one forward pass, no test-time augmentation.", "~112 ms"],
    ["Per examination", "Six frames, single-threaded, end to end on a laptop CPU", "0.7–0.8 s"],
    ["Still to spend", "~10 GPU-h for the remaining ablations, ~70 min for one instrumented re-run. Fixes 1–5 on slide 18 are CPU-only.", "≈ 2 days"],
  ];
  let ry = 1.86;
  res.forEach((r, i) => {
    card(s, M, ry, 5.5, 0.76, i % 2 ? WHITE : LIGHT);
    s.addText(r[0], {
      x: M + 0.16, y: ry + 0.05, w: 3.9, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11, bold: true, color: TEAL, valign: "middle",
    });
    s.addText(r[1], {
      x: M + 0.16, y: ry + 0.29, w: 4.15, h: 0.44, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 9.5, color: NAVY, valign: "top",
    });
    s.addText(r[2], {
      x: M + 4.34, y: ry + 0.05, w: 1.0, h: 0.66, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 13, bold: true, color: AMBER, align: "right", valign: "middle",
    });
    ry += 0.8;
  });
  s.addText("The honest scarcity is not compute. It is a paediatric ophthalmologist's time — see risk 6.", {
    x: M, y: 6.68, w: 5.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 10.5, italic: true, color: GREY,
  });

  s.addText("Risk register", {
    x: M + 5.85, y: 1.5, w: 6.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const rows = [
    hdr(["Risk", "Status", "Mitigation — already in place"]),
    ["Shortcut learning: the model reads the camera, not the retina",
      { text: "MEASURED\n0.997 decodable", options: { color: RED, bold: true } },
      "Five pre-training gates · a 0.187 metadata-only floor every model must clear · DANN adversary · colour, sharpness and zoom randomisation · a whole hospital held out"],
    ["Patient leakage inflating every score",
      { text: "CONTROLLED\n0 found", options: { color: GREEN, bold: true } },
      "Patient-grouped folds · MD5 plus 256-bit perceptual deduplication · re-verified from scratch for this review"],
    ["Confident false \"Normal\" on a diseased eye",
      { text: "PRESENT\nmean conf. 0.934", options: { color: RED, bold: true } },
      "No confidence threshold can fix this — so the model is restricted to research use and the app labels its score a ranking, not a probability"],
    ["Test-set contamination by iteration",
      { text: "CONTROLLED", options: { color: GREEN, bold: true } },
      "Dev half is lookable-at, sealed half is not · protocol committed before opening · the scoring script refuses a second run"],
    ["Patient identifiers reaching the public repository",
      { text: "OCCURRED\nfix is item 1", options: { color: AMBER, bold: true } },
      "Our own automated guard test detected it and is currently failing on purpose. Redaction, history rewrite and a blocking CI gate are the first item on the next-steps list"],
    ["No independent grader — labels unverified",
      { text: "OPEN\nthe long pole", options: { color: AMBER, bold: true } },
      "Outreach to a paediatric retina specialist is drafted and in progress; ~300 stratified images for re-grading. Needs a person, not compute"],
  ];
  tbl(s, rows, {
    x: M + 5.85, y: 1.86, w: CW - 5.85, colW: [1.92, 1.32, 3.59],
    rowH: [0.28, 0.9, 0.62, 0.84, 0.72, 0.9, 0.72], fontSize: 9.5,
  });
  s.addNotes("Risk 5 is the one to raise yourself. The project's own guard test caught it — that is the system working, and the fix is item 1 with a date.");
}

// =====================================================================
// 12 — C5 IT RUNS
// =====================================================================
{
  const s = newSlide();
  title(s, "Proof of concept: the whole thing runs, on a laptop", "One command from a clean clone. The served model is identified by SHA-256, not assumed.", "C5");

  s.addText("What happens when you run it", {
    x: M, y: 1.5, w: 6.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const flow = [
    ["./demo.sh", "One command. Starts the Flask app locally. 34 demo images are committed, so a fresh clone has something to open."],
    ["Upload a fundus photograph", "The out-of-domain guard runs first. It refuses adult retinas — added after an audit found the model had flagged 41 of 45 adult eyes as ROP-positive."],
    ["Preprocess → one forward pass", "circle-crop, letterbox, CLAHE, 384², normalise. A single forward pass — no test-time augmentation on the served path."],
    ["Read the result", "Predicted ICROP stage · all six class scores · a binary \"any disease\" score, labelled in the interface as a ranking, not a probability of disease."],
    ["Explain and record", "Grad-CAM heat map and a downloadable PDF report — with the heat map's own audited limitation stated on slide 15."],
  ];
  let fy = 1.86;
  flow.forEach((f, i) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: M, y: fy + 0.07, w: 0.32, h: 0.32,
      fill: { color: AMBER }, line: { color: AMBER },
    });
    s.addText(String(i + 1), {
      x: M, y: fy + 0.07, w: 0.32, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle",
    });
    s.addText(f[0], {
      x: M + 0.44, y: fy + 0.02, w: 5.55, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: NAVY, valign: "middle",
    });
    s.addText(f[1], {
      x: M + 0.44, y: fy + 0.31, w: 5.55, h: 0.64, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10.5, color: GREY, valign: "top",
    });
    fy += 0.98;
  });

  card(s, M + 6.35, 1.5, CW - 6.35, 2.4, LIGHT);
  s.addText("Measured latency, this machine", {
    x: M + 6.55, y: 1.6, w: 5.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 13.5, bold: true, color: NAVY, valign: "middle",
  });
  const lrows = [
    hdr(["Stage", "CPU", "MPS"]),
    ["Preprocessing", "26.7 ms", "—"],
    ["Forward pass, 384²", "85–115 ms", "32.3 ms"],
    ["End to end, one image", "~112 ms", "~59 ms"],
    [{ text: "One 6-frame examination", options: { bold: true } }, { text: "0.7–0.8 s", options: { bold: true, color: AMBER } }, { text: "~0.4 s", options: { bold: true, color: AMBER } }],
  ];
  tbl(s, lrows, {
    x: M + 6.55, y: 1.96, w: CW - 6.75, colW: [3.0, 1.6, 1.43],
    rowH: 0.33, fontSize: 10.5,
  });
  s.addText("Two independent timing passes disagreed on the forward pass; both are kept in the evidence package and the slower one is quoted here.", {
    x: M + 6.55, y: 3.58, w: CW - 6.75, h: 0.28, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 9.5, italic: true, color: GREY,
  });

  card(s, M + 6.35, 4.04, CW - 6.35, 2.9, TINT);
  s.addText("What makes this a credible demo rather than a screenshot", {
    x: M + 6.55, y: 4.14, w: 5.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 13.5, bold: true, color: NAVY, valign: "middle",
  });
  s.addText([
    { text: "The served checkpoint is identified, not assumed. ", options: { bold: true } },
    { text: "SHA-256 11b43960…, 82,964,554 bytes. We re-ran those exact weights over 663 images and matched them to the fold-0, seed-42 checkpoint at 100% prediction agreement (max probability difference 3 × 10⁻⁶).", options: { breakLine: true } },
    { text: "Preprocessing parity is verified. ", options: { bold: true } },
    { text: "Training, evaluation and the web app all call the same build_transforms() on the same config — so the demo is scoring images the way the reported numbers were computed.", options: { breakLine: true } },
    { text: "Raw per-photograph predictions are committed. ", options: { bold: true } },
    { text: "Every number in this deck was recomputed from them for this review rather than copied from an earlier report.", options: {} },
  ], { x: M + 6.55, y: 4.5, w: CW - 6.75, h: 2.36, isTextBox: true, margin: 0, fontFace: BF, fontSize: 10.5, color: NAVY, lineSpacing: 13.5 });

  s.addNotes("If there is a live demo slot, run ./demo.sh here. Have a demo image ready. Mention the out-of-domain guard — panels like a system that knows what it should refuse.");
}

// =====================================================================
// 13 — C5 TEST EVIDENCE
// =====================================================================
{
  const s = newSlide();
  title(s, "Test evidence: what it scores, at home and away", "Left: the numbers. Right: the same story as a picture. The external column was opened exactly once.", "C5");

  const rows = [
    hdr(["Metric", "Internal\n5-fold, patient-grouped\n6,700 images", "External locked\n1 new hospital\n661 images, opened once"]),
    [{ text: "Exact-category accuracy", options: { bold: true } }, "0.786", { text: "0.546", options: { color: RED } }],
    [{ text: "Ordered agreement (QWK)", options: { bold: true } }, "0.897  [0.869 – 0.923]", { text: "0.493 per image\n0.813 per eye", options: { color: AMBER, bold: true } }],
    [{ text: "\"Any disease\" AUC", options: { bold: true } }, "0.970", { text: "0.835", options: { color: GREEN } }],
    [{ text: "Normal recall", options: { bold: true } }, "0.951", { text: "0.996", options: { color: GREEN, bold: true } }],
    [{ text: "macro-F1  (floor = 0.187)", options: { bold: true } }, "0.692 ± 0.086", "—"],
    [{ text: "Stage 1 / 2 / 3 recall", options: { bold: true } }, "0.499 / 0.631 / 0.713", { text: "0.276 / 0.305 / 0.292", options: { color: RED } }],
    [{ text: "Stage 4/5 and AP-ROP", options: { bold: true } }, "0.775 / 0.345 — treat with suspicion", { text: "n = 4 and n = 12\nDeclared not externally\nvalidated, in advance", options: { color: GREY, italic: true } }],
  ];
  tbl(s, rows, {
    y: 1.58, w: 7.05, colW: [2.4, 2.35, 2.3],
    rowH: [0.62, 0.36, 0.52, 0.36, 0.36, 0.36, 0.36, 0.62], fontSize: 10.5,
  });

  s.addChart(pres.ChartType.bar, [
    { name: "Internal (5-fold, unseen babies)", labels: ["Accuracy", "QWK (eye level)", "Any-disease AUC", "Normal recall"], values: [0.786, 0.897, 0.970, 0.951] },
    { name: "External locked hospital", labels: ["Accuracy", "QWK (eye level)", "Any-disease AUC", "Normal recall"], values: [0.546, 0.813, 0.835, 0.996] },
  ], {
    x: M + 7.3, y: 1.55, w: CW - 7.3, h: 3.5,
    barDir: "col", chartColors: [TEAL, AMBER],
    showTitle: true, title: "Internal vs external, matched metrics", titleFontSize: 12, titleColor: NAVY, titleFontFace: BF,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 9, dataLabelColor: NAVY,
    showLegend: true, legendPos: "b", legendFontSize: 9, legendColor: NAVY,
    catAxisLabelColor: NAVY, catAxisLabelFontSize: 9, valAxisLabelColor: GREY, valAxisLabelFontSize: 9,
    valAxisMaxVal: 1.1, valAxisMinVal: 0, valGridLine: { color: "E4E9ED", size: 0.75 }, catGridLine: { style: "none" },
    barGapWidthPct: 55,
  });

  card(s, M + 7.3, 5.2, CW - 7.3, 1.62, TINT);
  s.addText([
    { text: "Read the last two bars together. ", options: { bold: true, color: AMBER } },
    { text: "\"Is anything wrong with this eye?\" transfers to a new hospital — AUC 0.835, and it finds 99.6% of healthy eyes. \"Which stage is it?\" largely does not: per-stage recall falls to about one call in four. Slide 14 is why, and how much of that is the model versus the labels.", options: { color: NAVY } },
  ], { x: M + 7.5, y: 5.3, w: CW - 7.7, h: 1.42, isTextBox: true, margin: 0, fontFace: BF, fontSize: 11, valign: "middle", lineSpacing: 14 });

  s.addText("Internal figures are pooled out-of-fold across the five patient-grouped folds; no model ever scored a baby it trained on. External figures are from the sealed half, scored once on 18 Aug 2026 under a protocol committed to version control beforehand.", {
    x: M, y: 6.5, w: 7.05, h: 0.6, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 9.5, italic: true, color: GREY,
  });
  s.addNotes("Do not hide the 0.546. Lead with it — then give the mechanism on the next slide. A panel trusts a team that volunteers its worst number.");
}

// =====================================================================
// 14 — C5 THE FINDING
// =====================================================================
{
  const s = newSlide(NAVY);
  title(s, "The result the project is actually for", "Why the external number looks bad — and why a large part of it is a labelling artefact we can prove.", "C5", true);

  s.addText([
    { text: "ROP grows at the outer (temporal) edge of the retina. ", options: { bold: true, color: AMBER } },
    { text: "A session photographs the eye in several directions and stamps the eye's one diagnosis onto every frame — including the frames pointing at the centre and the nose, which physically cannot show the ridge. Score those frames per photograph and you charge the model for annotations that were never about them.", options: { color: WHITE } },
  ], { x: M, y: 1.5, w: CW, h: 0.72, isTextBox: true, margin: 0, fontFace: BF, fontSize: 12.5, valign: "middle", lineSpacing: 16 });

  s.addChart(pres.ChartType.bar, [{
    name: "QWK, image level, no aggregation",
    labels: ["Temporal\n(outer edge —\nwhere ROP grows)", "Nasal\n(nose side)", "Posterior pole\n(back of the eye)"],
    values: [0.841, 0.352, 0.229],
  }], {
    x: M, y: 2.35, w: 6.0, h: 3.35,
    barDir: "col", chartColors: [AMBER, TEAL, TEAL],
    varyColors: true,
    showTitle: true, title: "Same predictions, split by where the camera pointed", titleFontSize: 12, titleColor: WHITE, titleFontFace: BF,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: WHITE,
    showLegend: false,
    catAxisLabelColor: "C2CEDA", catAxisLabelFontSize: 9, valAxisLabelColor: "8FA3B5", valAxisLabelFontSize: 9,
    valAxisMaxVal: 1.0, valAxisMinVal: 0, valGridLine: { color: "23425F", size: 0.75 }, catGridLine: { style: "none" },
    plotArea: { fill: { color: NAVY } }, chartArea: { fill: { color: NAVY } },
    barGapWidthPct: 60,
  });

  s.addText("No aggregation is involved. These are single photographs, scored one at a time, against the same labels.", {
    x: M, y: 5.72, w: 6.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 10, italic: true, color: "9FB2C4",
  });

  const pts = [
    ["It replicates.", "Two architectures × two random seeds = four independent cells. All four move in the same direction, at similar magnitude, on a set opened once. This is not an aggregation trick."],
    ["It is bigger than architecture.", "Changing the evaluation unit moves QWK by roughly 0.35. Changing the architecture moves nothing that survives a change of random seed."],
    ["It has a consequence for the field.", "ICROP staging results are not comparable across studies unless the label's native unit is stated. Models should be scored per eye, and papers should report both units."],
    ["It does not rescue our model.", "Even per eye, it still misses about 18 diseased eyes in every 100. We state this in the same breath as the finding, not three slides later."],
  ];
  let py = 2.35;
  pts.forEach((p) => {
    card(s, M + 6.35, py, CW - 6.35, 1.0, NAVY2);
    s.addText(p[0], {
      x: M + 6.55, y: py + 0.08, w: CW - 6.75, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: AMBER, valign: "middle",
    });
    s.addText(p[1], {
      x: M + 6.55, y: py + 0.36, w: CW - 6.75, h: 0.58, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10.5, color: WHITE, valign: "top",
    });
    py += 1.08;
  });

  s.addText([
    { text: "The two other things the numbers say, which we are not hiding:   ", options: { bold: true, color: AMBER } },
    { text: "(1) the percentages are not probabilities — calibration error 0.109 internally and 0.334 externally, where certainty-when-right and certainty-when-wrong differ by only 0.041.   (2) The model is most confident exactly when it makes its most dangerous error: calling a diseased eye Normal carries its highest mean confidence, 0.934, and 141 of those 186 cases were above 90%. No confidence threshold makes that safe.", options: { color: "D9E3EC" } },
  ], { x: M, y: 6.2, w: CW, h: 0.82, isTextBox: true, margin: 0, fontFace: BF, fontSize: 10.5, valign: "middle", lineSpacing: 13.5 });

  s.addNotes("This is the slide to spend time on. It is the project's contribution, it is defensible, and it is the reason the paper exists. Finish with the fourth card so you are not overselling.");
}

// =====================================================================
// 15 — C5 WHY TRUST IT
// =====================================================================
{
  const s = newSlide();
  title(s, "Why these numbers can be trusted", "The tests we ran against ourselves, and the results we published against our own interest.", "C5");

  s.addText("Verification", {
    x: M, y: 1.5, w: 6.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const ver = [
    ["Zero leakage, recomputed from scratch", "0 shared infants and 0 shared images across every train/val, corpus/dev, corpus/locked and dev/locked comparison. The five folds exactly partition all 6,700 images."],
    ["The model beats its own shortcut floor", "A metadata-only classifier using no pixels scores macro-F1 0.187. The image model scores 0.692. The gap is the evidence that it reads retinas."],
    ["The sealed set was opened once", "Protocol written and committed before opening; opened 18 Aug 2026; the script refuses a second run. It has not been re-scored since."],
    ["Every number is recomputable", "Raw per-photograph predictions are committed. The 275-file evidence package — 54 charts, 49 tables, 60 documents — was rebuilt from them for this review."],
  ];
  let vy = 1.86;
  ver.forEach((v) => {
    card(s, M, vy, 6.0, 1.07, LIGHT);
    s.addText(v[0], {
      x: M + 0.18, y: vy + 0.08, w: 5.64, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: GREEN, valign: "middle",
    });
    s.addText(v[1], {
      x: M + 0.18, y: vy + 0.36, w: 5.64, h: 0.64, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10.5, color: NAVY, valign: "top",
    });
    vy += 1.15;
  });

  s.addText("Published against our own interest", {
    x: M + 6.35, y: 1.5, w: 6.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 14, bold: true, color: NAVY,
  });
  const neg = [
    ["We retired a model that looked good", "The earlier ResNet50 screener flagged 663 of 663 images at a new hospital — every healthy eye included. Withdrawn, and kept in the package as the canonical demonstration that thresholds do not transfer."],
    ["We dropped a disease rather than quote it", "The glaucoma arm double-mounts per eye, which contaminates its AUC. Rather than publish a broken 0.973 we removed glaucoma from scope."],
    ["We corrected our own headline metric", "An audit found the reported QWK did not match its own declared definition — AP-ROP was being scored as ordinal rank 5. Correcting it moved fold-0 QWK from 0.569 to 0.908. Both values are published, with the audit trail."],
    ["We report two negative results at full weight", "The site adversary does not remove site information under a disease-controlled probe. The taxonomy-faithful head does not beat a flat softmax — and is 3.2× worse calibrated at equal accuracy."],
    ["We publish the limits of our own explanation", "Grad-CAM attention on the retina is at chance: 56.0% of attention on a region occupying 53.8% of the frame, and 30.7% of peaks fall outside the retina. So the heat map cannot be used to verify the model's reasoning, and we say so."],
  ];
  let ny = 1.86;
  neg.forEach((n) => {
    card(s, M + 6.35, ny, CW - 6.35, 0.92, TINT);
    s.addText(n[0], {
      x: M + 6.53, y: ny + 0.06, w: CW - 6.71, h: 0.26, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, bold: true, color: RED, valign: "middle",
    });
    s.addText(n[1], {
      x: M + 6.53, y: ny + 0.32, w: CW - 6.71, h: 0.56, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10, color: NAVY, valign: "top",
    });
    ny += 1.0;
  });

  s.addNotes("This slide is the one that separates a 4 from a 5 on 'test evidence is shown and behaviour is explained'. The point: we tried hardest to break our own results.");
}

// =====================================================================
// 16 — C6 TEAM
// =====================================================================
{
  const s = newSlide();
  title(s, "Who owns what — and what each of us can defend", "Four workstreams, four owners. The last column is what that member should be asked about.", "C6");

  const rows = [
    hdr(["Owner", "Workstream", "Artefacts owned", "Can explain, without notes"]),
    [{ text: "[Member 1]", options: { bold: true, color: TEAL } },
      { text: "Data, provenance and ethics", options: { bold: true } },
      "11 dataset acquisitions with DOIs and licences · PROVENANCE.json · de-identification and filename rewrite · the inclusion cascade (13,037 → 3,112)",
      "Why only 4 of 11 datasets are usable · why 43% of one corpus's \"ROP-positive\" is not ROP · why we cannot collect our own data · what CC BY-NC forbids us from doing"],
    [{ text: "[Member 2]", options: { bold: true, color: TEAL } },
      { text: "Modelling and training", options: { bold: true } },
      "EfficientNetV2-S backbone · CORN ordinal head · AP-ROP branch · DANN site adversary · the 5-fold runs at two seeds · the retired v1 arm",
      "Why natural prevalence and not a balanced sampler · why last-epoch weights · what CORN does and why AP-ROP is masked out of it · why the backbone sweep was stopped"],
    [{ text: "[Member 3]", options: { bold: true, color: TEAL } },
      { text: "Evaluation and statistics", options: { bold: true } },
      "The five shortcut gates · patient-clustered confidence intervals · the locked-set protocol · calibration analysis · the evaluation-unit study",
      "What QWK and ECE measure · why image-level ≠ eye-level and by how much · why a paired t-test across folds, and what p = 0.589 does and does not mean"],
    [{ text: "[Member 4]", options: { bold: true, color: TEAL } },
      { text: "Application, deployment and reproducibility", options: { bold: true } },
      "Flask app · Grad-CAM · PDF report · demo.sh and the 34 demo images · the repo guard tests · the 275-file evidence package build scripts",
      "The latency budget and where it goes · the out-of-domain guard and why it exists · how the served checkpoint was identified by hash · how any number here is recomputed"],
  ];
  tbl(s, rows, {
    y: 1.6, colW: [1.25, 2.1, 4.35, 4.53],
    rowH: [0.34, 1.2, 1.2, 1.2, 1.28], fontSize: 10,
  });

  s.addText([
    { text: "Shared across all four: ", options: { bold: true, color: AMBER } },
    { text: "the inclusion cascade, the locked-set rule, and the evaluation-unit finding. Any one of us should be able to state those three unprompted — they are the project's spine.", options: { color: NAVY } },
  ], { x: M, y: 6.5, w: CW, h: 0.45, isTextBox: true, margin: 0, fontFace: BF, fontSize: 11.5, valign: "middle" });

  s.addNotes("REPLACE THE FOUR PLACEHOLDER NAMES BEFORE PRESENTING. The rubric's top band requires that all members explain the work, so rehearse the last column — the panel will pick a member and a row.");
}

// =====================================================================
// 17 — C6 PROGRESS
// =====================================================================
{
  const s = newSlide();
  title(s, "Progress since the proposal", "Eight milestones, four of which are decisions to withdraw something. Dates are from the repository history.", "C6");

  const tl = [
    ["Jun 2026", "First dataset and a working baseline", "Shenzhen ROP acquired; binary ROP screening model trained end to end. The project has a pipeline.", TEAL],
    ["Jul 2026", "Corpus assembled and audited", "All 11 datasets acquired (13,037 images), de-identified, licences and DOIs logged. DR and glaucoma arms built alongside.", TEAL],
    ["14 Aug 2026", "Hosted app deleted to purge patient images from git history", "A privacy problem found in our own repository, fixed by removing the deployment rather than patching around it.", RED],
    ["18 Aug 2026", "The sealed external half opened — once", "661 images, 116 infants, scored under a protocol committed beforehand. Not re-scored since.", AMBER],
    ["19 Aug 2026", "ResNet50 screener retired after the 663/663 audit", "The screening head was re-based on the structured model. The failure is kept in the package as evidence, not deleted.", RED],
    ["Aug 2026", "Glaucoma withdrawn from scope · 12-paper survey completed", "The contaminated 0.973 was dropped rather than quoted. The survey set the comparison table on slide 5.", RED],
    ["Sep 2026", "Structured ≈ flat established; the unit finding replicated", "5-fold CV at two seeds shows architectural equivalence; the evaluation-unit effect holds in all four model-seed cells.", TEAL],
    ["21 Sep 2026", "Full evidence package rebuilt from raw predictions", "275 files, 54 charts, 49 tables, 60 documents — every number recomputed rather than copied forward. This deck is built from it.", TEAL],
  ];
  let ty = 1.52;
  tl.forEach((t) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: M + 1.5, y: ty + 0.18, w: 0.2, h: 0.2,
      fill: { color: t[3] }, line: { color: t[3] },
    });
    s.addText(t[0], {
      x: M, y: ty + 0.06, w: 1.4, h: 0.42, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, bold: true, color: t[3], align: "right", valign: "middle",
    });
    s.addText(t[1], {
      x: M + 1.9, y: ty, w: 5.0, h: 0.56, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: NAVY, valign: "middle",
    });
    s.addText(t[2], {
      x: M + 7.0, y: ty, w: CW - 7.0, h: 0.56, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10.5, color: GREY, valign: "middle",
    });
    ty += 0.63;
  });

  s.addText([
    { text: "Four of these eight milestones are withdrawals. ", options: { bold: true, color: AMBER } },
    { text: "That is deliberate: the fastest progress this project made came from finding out that something we had built did not survive an audit, and removing it.", options: { color: NAVY } },
  ], { x: M, y: 6.66, w: CW, h: 0.4, isTextBox: true, margin: 0, fontFace: BF, fontSize: 11.5, valign: "middle" });

  s.addNotes("The red dots are the withdrawals. Point at them on purpose.");
}

// =====================================================================
// 18 — C6 NEXT PLAN
// =====================================================================
{
  const s = newSlide();
  title(s, "The plan from here — ten items, dated and owned", "Items 1–5 need no GPU and are two days of work. Item 8 is the deadline everything else serves.", "C6");

  const rows = [
    hdr(["#", "Action", "Owner", "GPU?", "By"]),
    ["1", { text: "Redact the two patient-identifying filenames, rewrite git history, and make the guard test block commits instead of merely reporting", options: { bold: true, color: RED } }, "[M4] App", "No", { text: "23 Sep 2026", options: { bold: true } }],
    ["2", "Temperature-scale the probabilities on the already-saved out-of-fold logits — fixes the calibration failure without retraining", "[M3] Eval", "No", "26 Sep 2026"],
    ["3", "Set weighted_sampler: false in the committed config and stamp the checkpoint's fold, seed and hash into WEIGHTS.json, so the repository reproduces its own headline model", "[M2] Model", "No", "26 Sep 2026"],
    ["4", "Re-derive the served screening threshold on data the served checkpoint never saw — the current 0.0155 was chosen on pooled folds it partly trained on", "[M3] Eval", "No", "30 Sep 2026"],
    ["5", "Make the temporal-view result the paper's headline and rebuild the figure set around it", "All", "No", "3 Oct 2026"],
    ["6", "Re-run the component ablations (CORN-removed, branch-removed, w_site = 0) on folds 1–4 at both seeds — the one surviving head effect was measured on fold 0 only", "[M2] Model", { text: "~10 h", options: { color: AMBER } }, "12 Oct 2026"],
    ["7", "Log validation loss and learning rate, re-run one fold, and check for late-epoch overfitting the current logs cannot show", "[M2] Model", { text: "~70 min", options: { color: AMBER } }, "14 Oct 2026"],
    ["8", { text: "ISBI 2027 submission — 4 pages, the locked-set evaluation as its evidence base", options: { bold: true, color: AMBER } }, "All", "No", { text: "26 Oct 2026", options: { bold: true, color: AMBER } }],
    ["9", { text: "Paediatric ophthalmologist re-grades ~300 stratified images — the only way to separate a model failure from a label disagreement, and to get a human ceiling", options: { bold: true } }, "[M1] Data", "No", "started — weeks"],
    ["10", "Locate a second external hospital, to tell \"this site is unusual\" apart from \"the model degrades everywhere\"", "[M1] Data", "No", "months"],
  ];
  tbl(s, rows, {
    y: 1.56, colW: [0.36, 7.7, 1.42, 1.1, 1.65],
    rowH: [0.28, 0.58, 0.48, 0.58, 0.58, 0.4, 0.58, 0.48, 0.4, 0.58, 0.44], fontSize: 10,
  });
  s.addText("Item 9 is the long pole and it cannot be bought with compute. It is why the outreach draft was written before this review, not after it.", {
    x: M, y: 6.76, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 10.5, italic: true, color: GREY,
  });
  s.addNotes("Item 1 is first for a reason — it is a live privacy defect in a public repo, caught by our own test. Say that plainly; a panel respects it more than a clean-looking slide.");
}

// =====================================================================
// 19 — CLOSING
// =====================================================================
{
  const s = newSlide(NAVY);
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.5, y: -1.4, w: 5.1, h: 5.2, rectRadius: 0.5,
    fill: { color: NAVY2 }, line: { color: NAVY2 },
  });

  s.addText("Where this project stands", {
    x: M, y: 0.85, w: 8.0, h: 0.6, isTextBox: true, margin: 0,
    fontFace: HF, fontSize: 34, bold: true, color: WHITE,
  });

  s.addText("The unit an ROP model is scored over changes its headline agreement by more than its architecture, its initialisation, or both together — and we can show it four times over, on a test set we opened once.",
    { x: M, y: 1.58, w: 7.7, h: 1.72, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 17, color: WHITE, italic: true, lineSpacing: 25, valign: "top" });

  const close = [
    ["What we are claiming", "An audited, patient-grouped, externally validated ICROP staging benchmark; a measurement result about how these models should be scored; and a CPU-only prototype that reports its own failures.", GREEN],
    ["What we are not claiming", "Any clinical validity, utility or safety. Per-stage staging at a new hospital is about one call in four, the probabilities are not probabilities, and no clinician has re-read our labels.", RED],
    ["What we are asking the panel for", "Assess the audit, the honesty of the reporting, and the plan — not a headline accuracy. And if anyone on the panel can introduce us to a paediatric retina specialist, that unblocks item 9.", AMBER],
  ];
  let cy = 3.32;
  close.forEach((c) => {
    s.addText(c[0], {
      x: M, y: cy, w: 7.6, h: 0.3, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 13, bold: true, color: c[2], valign: "middle",
    });
    s.addText(c[1], {
      x: M, y: cy + 0.3, w: 7.6, h: 0.75, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 11.5, color: "D9E3EC", valign: "top",
    });
    cy += 1.12;
  });

  const facts = [
    ["13,037 → 3,112", "acquired → trained on, every exclusion counted"],
    ["0", "patients or images leaked across any split"],
    ["661", "sealed images, opened exactly once"],
    ["275", "files of evidence, all recomputable"],
  ];
  facts.forEach((f, i) => {
    const y = 1.55 + i * 0.92;
    s.addText(f[0], {
      x: 9.15, y, w: 3.6, h: 0.42, isTextBox: true, margin: 0,
      fontFace: HF, fontSize: 24, bold: true, color: AMBER, valign: "middle",
    });
    s.addText(f[1], {
      x: 9.15, y: y + 0.42, w: 3.6, h: 0.42, isTextBox: true, margin: 0,
      fontFace: BF, fontSize: 10, color: "9FB2C4", valign: "top",
    });
  });

  s.addText("Thank you — questions welcome, including the uncomfortable ones.", {
    x: M, y: 6.72, w: 8.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BF, fontSize: 13, bold: true, color: WHITE, valign: "middle",
  });
  s.addNotes("Close on the ask. Inviting hard questions signals you have already asked them yourselves — which is the whole thesis of this deck.");
}

pres.writeFile({ fileName: "RetinAI_Post_Proposal_Review.pptx" })
  .then(f => console.log("WROTE " + f));
