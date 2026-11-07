const $ = (id) => document.getElementById(id);
let file = null, heatmaps = [], context = null;

const drop = $("drop"), fileInput = $("file"), preview = $("preview"), dropmsg = $("dropmsg");
$("browse").onclick = (e) => { e.preventDefault(); fileInput.click(); };
fileInput.onchange = (e) => setFile(e.target.files[0]);
["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

// ── patient context — read from the server so the UI never hard-codes patient types ──
(async function loadContexts() {
  const group = $("ctxgroup");
  if (!group) return;
  let choices = [];
  try { choices = await fetch("/contexts").then(r => r.json()); }
  catch { group.innerHTML = `<p class="muted small">Could not load patient contexts.</p>`; return; }
  group.innerHTML = choices.map(c => `
    <label class="ctx-opt${c.advisory ? " advisory" : ""}">
      <input type="radio" name="context" value="${c.key}">
      <span class="ctx-label">${c.label}${c.advisory ? ' <em class="ctx-warn">not clinically valid</em>' : ""}</span>
      <span class="ctx-hint">${c.hint || ""}</span>
    </label>`).join("");
  group.querySelectorAll('input[name="context"]').forEach(r => {
    r.onchange = () => { context = r.value; syncAnalyze(); };
  });
  syncAnalyze();
})();

// Analyse stays disabled until BOTH an image and a context exist. There is no default
// context: defaulting would silently reinstate the unrouted behaviour this phase removes.
function syncAnalyze() {
  const btn = $("analyze");
  if (!btn) return;
  btn.disabled = !(file && context);
  btn.title = btn.disabled
    ? (!context ? "Choose a patient context first" : "Add a fundus image")
    : "";
}

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  eyeSummary = null;
  file = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false; dropmsg.hidden = true;
  syncAnalyze();
}

// A plain status line while the two models run. No forced wait: the page shows the
// answer as soon as the server has it.
function busy(on, text) {
  const s = $("status");
  if (s) { s.textContent = text || ""; s.hidden = !on; }
  ["analyze", "analyze-eye"].forEach(id => { const b = $(id); if (b) b.disabled = on || b.disabled; });
  if (!on) { syncAnalyze(); syncEye(); }
}

$("analyze").onclick = async () => {
  if (!file) return;
  busy(true, "Analysing — both models are running on this photo…");
  $("results").hidden = true;
  eyeSummary = null; $("eyecard").hidden = true; $("eyephotos").hidden = true;
  const fd = new FormData(); fd.append("image", file); fd.append("context", context);
  let res;
  try {
    const r = await fetch("/predict", { method: "POST", body: fd });
    res = await readJson(r);
    if (!r.ok) throw new Error(res.detail || res.error || "Analysis failed");
  } catch (e) {
    busy(false);
    alert(e.message || "Analysis failed."); return;
  }
  render(res);
  busy(false);
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
};

// The 6-class ICROP staging research preview. The backend has always returned it; nothing
// rendered it, so the preview advertised on three pages never once reached the screen.
// Reconcile the two models IN THE UI. They answer different questions with different
// decision rules, so "ROP Detected" beside "ICROP stage: Normal" reads as a contradiction
// when it is not one — the screening head asks "any ROP?" at a threshold tuned to
// over-refer, while the staging head reports its single most likely class at argmax.
// Comparing them on the same quantity (any-ROP probability) makes the disagreement legible.
function stagingVsScreening(st, rop) {
  if (!rop || !rop.decision || st.any_rop_probability === undefined) return "";
  const d = rop.decision;
  const pos = d.positive, normal = st.prediction === "Normal";
  if (pos === normal) return "";                       // nothing to reconcile
  // Since the re-basing these come from ONE model, so this is no longer two opinions in
  // conflict — it is one number read two ways. The verdict adds up every disease stage and
  // compares the total to a deliberately low line; the stage label just names the single
  // tallest bar. Both can be true at once.
  const same = (d.model || "").indexOf("staging") >= 0;
  if (same) {
    return `<div class="reconcile"><b>Same model, two questions.</b>
      The verdict adds up <i>all</i> the disease stages (<b>${st.any_rop_probability}%</b>)
      and flags anything over <b>${d.threshold}%</b>. The stage below is just the tallest
      single bar. So "${st.prediction}" and "${pos ? "flagged" : "not flagged"}" do not
      contradict each other.</div>`;
  }
  return `<div class="reconcile"><b>The two models disagree.</b>
    Screening ${pos ? "flagged this eye" : "called it negative"}; staging puts any ROP at
    <b>${st.any_rop_probability}%</b>. Neither is a diagnosis — a clinician decides.</div>`;
}

function renderStaging(st, rop) {
  const host = $("stagingcard");
  if (!host) return;
  if (!st) { host.hidden = true; host.innerHTML = ""; return; }
  const entries = Object.entries(st.scores).sort((a, b) => b[1] - a[1]);
  const bars = entries.map(([k, v]) =>
    `<div class="scoreline"><span${k === st.prediction ? ' class="on"' : ""}>${k}</span><span>${v}%</span></div>
     <div class="bar"><span class="${k === st.prediction ? "on" : ""}" style="width:${v}%"></span></div>`).join("");
  const any = st.any_rop_probability === undefined ? "" :
    `<div class="anyrop"><span>Any ROP <i>(stages 1–3, 4/5 and AP-ROP added together)</i></span>
     <b>${st.any_rop_probability}%</b></div>`;
  host.innerHTML = `
    <div class="card staging">
      <h3>Our model — ICROP stage: ${st.prediction}
        <span class="risk r-preview">research preview</span></h3>
      <div class="rec">Six-class ICROP-3 staging by the served ensemble. This is <b>not</b> the
        screening decision above and must not be read as one.</div>
      ${any}
      ${stagingVsScreening(st, rop)}
      ${bars}
      <p class="dm-raw">${st.note || ""}</p>
    </div>`;
  host.hidden = false;
}

// The comparison model's stage, beside ours. It never decides: the verdict, the Grad-CAM and
// the PDF stay with the served model. The agreement mark is the whole point of the panel.
function agreeBadge(a) {
  if (!a) return "";
  const cls = a.status === "agree" ? "agree-yes" : a.status === "disagree" ? "agree-no" : "agree-na";
  const txt = a.status === "agree" ? "models agree"
    : a.status === "disagree" ? (a.adjacent ? "disagree · one stage apart" : "disagree")
    : a.status === "not_covered" ? "not covered by comparison model" : "no reference";
  return `<span class="agree ${cls}">${txt}</span>`;
}

function renderComparison(cmp, st) {
  const host = $("comparisoncard");
  if (!host) return;
  if (!cmp) { host.hidden = true; host.innerHTML = ""; return; }
  const entries = Object.entries(cmp.scores).sort((a, b) => b[1] - a[1]);
  const bars = entries.map(([k, v]) =>
    `<div class="scoreline"><span${k === cmp.prediction ? ' class="on"' : ""}>${k}</span><span>${v}%</span></div>
     <div class="bar"><span class="${k === cmp.prediction ? "on" : ""}" style="width:${v}%"></span></div>`).join("");
  const a = cmp.agreement || {};
  host.innerHTML = `
    <div class="card comparison">
      <h3>Comparison model — stage: ${cmp.prediction} ${agreeBadge(a)}</h3>
      <div class="compare-row">
        <div>Our model<b>${st ? esc(st.prediction) : "—"}</b></div>
        <div>Comparison model<b>${esc(cmp.prediction)}</b></div>
      </div>
      <div class="rec">${esc(a.message || "")}</div>
      <div class="anyrop"><span>Any ROP <i>(stages 1–3 added together; this model has no Stage 4/5 or AP-ROP)</i></span>
        <b>${cmp.any_rop_probability}%</b></div>
      ${bars}
      <p class="dm-raw">${esc(cmp.model)} · ${esc(cmp.credit || "")} · ${cmp.n_members} seeds averaged.
        ${esc(cmp.note || "")}</p>
    </div>`;
  host.hidden = false;
}

function render(res) {
  // Gradability gate — when the image is not gradable there are no findings to show, and
  // the report download must stay blocked. Fail closed in the UI too.
  const q = $("qualitybanner"), dl = $("download");
  const ungradable = res.gradable === false;
  // A gradable-but-downscaled image still gets an answer, and that answer moved by 0.08 AUC
  // in testing. Showing nothing would imply the result is unaffected, so the same banner
  // carries the resolution warning when the image passed the gate but is small.
  const rn = res.resolution_notice;
  if (q) {
    if (ungradable) {
      q.textContent = `Not gradable — ${res.quality.reason} No models were run on this image.`;
      q.classList.remove("warn-soft");
      q.hidden = false;
    } else if (rn) {
      q.textContent = `Low-resolution image — ${rn.message}`;
      q.classList.add("warn-soft");
      q.hidden = false;
    } else {
      q.textContent = "";
      q.classList.remove("warn-soft");
      q.hidden = true;
    }
  }
  // The seen-photo guard: a photo either model trained or tested on is not a fair demo.
  const sb = $("seenbanner");
  if (sb) {
    const sd = res.seen_in_data;
    sb.textContent = sd ? sd.message : "";
    sb.classList.toggle("warn-soft", !sd || sd.level === "warn");
    sb.classList.toggle("info", !!sd && sd.level === "info");
    sb.hidden = !sd;
  }
  if (dl) {
    dl.disabled = ungradable;
    dl.title = ungradable ? "No report can be issued for an ungradable image" : "";
  }

  const banner = $("ctxbanner");
  if (banner && !ungradable) {
    banner.textContent = res.advisory
      ? `Demonstration mode — every model was run regardless of population. These results are NOT clinically valid.`
      : `Screened as: ${res.context_label}. Models outside this population were not run.`;
    banner.classList.toggle("advisory", !!res.advisory);
    banner.hidden = false;
  } else if (banner) {
    banner.hidden = true;
  }

  $("cards").innerHTML = res.findings.map(f => {
    // A routed-out disease is shown explicitly. Hiding it would read as "nothing found",
    // which is the same false reassurance the PDF used to give.
    if (!f.available) {
      const na = f.status === "not_applicable";
      return `<div class="card unavailable${na ? " not-applicable" : ""}">
        <h3>${f.disease}: <span class="na">${na ? "Not assessed" : "Not available"}</span></h3>
        <p class="muted small">${f.note || ""}</p></div>`;
    }
    // Two competing bars imply a 50% decision boundary. This screening model's boundary is
    // 19.3% (tuned for high sensitivity), so "No ROP 43.7%" read as "43.7% chance it's
    // healthy" when it is only 1 - score. A binary finding therefore shows no bars at all:
    // the verdict and the score carry it. The threshold, the measured false-alarm rates and
    // the external audit stay in the /predict response under `decision`.
    const bars = f.decision && f.decision.positive_class
      ? ""
      : Object.entries(f.scores).map(([k, v]) =>
        `<div class="scoreline"><span>${k}</span><span>${v}%</span></div>
         <div class="bar"><span style="width:${v}%"></span></div>`).join("");
    const risk = f.risk ? `<span class="risk r-${f.risk.toLowerCase()}">${f.risk}</span>` : "";
    const adv = f.advisory ? `<span class="risk r-advisory">advisory</span>` : "";
    // C6 — abstention. An uncertain case is NOT a result; say so before the numbers.
    const unc = f.uncertain
      ? `<div class="uncertain-note"><b>Uncertain — needs human review.</b> ${f.uncertain_reason || ""}</div>`
      : "";
    // C4/C8 — never present a raw score as "confidence". State what is actually known.
    const cal = f.calibration || {};
    const calTag = cal.status === "calibrated"
      ? `<span class="cal ok" title="${cal.note || ""}">calibrated</span>`
      : cal.status === "verified-uncalibrated"
        ? `<span class="cal warn" title="${cal.note || ""}">uncalibrated (measured)</span>`
        : `<span class="cal bad" title="${cal.note || ""}">calibration unverified</span>`;
    const ref = f.referable
      ? `<div class="rec">Referable (grade ≥ ${f.referable.grade_threshold}): <b>${f.referable.is_referable ? "yes" : "no"}</b> · ${f.referable.probability}%</div>`
      : "";
    return `<div class="card${f.uncertain ? " uncertain" : ""}"><h3>${f.disease}: ${f.prediction} ${risk}${adv}</h3>
      ${unc}${ref}
      <div class="rec">${f.recommendation || ""} · model score ${f.score}% ${calTag}</div>${bars}</div>`;
  }).join("");

  // Ungradable images carry no staging key at all; pass null explicitly so a stale card
  // from the previous upload can never survive into a "not gradable" result.
  const st = ungradable ? null : res.staging;
  renderStaging(st, res.findings.find(f => f.disease === "ROP" && f.available));
  renderComparison(ungradable ? null : res.comparison, st);

  // visualization — one heatmap per positive disease, not one for the whole image
  heatmaps = res.heatmaps || (res.heatmap ? [{ disease: res.heatmap_disease, image: res.heatmap }] : []);
  $("origimg").src = preview.src;
  const tabs = $("viztabs");
  tabs.innerHTML = `<button class="tab on" data-v="orig">Original</button>` +
    heatmaps.map((h, i) => `<button class="tab" data-v="heat:${i}">${h.disease} heatmap</button>`).join("");
  tabs.querySelectorAll(".tab").forEach(t => t.onclick = () => selectViz(t.dataset.v));
  $("viz").hidden = heatmaps.length === 0 && !preview.src;
  selectViz("orig");
}

// What the map is worth, in numbers, beside the map. A heatmap alone invites the reader
// to believe whichever blob is brightest; these two measurements say whether the blob
// deserves it — how much attention missed the retina entirely, and how tightly it focused.
function evidencePanel(h) {
  const box = $("camevidence");
  if (!box) return;
  const e = h && h.evidence;
  if (!e || !e.measurable) { box.hidden = true; box.innerHTML = ""; return; }
  const rows = [];
  if (e.off_retina_pct !== undefined) {
    const bad = e.off_retina_pct >= 15;
    // Only the CAMERA's surround counts here. Letterbox padding added by preprocessing is
    // black by construction and carries no information, so counting it made this model look
    // like it reads the frame five times more than it does.
    const pad = e.on_padding_pct > 1
      ? ` A further <b>${e.on_padding_pct}%</b> landed on the black padding preprocessing adds,
          which carries no information either way.` : "";
    const lift = e.off_retina_lift !== undefined && e.off_retina_lift !== null
      ? ` Relative to how much of the frame is surround, that is
          <b>${e.off_retina_lift}×</b> what a uniform map would put there.` : "";
    rows.push(`<li class="${bad ? "warn" : "ok"}"><b>${e.off_retina_pct}%</b> of the
      attention fell <b>outside the retina</b>, on the camera's own surround.${lift}
      ${bad ? "That is high — on this image the score is partly driven by the frame, "
            + "not the eye. Treat the map as unreliable."
            : "Low: the model is reading retina, not frame."}${pad}</li>`);
  }
  if (e.concentration_pct !== undefined) {
    const diffuse = e.concentration_pct >= 25;
    rows.push(`<li class="${diffuse ? "warn" : "ok"}">Half the attention sits in
      <b>${e.concentration_pct}%</b> of the image.
      ${diffuse ? "Spread out — the model has not localised a single region."
                : "Focused on a specific region."}</li>`);
  }
  if (e.peak_inside_retina === false) {
    rows.push(`<li class="warn">The single <b>hottest point of the raw map was outside the
      retina</b> altogether — on this image the strongest driver of the score is not
      retinal tissue.</li>`);
  }
  if (e.peak_zone) {
    rows.push(`<li>Strongest region <i>within the retina</i>: <b>${e.peak_zone}</b>, around
      <b>${e.peak_clock} o'clock</b> as displayed.
      <span class="muted">(Clock position only — the app is never told which eye this is,
      so it cannot say temporal or nasal.)</span></li>`);
  }
  box.innerHTML = `<h4>What the model looked at</h4><ul class="cam-ev">${rows.join("")}</ul>
    <p class="dm-raw">${e.note}</p>`;
  box.hidden = false;
}

function selectViz(v) {
  document.querySelectorAll("#viztabs .tab").forEach(t => t.classList.toggle("on", t.dataset.v === v));
  const isHeat = v.startsWith("heat:");
  $("origimg").hidden = isHeat;
  $("heatimg").hidden = !isHeat;
  if (isHeat) {
    const h = heatmaps[Number(v.split(":")[1])];
    $("heatimg").src = "data:image/png;base64," + h.image;
    $("heatcap").textContent = "Grad-CAM — " + h.disease
      + " · from the served model only · coloured inside the retinal field; the white outline is its edge.";
    evidencePanel(h);
  } else {
    $("heatcap").textContent = "";
    evidencePanel(null);
  }
}

$("download").onclick = async () => {
  if (!file || !context || $("download").disabled) return;
  const fd = new FormData(); fd.append("image", file); fd.append("context", context);
  ["name", "age", "gender"].forEach(k => fd.append(k, ($(k) || {}).value || ""));
  fd.append("id", ($("pid") || {}).value || "");
  if (eyeSummary) fd.append("eye_summary", JSON.stringify(eyeSummary));
  const r = await fetch("/report", { method: "POST", body: fd });
  if (!r.ok) { alert("Report failed."); return; }
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "retinal_report.pdf"; a.click();
};

$("again").onclick = () => {
  file = null; preview.hidden = true; dropmsg.hidden = false;
  $("results").hidden = true; fileInput.value = "";
  $("download").disabled = false;
  eyeSummary = null; $("eyecard").hidden = true; $("eyephotos").hidden = true;
  const q = $("qualitybanner"); if (q) q.hidden = true;
  const sb = $("seenbanner"); if (sb) sb.hidden = true;
  syncAnalyze();          // keeps the chosen context, re-blocks on the missing image
};

// ── whole-eye mode: many photographs of ONE eye, one worst-view answer ──────────────
let eyeFiles = [], eyeSummary = null;
const lat = () => (document.querySelector('input[name="laterality"]:checked') || {}).value;
function syncEye() {
  const b = $("analyze-eye"); if (!b) return;
  b.disabled = !(eyeFiles.length && eyeFiles.length <= 16 && lat() && context);
  b.title = b.disabled ? "Choose a context, OD/OS and 1-16 photographs" : "";
}
if ($("eyefiles")) {
  $("eyefiles").onchange = (e) => { eyeFiles = [...e.target.files]; eyeSummary = null; syncEye(); };
  document.querySelectorAll('input[name="laterality"]').forEach(r => r.onchange = syncEye);
  document.addEventListener("change", e => { if (e.target.name === "context") syncEye(); });
}
const esc = (t) => String(t).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function renderEye(res) {
  eyeSummary = { eye: res.eye, laterality: res.laterality, warnings: res.warnings };
  const e = res.eye, ce = res.comparison_eye;
  const cmpLine = ce
    ? `<div class="compare-row">
         <div>Our model (eye)<b>${esc(e.stage)}</b></div>
         <div>Comparison model (eye) ${agreeBadge(ce.agreement)}<b>${esc(ce.stage)}</b></div>
       </div><div class="rec">${esc((ce.agreement || {}).message || "")}
         Comparison driven by photo ${ce.driving_photo + 1}.</div>`
    : "";
  $("eyecard").innerHTML = `<div class="card staging eye-card">
    <h3>Eye summary (${esc(res.laterality)}): ${esc(e.stage)}</h3>
    <div class="rec">Worst view across ${res.per_photo.length} photograph(s) &mdash; driven by
      <b>photo ${e.driving_photo + 1}</b>. Research preview, not a clinical grade.</div>
    ${cmpLine}
    ${res.warnings.map(w => `<p class="muted small">&#9888; ${esc(w)}</p>`).join("")}</div>`;
  $("eyecard").hidden = false;
  $("eyephotos").innerHTML = res.per_photo.map((p, i) => {
    const st = p.staging ? esc(p.staging.prediction) : "not graded";
    const cmp = p.comparison ? ` · comparison: ${esc(p.comparison.prediction)} ${agreeBadge(p.comparison.agreement)}` : "";
    const seen = p.seen_in_data ? `<p class="muted small">&#9888; ${esc(p.seen_in_data.message)}</p>` : "";
    const drv = i === e.driving_photo;
    return `<div class="card${drv ? " staging driving" : ""}">
      <h3>Photo ${i + 1}${drv ? " &mdash; drives the eye stage" : ""}: ${st}${cmp}</h3>${seen}</div>`;
  }).join("");
  $("eyephotos").hidden = false;
  // the page's single-image machinery shows the driving photo's own result and heatmaps
  file = eyeFiles[e.driving_photo];
  preview.src = URL.createObjectURL(file);
  render(res.per_photo[e.driving_photo]);
}

if ($("analyze-eye")) $("analyze-eye").onclick = async () => {
  busy(true, `Analysing ${eyeFiles.length} photograph(s) — both models run on each one…`);
  $("results").hidden = true;
  const fd = new FormData();
  eyeFiles.forEach(f => fd.append("images", f));
  fd.append("eye", lat()); fd.append("context", context);
  let res;
  try {
    const r = await fetch("/predict_eye", { method: "POST", body: fd });
    res = await readJson(r);
    if (!r.ok) throw new Error(res.detail || res.error || "Analysis failed");
  } catch (e) {
    busy(false);
    alert(e.message || "Analysis failed."); return;
  }
  renderEye(res);
  busy(false);
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
};

// Non-JSON bodies (a proxy's HTML 413, say) must surface as a message, not a parse error.
async function readJson(r) {
  const t = await r.text();
  try { return JSON.parse(t); }
  catch { return { error: r.ok ? "Unexpected server response" : `Server error ${r.status}` }; }
}
