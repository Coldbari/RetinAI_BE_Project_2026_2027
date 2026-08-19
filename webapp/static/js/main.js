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
  file = f;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false; dropmsg.hidden = true;
  syncAnalyze();
}

const MIN_LOADER_MS = 5000;   // keep the full-screen loader on screen for at least 5s

$("analyze").onclick = async () => {
  if (!file) return;
  const overlay = $("loader-overlay");
  overlay.classList.add("show");                 // open the full-screen black loading page
  document.body.style.overflow = "hidden";
  $("results").hidden = true;

  const fd = new FormData(); fd.append("image", file); fd.append("context", context);
  const started = Date.now();
  let res;
  try {
    const r = await fetch("/predict", { method: "POST", body: fd });
    res = await r.json();
    if (!r.ok) throw new Error(res.detail || res.error || "Analysis failed");
  } catch (e) {
    overlay.classList.remove("show"); document.body.style.overflow = "";
    alert(e.message || "Analysis failed."); return;
  }

  // hold the loading animation for a minimum of 5 seconds, even if analysis finished sooner
  const elapsed = Date.now() - started;
  if (elapsed < MIN_LOADER_MS) await wait(MIN_LOADER_MS - elapsed);

  render(res);
  overlay.classList.remove("show");
  document.body.style.overflow = "";
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
};

// What a positive is actually worth. Buying 99% sensitivity at this threshold costs
// specificity, and the price is measured: most healthy eyes are flagged too. A verdict
// that reads "ROP Detected" without that number beside it will be read as a diagnosis,
// which is how a correctly-behaving screen looks like a broken one on a healthy baby.
// ONE short warning, not two long ones. Field feedback on the first version: "the two box
// section, you have shown that doesn't know the meaning as well" — a caveat nobody finishes
// reading protects nobody. Both measurements still appear; they are just one sentence each.
function falseAlarmNote(d) {
  const m = d.measured, e = d.external;
  if (!m || !d.positive) return "";
  // Both measurements, one sentence each. The external one is always the worse of the two
  // and is the number a user in a different NICU actually needs.
  const ext = e ? ` At a hospital it never trained on that rises to
    <b>${e.false_alarm}%</b> (AUC ${e.auc}).` : "";
  return `<p class="dm-fa hard"><b>Do not read this as a diagnosis.</b>
    This cut-off flags <b>${m.false_alarm}% of healthy eyes</b> at its own hospitals.${ext}
    A positive means <b>"have a clinician look"</b> — nothing more.</p>`;
}

// A screening score drawn against its OWN decision line, not against an implied 50%.
// The line is the operating point the model actually ships with; the distance from it is
// the only thing the number means.
function decisionMeter(f) {
  const d = f.decision, s = d.score, t = d.threshold;
  const side = d.positive
    ? `<b class="dm-pos">above</b> the decision line — screened positive`
    : `<b class="dm-neg">below</b> the decision line — screened negative`;
  const ratio = t > 0 ? (s / t).toFixed(1) : null;
  const raw = Object.entries(f.scores).map(([k, v]) => `${k} ${v}%`).join(" · ");
  return `
    <div class="dm">
      <div class="scoreline"><span>Model score</span><span><b>${s}%</b></span></div>
      <div class="dm-track">
        <span class="dm-fill${d.positive ? " on" : ""}" style="width:${Math.min(s, 100)}%"></span>
        <i class="dm-tick" style="left:${Math.min(t, 100)}%"></i>
      </div>
      <div class="dm-axis"><span>0%</span>
        <span class="dm-thr" style="left:${Math.min(t, 100)}%">▲ flagged above ${t}%</span>
        <span>100%</span></div>
      <p class="dm-why">Anything scoring above <b>${t}%</b> gets flagged, so this one is
         ${side}${ratio ? ` (${ratio}× the cut-off)` : ""}. The cut-off is set that low on
         purpose — the model is built to rarely miss disease, which means it flags many
         healthy eyes too.</p>
      ${falseAlarmNote(d)}
      <p class="dm-raw">${raw} — the second number is just <i>100 − score</i>, not a
         separate finding of health.</p>
    </div>`;
}

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
      <h3>ICROP stage: ${st.prediction}
        <span class="risk r-preview">research preview</span></h3>
      <div class="rec">Six-class ICROP-3 staging. This is <b>not</b> the screening decision
        above and must not be read as one.</div>
      ${any}
      ${stagingVsScreening(st, rop)}
      ${bars}
      <p class="dm-raw">${st.note || ""}</p>
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
    // healthy" when it is only 1 - score. A binary finding gets a threshold-anchored meter
    // instead; the raw class numbers stay, demoted to what they are.
    const bars = f.decision && f.decision.positive_class
      ? decisionMeter(f)
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
  renderStaging(ungradable ? null : res.staging,
                res.findings.find(f => f.disease === "ROP" && f.available));

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
    rows.push(`<li class="${bad ? "warn" : "ok"}"><b>${e.off_retina_pct}%</b> of the
      attention fell <b>outside the retina</b> (on the black surround).
      ${bad ? "That is high — on this image the score is partly driven by the frame, "
            + "not the eye. Treat the map as unreliable."
            : "Low: the model is reading retina, not frame."}</li>`);
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
      + " · coloured inside the retinal field only; the white outline is its edge.";
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
  const q = $("qualitybanner"); if (q) q.hidden = true;
  syncAnalyze();          // keeps the chosen context, re-blocks on the missing image
};

const wait = (ms) => new Promise(r => setTimeout(r, ms));
