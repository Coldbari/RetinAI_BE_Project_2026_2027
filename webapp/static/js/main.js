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
    const bars = Object.entries(f.scores).map(([k, v]) =>
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

  // ICROP staging research preview — rendered as its own card, visually subordinate to
  // the product result and explicitly labelled. Never confuse it with a clinical grade.
  if (res.staging) {
    const sbars = Object.entries(res.staging.scores).map(([k, v]) =>
      `<div class="scoreline"><span>${k}</span><span>${v}%</span></div>
       <div class="bar"><span style="width:${v}%"></span></div>`).join("");
    $("cards").innerHTML += `<div class="card">
      <h3>ICROP staging: ${res.staging.prediction} <span class="risk r-advisory">research preview</span></h3>
      <p class="muted small">${res.staging.note}</p>${sbars}</div>`;
  }

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

function selectViz(v) {
  document.querySelectorAll("#viztabs .tab").forEach(t => t.classList.toggle("on", t.dataset.v === v));
  const isHeat = v.startsWith("heat:");
  $("origimg").hidden = isHeat;
  $("heatimg").hidden = !isHeat;
  if (isHeat) {
    const h = heatmaps[Number(v.split(":")[1])];
    $("heatimg").src = "data:image/png;base64," + h.image;
    $("heatcap").textContent = "Grad-CAM — " + h.disease;
  } else {
    $("heatcap").textContent = "";
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
