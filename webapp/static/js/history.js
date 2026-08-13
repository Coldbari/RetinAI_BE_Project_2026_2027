// History page — per-row actions: View (modal), Download report (PDF), Delete.
const $h = (id) => document.getElementById(id);
const modal = $h("hist-modal");

function dlReport(id) {
  // GET endpoint returns the PDF as an attachment → browser downloads, page stays put
  window.location = `/history/report/${id}`;
}

function viewItem(id) {
  fetch(`/history/item/${id}`)
    .then(r => r.json())
    .then(d => {
      if (d.error) { alert("This screening is no longer available."); return; }
      $h("hm-time").textContent = d.time;
      $h("hm-img").src = "data:image/jpeg;base64," + d.image;
      $h("hm-findings").innerHTML = d.findings.map(f => {
        const risk = f.risk ? `<span class="risk r-${f.risk.toLowerCase()}">${f.risk}</span>` : "";
        return `<div class="hm-find"><span>${f.disease}: <b>${f.prediction}</b> (${f.score}%)</span>${risk}</div>`;
      }).join("");
      $h("hm-dl").onclick = () => dlReport(id);
      modal.hidden = false;
    })
    .catch(() => alert("Could not load this screening."));
}

function delRow(id) {
  if (!confirm("Delete this screening from history? This cannot be undone.")) return;
  fetch(`/history/delete/${id}`, { method: "POST" })
    .then(r => r.json())
    .then(d => {
      if (!d.ok) { alert("Could not delete this screening."); return; }
      const row = document.querySelector(`.hist-row[data-id="${id}"]`);
      if (row) row.remove();
      if (!document.querySelectorAll(".hist-row").length) {
        const list = document.querySelector(".hist-list");
        if (list) list.outerHTML =
          '<div class="panel"><p class="muted">No screenings yet. Run one from the Screen page.</p></div>';
      }
    })
    .catch(() => alert("Could not delete this screening."));
}

document.querySelectorAll(".hist-actions .view").forEach(b => b.onclick = () => viewItem(b.dataset.id));
document.querySelectorAll(".hist-actions .dl").forEach(b => b.onclick = () => dlReport(b.dataset.id));
document.querySelectorAll(".hist-actions .del").forEach(b => b.onclick = () => delRow(b.dataset.id));

// close the modal on backdrop click, ✕, or Escape
if (modal) {
  modal.addEventListener("click", e => {
    if (e.target === modal || e.target.classList.contains("hist-modal-x")) modal.hidden = true;
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") modal.hidden = true; });
}
