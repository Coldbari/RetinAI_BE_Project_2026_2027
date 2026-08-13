// Interactive flip-book — vanilla port of the framer-motion <InteractiveBook/> component.
// State: closed → open (cover flips left, spread shifts right) → page-by-page flipping.
(function () {
  const stage = document.getElementById("journeyBook");
  if (!stage) return;
  const book = stage.querySelector(".ibook");
  const cover = stage.querySelector(".ibook-cover");
  const pages = Array.from(stage.querySelectorAll(".ibook-page"));
  const closeBtn = stage.querySelector(".ibook-close");
  const hint = stage.querySelector(".ibook-hint");
  const N = pages.length;

  let isOpen = false;
  let idx = -1;                 // currentPageIndex: pages 0..idx are flipped

  // Resting z-index for an unflipped page (top of the stack first).
  pages.forEach((p, i) => { p.style.zIndex = String(N - i); });

  function updatePages() {
    pages.forEach((p, i) => {
      const flipped = i <= idx;
      p.classList.toggle("is-flipped", flipped);
      p.style.zIndex = String(flipped ? i + 1 : N - i);
    });
  }

  function open() {
    if (isOpen) return;
    isOpen = true;
    stage.classList.add("is-open");
  }

  function close() {
    isOpen = false;
    idx = -1;
    stage.classList.remove("is-open");
    updatePages();
  }

  function next() {
    if (!isOpen) return;
    if (idx < N - 1) { idx += 1; updatePages(); }
  }

  function prev() {
    if (!isOpen) return;
    if (idx >= 0) { idx -= 1; updatePages(); }
  }

  function restart() { idx = -1; updatePages(); }

  // --- click targets (mirror the React handlers) ---
  cover.querySelector(".ibook-cover-front").addEventListener("click", (e) => {
    e.stopPropagation();
    open();
  });
  cover.querySelector(".ibook-cover-back").addEventListener("click", (e) => {
    e.stopPropagation();
    prev();
  });
  if (hint) hint.addEventListener("click", open);
  if (closeBtn) closeBtn.addEventListener("click", (e) => { e.stopPropagation(); close(); });

  pages.forEach((p) => {
    const front = p.querySelector(".ibook-page-front");
    const back = p.querySelector(".ibook-page-back");
    if (front) front.addEventListener("click", (e) => { e.stopPropagation(); next(); });
    if (back) back.addEventListener("click", (e) => { e.stopPropagation(); prev(); });
  });

  const again = stage.querySelector(".ibook-again");
  if (again) again.addEventListener("click", (e) => { e.stopPropagation(); restart(); });

  // --- keyboard navigation (only while open) ---
  window.addEventListener("keydown", (e) => {
    if (!isOpen) return;
    if (e.key === "ArrowRight") next();
    else if (e.key === "ArrowLeft") prev();
    else if (e.key === "Escape") close();
  });

  updatePages();
})();
