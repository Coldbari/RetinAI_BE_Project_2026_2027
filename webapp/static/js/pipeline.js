// "How it works" cinematic pipeline — drives one fundus image through 4 phases.
// Phase state lives in stage.dataset.phase (1→4); all visual work is CSS keyed to it.
// Auto-plays only while on-screen, pauses off-screen, and falls back to a static
// final frame under prefers-reduced-motion.
(function () {
  const stage = document.getElementById("pipeline");
  if (!stage) return;

  const caption = document.getElementById("plCaption");
  const replay = document.getElementById("plReplay");
  const nodes = Array.from(stage.querySelectorAll(".pl-node"));

  // captions + dwell time (ms) per phase — slow and deliberate, easy to tune
  const CAPS = [
    "Watch one fundus image flow from capture to report.",
    "Acquiring the fundus image — 640 × 480 px.",
    "Circle-crop to the retina, then CLAHE contrast equalization.",
    "Three dedicated models score the image with test-time augmentation.",
    "Grad-CAM highlights the evidence · risk is graded · a PDF report is written.",
  ];
  const DUR = [0, 3600, 3900, 3600, 4900];

  const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';

  let phase = 0, timer = null, playing = false;

  function setPhase(p) {
    phase = p;
    stage.dataset.phase = String(p);
    if (caption) {
      caption.classList.add("swap");
      setTimeout(() => {
        caption.textContent = CAPS[p] || CAPS[0];
        caption.classList.remove("swap");
      }, 200);
    }
    nodes.forEach((n, i) => {
      const idx = i + 1;
      n.classList.toggle("is-active", idx === p);
      n.classList.toggle("is-done", idx < p);
    });
  }

  function advance() {
    const next = phase >= 4 ? 1 : phase + 1;
    setPhase(next);
    timer = setTimeout(advance, DUR[next]);
  }

  function play() {
    if (playing) return;
    playing = true;
    if (replay) replay.innerHTML = ICON_PAUSE;
    if (phase === 0) setPhase(1);
    timer = setTimeout(advance, DUR[phase] || DUR[1]);
  }

  function pause() {
    playing = false;
    clearTimeout(timer);
    timer = null;
    if (replay) replay.innerHTML = ICON_PLAY;
  }

  // Reduced motion: show the finished pipeline statically and stop.
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    setPhase(4);
    if (replay) replay.style.display = "none";
    return;
  }

  if (replay) replay.addEventListener("click", () => (playing ? pause() : play()));

  // click a station to jump straight to it
  nodes.forEach((n) => n.addEventListener("click", () => {
    const go = parseInt(n.dataset.go, 10);
    if (!go) return;
    clearTimeout(timer);
    setPhase(go);
    if (playing) timer = setTimeout(advance, DUR[go]);
  }));

  // autoplay only while the pipeline is in view
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? play() : pause()));
    }, { threshold: 0.35 });
    io.observe(stage);
  } else {
    play();
  }
})();
