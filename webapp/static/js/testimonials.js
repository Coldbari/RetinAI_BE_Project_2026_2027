// 3D card-stack testimonials carousel — vanilla port of the framer-motion component.
(function () {
  const root = document.getElementById("tcard");
  if (!root) return;
  const stack = document.getElementById("tstack");
  const slides = Array.from(stack.querySelectorAll(".tslide"));
  const N = slides.length;
  if (!N) return;

  const ttext = document.getElementById("ttext");
  const elQuote = document.getElementById("tquote");
  const elName = document.getElementById("tname");
  const elRole = document.getElementById("trole");
  const elAvatar = document.getElementById("tavatar");
  const elCur = document.getElementById("tcur");
  const prevBtn = document.getElementById("tprev");
  const nextBtn = document.getElementById("tnext");

  const quotes = JSON.parse(ttext.dataset.quotes || "[]");
  const names = JSON.parse(ttext.dataset.names || "[]");
  const roles = JSON.parse(ttext.dataset.roles || "[]");
  const avatars = JSON.parse(ttext.dataset.avatars || "[]");

  const ROT = [4, -2, -9, 7];           // per-card resting tilt (matches the component)
  let active = 0;
  let dir = 1;
  let timer = null;
  const AUTOPLAY = 4200;

  // Position every non-active card in the fanned stack; the active one is animated by CSS.
  function layout() {
    slides.forEach((el, i) => {
      const off = i - active;
      const a = Math.abs(off);
      if (i === active) {
        el.style.zIndex = 100;
        el.style.opacity = 1;
        // active transform is the resting end-state; tpop animation plays over it
        el.style.transform = "translateX(0) translateZ(250px) scale(1) rotateZ(0deg)";
      } else {
        el.style.zIndex = 10 - a;
        el.style.opacity = (0.55 - a * 0.12).toFixed(2);
        el.style.transform =
          `translateX(${off * 16}px) translateY(${a * 7}px) ` +
          `translateZ(${-150 * a}px) scale(${(0.85 - a * 0.04).toFixed(3)}) ` +
          `rotateZ(${ROT[i % 4]}deg)`;
      }
    });
  }

  function setText(i) {
    elQuote.textContent = "“" + (quotes[i] || "") + "”";
    elName.textContent = names[i] || "";
    elRole.textContent = roles[i] || "";
    if (elAvatar) { elAvatar.src = avatars[i] || ""; elAvatar.alt = names[i] || ""; }
    elCur.textContent = i + 1;
  }

  function render(animate) {
    const el = slides[active];
    if (animate) {
      slides.forEach((s) => s.classList.remove("is-active"));
      el.style.setProperty("--dir", dir);
      void el.offsetWidth;            // restart the CSS animation
      el.classList.add("is-active");
      // crossfade the text
      ttext.classList.add("swap");
      setTimeout(() => { setText(active); ttext.classList.remove("swap"); }, 200);
    } else {
      setText(active);
    }
    layout();
    prevBtn.disabled = active === 0;
    nextBtn.disabled = active === N - 1;
  }

  function go(i, d) {
    if (i < 0 || i > N - 1 || i === active) return;
    dir = d;
    active = i;
    render(true);
  }
  const next = () => go(active + 1, 1);
  const prev = () => go(active - 1, -1);

  nextBtn.addEventListener("click", () => { stop(); next(); start(); });
  prevBtn.addEventListener("click", () => { stop(); prev(); start(); });

  // autoplay loops with wrap-around (manual arrows stay bounded, like the component)
  function start() {
    if (N <= 1) return;
    stop();
    timer = setInterval(() => { dir = 1; active = (active + 1) % N; render(true); }, AUTOPLAY);
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  root.addEventListener("mouseenter", stop);
  root.addEventListener("mouseleave", start);

  render(false);
  start();
})();
