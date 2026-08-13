// CursorCard — hover a tagged term to show an image preview that spring-follows the
// cursor. Vanilla port of the framer-motion component (createPortal + useSpring).
(function () {
  const triggers = document.querySelectorAll("[data-cc-img]");
  if (!triggers.length) return;

  // one shared portal element on <body>
  const pop = document.createElement("div");
  pop.className = "cc-pop";
  pop.innerHTML = '<div class="cc-pop-inner"><img alt="preview"><p></p></div>';
  document.body.appendChild(pop);
  const popImg = pop.querySelector("img");
  const popTxt = pop.querySelector("p");

  let tx = 0, ty = 0, cx = 0, cy = 0;   // target vs current (springed) position
  let active = false, raf = null;

  function loop() {
    cx += (tx - cx) * 0.18;             // spring follow (~damping 25 / stiffness 300)
    cy += (ty - cy) * 0.18;
    pop.style.transform = `translate(${cx.toFixed(1)}px, ${cy.toFixed(1)}px)`;
    if (active || Math.abs(tx - cx) > 0.5 || Math.abs(ty - cy) > 0.5) {
      raf = requestAnimationFrame(loop);
    } else { raf = null; }
  }
  function kick() { if (!raf) raf = requestAnimationFrame(loop); }

  function move(e) {
    tx = e.clientX - 120;               // centre the 240px card horizontally
    ty = e.clientY + 20;                // sit just below the cursor
    kick();
  }

  triggers.forEach((el) => {
    el.addEventListener("mouseenter", (e) => {
      popImg.src = el.getAttribute("data-cc-img");
      popTxt.textContent = el.getAttribute("data-cc-desc") || "";
      // jump near the cursor on first show so it doesn't fly in from the corner
      tx = cx = e.clientX - 120; ty = cy = e.clientY + 20;
      active = true;
      pop.classList.add("show");
      kick();
    });
    el.addEventListener("mousemove", move);
    el.addEventListener("mouseleave", () => {
      active = false;
      pop.classList.remove("show");
    });
  });
})();
