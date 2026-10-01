/* =============================================================================
   ChurnIntel — motion choreography
   IntersectionObserver untuk reveal (tanpa scroll listener),
   counter animation, nav overlay dengan staggered mask reveal.
   ============================================================================= */

(() => {
  "use strict";

  /* ---------------------------------------------------------------------------
   1. SCROLL REVEAL — IntersectionObserver (GPU-safe, transform/opacity only)
  --------------------------------------------------------------------------- */
  const revealEls = document.querySelectorAll(
    ".reveal, .metric-card, .imp-row, .rate-row, .balance-row"
  );
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: "0px 0px -8% 0px" }
  );
  revealEls.forEach((el) => revealObserver.observe(el));

  /* ---------------------------------------------------------------------------
   2. NAV OVERLAY — hamburger morph + staggered mask reveal
  --------------------------------------------------------------------------- */
  const burger = document.getElementById("burger");
  const overlay = document.getElementById("menuOverlay");

  const setMenu = (open) => {
    burger.classList.toggle("is-open", open);
    overlay.classList.toggle("is-open", open);
    burger.setAttribute("aria-expanded", String(open));
    overlay.setAttribute("aria-hidden", String(!open));
    document.body.style.overflow = open ? "hidden" : "";
  };

  burger.addEventListener("click", () =>
    setMenu(!overlay.classList.contains("is-open"))
  );
  overlay.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => setMenu(false))
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setMenu(false);
  });

  /* ---------------------------------------------------------------------------
   3. COUNTER ANIMATION — angka statistik hero & metrik evaluasi
  --------------------------------------------------------------------------- */
  const animateCount = (el) => {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const duration = 1400;
    const start = performance.now();

    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      // ease-out kare — terasa seperti massa yang melambat
      const eased = 1 - Math.pow(1 - t, 4);
      const value = target * eased;
      el.textContent =
        value.toLocaleString("id-ID", {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }) + suffix;
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCount(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.4 }
  );
  document
    .querySelectorAll("[data-count]")
    .forEach((el) => counterObserver.observe(el));
})();
