// A draggable 3D particle sphere: 1,200 dots, one per ~211 survey respondents.
// Coral dots are the 13.9% with diabetes or prediabetes. Particles fly in from
// random positions and assemble into the sphere on load.
import { reduced } from "./fx.js";

export function particleSphere(canvas, { n = 1200, positiveRate = 0.139 } = {}) {
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, dpr = Math.min(devicePixelRatio || 1, 2);
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const th = golden * i;
    pts.push({
      x: Math.cos(th) * r, y, z: Math.sin(th) * r,
      pos: rand() < positiveRate,
      sx: (rand() - 0.5) * 6, sy: (rand() - 0.5) * 6, sz: (rand() - 0.5) * 6,
      delay: rand() * 0.6,
    });
  }

  let rotX = -0.35, rotY = 0, velX = 0, velY = 0.0035;
  let dragging = false, lastX = 0, lastY = 0;
  let tiltX = 0, tiltY = 0, targetTX = 0, targetTY = 0;
  let assemble = reduced ? 1 : 0;
  const t0 = performance.now();
  let visible = true, scrollBoost = 0;

  const resize = () => {
    const r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    targetTX = ((e.clientY - r.top) / r.height - 0.5) * 0.4;
    targetTY = ((e.clientX - r.left) / r.width - 0.5) * 0.4;
    if (!dragging) return;
    velY = (e.clientX - lastX) * 0.005;
    velX = (e.clientY - lastY) * 0.005;
    lastX = e.clientX; lastY = e.clientY;
  });
  const end = () => { dragging = false; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => { targetTX = targetTY = 0; });
  let lastScroll = scrollY;
  addEventListener("scroll", () => { scrollBoost += Math.abs(scrollY - lastScroll) * 0.00012; lastScroll = scrollY; }, { passive: true });
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(canvas);

  const draw = (now) => {
    requestAnimationFrame(draw);
    if (!visible) return;
    const t = (now - t0) / 1000;
    if (!reduced) assemble = Math.min(1, (now - t0 - 700) / 2200);
    if (!dragging) {
      velY += (0.0035 - velY) * 0.02;
      velX += (0 - velX) * 0.05;
    }
    scrollBoost *= 0.92;
    rotY += reduced ? 0.001 : velY + scrollBoost;
    rotX += reduced ? 0 : velX;
    tiltX += (targetTX - tiltX) * 0.06;
    tiltY += (targetTY - tiltY) * 0.06;

    ctx.clearRect(0, 0, W, H);
    const R = Math.min(W, H) * 0.38;
    const cx = W / 2, cy = H / 2;
    const ax = rotX + tiltX, ay = rotY + tiltY;
    const cX = Math.cos(ax), sX = Math.sin(ax), cY = Math.cos(ay), sY = Math.sin(ay);

    // glow core
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.3);
    g.addColorStop(0, "rgba(45,226,196,0.10)");
    g.addColorStop(0.6, "rgba(139,124,240,0.05)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // orbit rings
    ctx.save();
    ctx.translate(cx, cy);
    for (let k = 0; k < 2; k++) {
      ctx.rotate(k === 0 ? t * 0.05 : -t * 0.035);
      ctx.beginPath();
      ctx.ellipse(0, 0, R * (1.28 + k * 0.14), R * (0.34 + k * 0.08), 0.4 + k, 0, Math.PI * 2);
      ctx.strokeStyle = k === 0 ? "rgba(45,226,196,0.22)" : "rgba(139,124,240,0.18)";
      ctx.setLineDash([2, 7]);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();

    const wave = (t * 0.6) % 3 - 1.5; // a bright band sweeping down the sphere
    const proj = [];
    for (const p of pts) {
      const a = reduced ? 1 : ease(Math.max(0, Math.min(1, (assemble - p.delay * 0.5) / 0.6)));
      const px = p.sx + (p.x - p.sx) * a, py = p.sy + (p.y - p.sy) * a, pz = p.sz + (p.z - p.sz) * a;
      let x = px * cY - pz * sY;
      let z = px * sY + pz * cY;
      let y = py * cX - z * sX;
      z = py * sX + z * cX;
      if (2.6 + z < 0.4) continue; // still flying in from behind the camera
      const persp = Math.min(3, 2.6 / (2.6 + z));
      proj.push({ X: cx + x * R * persp, Y: cy + y * R * persp, z, pos: p.pos, persp, band: Math.exp(-((p.y - wave) ** 2) * 18) });
    }
    proj.sort((a, b) => b.z - a.z);
    for (const q of proj) {
      const depth = Math.max(0, Math.min(1, (1 - q.z) / 2)); // 0 back .. 1 front
      const alpha = 0.18 + depth * 0.82;
      const size = (q.pos ? 2.3 : 1.55) * q.persp * (1 + q.band * 0.8);
      if (q.pos) {
        ctx.fillStyle = `rgba(255,122,69,${alpha})`;
        if (depth > 0.55) { ctx.shadowColor = "rgba(255,122,69,.9)"; ctx.shadowBlur = 10 * depth; }
      } else {
        const b = Math.min(1, alpha + q.band * 0.5);
        ctx.fillStyle = `rgba(${45 + q.band * 120},${226},${196 + q.band * 40},${b * 0.85})`;
        ctx.shadowBlur = 0;
      }
      ctx.beginPath();
      ctx.arc(q.X, q.Y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  };
  requestAnimationFrame(draw);
}

function ease(t) { return t === 1 ? 1 : 1 - Math.pow(2, -10 * t); }
