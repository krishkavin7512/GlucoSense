import { reduced, fine, whenVisible, tween, fmt, countTo, initReveals, splitWords, magnetic, tilt, spotlight, cursor, marquee,
  scramble, slidingPill, setRangeFill, debounce, toast } from "./fx.js";
import { particleSphere } from "./sphere.js";
import { C, layout, axis, render, CONFIG } from "./charts.js";

// ---------------------------------------------------------------- capture mode (for report screenshots)
// ?capture=<section id|hero|footer>&set=<selector>=<value>&click=<selector>|<selector>
const Q = new URLSearchParams(location.search);
const CAPTURE = Q.get("capture");
if (CAPTURE) {
  document.documentElement.classList.add("capture");
  const note = (m) => { document.documentElement.dataset.err = (document.documentElement.dataset.err || "") + " | " + m; };
  addEventListener("error", (e) => note(e.message));
  addEventListener("unhandledrejection", (e) => note(String(e.reason && (e.reason.stack || e.reason))));
  try { sessionStorage.setItem("gs-intro", "1"); } catch {}
  const t = CAPTURE === "hero" ? document.querySelector(".hero") : CAPTURE === "footer" ? document.querySelector("footer") : document.getElementById(CAPTURE);
  t?.classList.add("cap-target");
}
async function runCaptureActions() {
  if (!CAPTURE) return;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  await wait(700);
  for (const s of Q.getAll("set")) {
    const i = s.lastIndexOf("=");
    const el = document.querySelector(s.slice(0, i));
    if (el) { el.value = s.slice(i + 1); el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); await wait(500); }
  }
  for (const sel of (Q.get("click") || "").split("|").filter(Boolean)) { document.querySelector(sel)?.click(); await wait(900); }
  document.documentElement.classList.add("capture-ready");
}

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (path, opts) => fetch(path, opts).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); });

// ---------------------------------------------------------------- boot
const data = Promise.all([
  api("/api/overview"), api("/api/features"), api("/api/stats/probability"), api("/api/stats/bmi"), api("/api/data/summary"),
]);

runLoader(data).then(() => {
  document.body.classList.remove("is-loading");
  $("#heroTitle").classList.add("in");
});

splitWords($("#heroTitle"), 250);
$$("[data-split]").forEach((h) => splitWords(h));
$$(".split em .w > span, .split .grad .w > span").forEach((s) => s.classList.add("grad-text"));
cursor();
$$(".magnet, .nav__cta .btn").forEach((b) => magnetic(b));
$$(".card").forEach(spotlight);
initNav();
buildMarquee();
particleSphere($("#sphere"));

data.then(([overview, features, prob, bmi, summary]) => {
  initReveals();
  heroNumbers(overview);
  initData(summary);
  initBayes(prob);
  initDensity(bmi);
  initFit();
  initScreener(features, overview);
  initReport(overview);
  runCaptureActions();
}).catch((e) => { console.error(e); toast("Could not reach the backend. Is `python run.py` running?"); });

// ---------------------------------------------------------------- loader
async function runLoader(ready) {
  const loader = $("#loader");
  let seen = false;
  try { seen = sessionStorage.getItem("gs-intro") === "1"; } catch {}
  if (reduced || seen) { loader.remove(); await ready.catch(() => {}); return; }
  const word = $(".loader__word");
  "GLUCOSENSE".split("").forEach((ch, i) => {
    const s = document.createElement("span");
    s.textContent = ch;
    s.style.setProperty("--i", i);
    if (i >= 5) s.classList.add("accent");
    word.appendChild(s);
  });
  const bar = $(".loader__bar i"), pct = $("#loaderPct"), msg = $("#loaderMsg");
  const msgs = ["Loading 253,680 answers", "Counting likelihoods", "Calibrating probabilities", "Ready"];
  await tween(0, 1, 1900, (v) => {
    bar.style.transform = `scaleX(${v})`;
    pct.textContent = String(Math.round(v * 100)).padStart(3, "0");
    msg.textContent = msgs[Math.min(3, Math.floor(v * 3.2))];
  });
  await ready.catch(() => {});
  loader.classList.add("done");
  try { sessionStorage.setItem("gs-intro", "1"); } catch {}
  setTimeout(() => loader.remove(), 1200);
  await new Promise((r) => setTimeout(r, 350));
}

// ---------------------------------------------------------------- nav + chrome
function initNav() {
  const nav = $("#nav"), links = $("#navLinks"), pill = $(".nav__pill", links), progress = $("#progress");
  let lastY = scrollY;
  const movePill = () => {
    const a = $("a.active", links);
    if (!a) { pill.style.width = "0px"; return; }
    pill.style.width = a.offsetWidth + "px";
    pill.style.transform = `translateX(${a.offsetLeft}px)`;
  };
  addEventListener("scroll", () => {
    const y = scrollY;
    const max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav.classList.toggle("hidden", y > lastY && y > 400);
    lastY = y;
  }, { passive: true });
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      $$("a", links).forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
      movePill();
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("main section[id]").forEach((s) => io.observe(s));
  addEventListener("resize", movePill);
}

function buildMarquee() {
  const words = ["Bayes' rule", "Conditional independence", "Quantiles", "Expectation", "Covariance",
    "Polynomial curve fitting", "Probability densities", "Naive Bayes"];
  const track = $("#marquee");
  const html = words.map((w) => `<span>${w} <b>&#10022;</b></span>`).join("");
  track.innerHTML = html + html;
  marquee(track);
}

function heroNumbers(o) {
  $("#heroAuc").textContent = o.test.roc_auc.toFixed(3);
  $("#heroRecall").textContent = Math.round(o.test.recall * 100) + "%";
}

// ---------------------------------------------------------------- 01 data
function initData(summary) {
  const tabs = $("#dataTabs");
  const move = slidingPill(tabs, $(".tabs__pill", tabs));
  $$("button", tabs).forEach((b) => b.addEventListener("click", () => {
    $$("button", tabs).forEach((x) => x.classList.toggle("active", x === b));
    move();
    $("#tab-rows").hidden = b.dataset.tab !== "rows";
    $("#tab-summary").hidden = b.dataset.tab !== "summary";
  }));

  let offset = 0;
  const limit = 14;
  let total = 253680;
  const binaryCols = new Set(["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"]);
  const load = async () => {
    const p = await api(`/api/data/preview?offset=${offset}&limit=${limit}`);
    total = p.total;
    const t = document.createElement("table");
    t.className = "data";
    const h = t.createTHead().insertRow();
    ["#", ...p.columns].forEach((c) => { const th = document.createElement("th"); th.textContent = c; h.appendChild(th); });
    const b = t.createTBody();
    p.rows.forEach((row, i) => {
      const tr = b.insertRow();
      tr.style.setProperty("--r", i);
      tr.insertCell().textContent = (offset + i).toLocaleString();
      row.forEach((v, j) => {
        const td = tr.insertCell();
        const col = p.columns[j];
        if (col === "Diabetes_binary") td.innerHTML = v ? '<span class="pill-pos">yes</span>' : '<span class="pill-neg">no</span>';
        else if (binaryCols.has(col)) td.innerHTML = `<span class="bin ${v ? "on" : ""}" title="${v}"></span>`;
        else td.textContent = v;
      });
    });
    const wrap = $("#rowsTable");
    wrap.replaceChildren(t);
    $("#pagerInfo").textContent = `Rows ${(offset + 1).toLocaleString()}–${(offset + p.rows.length).toLocaleString()} of ${total.toLocaleString()}`;
  };
  $("#prevPage").onclick = () => { offset = Math.max(0, offset - limit); load(); };
  $("#nextPage").onclick = () => { offset = Math.min(total - limit, offset + limit); load(); };
  $("#randPage").onclick = () => { offset = Math.floor(Math.random() * (total - limit)); load(); };
  load();

  const t = document.createElement("table");
  t.className = "data";
  const cols = ["Column", "Kind", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skew", "Unique"];
  const h = t.createTHead().insertRow();
  cols.forEach((c) => { const th = document.createElement("th"); th.textContent = c; h.appendChild(th); });
  const b = t.createTBody();
  summary.summary.forEach((r, i) => {
    const tr = b.insertRow();
    tr.style.setProperty("--r", i);
    const name = tr.insertCell();
    name.innerHTML = `<b style="color:var(--ink)">${r.column}</b><br><span class="subtle">${r.label}</span>`;
    tr.insertCell().innerHTML = `<span class="kind">${r.kind}</span>`;
    [r.mean, r.std, r.min, r.q25, r.median, r.q75, r.max, r.skewness].forEach((v) => {
      tr.insertCell().textContent = Math.abs(v) >= 100 ? fmt(v, { decimals: 1 }) : v.toFixed(3);
    });
    tr.insertCell().textContent = r.unique;
  });
  $("#summaryTable").replaceChildren(t);
}

// ---------------------------------------------------------------- 02 Bayes
function initBayes(prob) {
  const chips = $("#eventChips");
  const events = prob.bayes;
  const prior = prob.base_rate;
  const mk = (grid) => { grid.innerHTML = Array.from({ length: 100 }, (_, k) => `<i style="--k:${spiral(k)}"></i>`).join(""); return $$("i", grid); };
  const priorDots = mk($("#iaPrior"));
  const postDots = mk($("#iaPost"));
  // Standard icon array: fill row by row from the top-left; the ripple is only the animation timing.
  const fill = (dots, n) => dots.forEach((d, idx) => d.classList.toggle("on", idx < n));
  setTimeout(() => fill(priorDots, Math.round(prior * 100)), 600);

  events.forEach((ev, i) => {
    const b = document.createElement("button");
    b.textContent = ev.label;
    b.addEventListener("click", () => select(i));
    chips.appendChild(b);
  });
  const select = (i) => {
    const ev = events[i];
    $$("button", chips).forEach((b, j) => b.classList.toggle("active", j === i));
    fill(postDots, Math.round(ev.posterior * 100));
    scramble($("#iaPostTitle"), `Posterior: 100 adults with "${ev.label.toLowerCase()}"`);
    countTo($("#priorPct"), prior, { pct: true, decimals: 1, duration: 600 });
    countTo($("#postPct"), ev.posterior, { pct: true, decimals: 1, duration: 1000 });
    countTo($("#kLR"), ev.likelihood_ratio, { decimals: 2, suffix: "×", duration: 900 });
    countTo($("#kLift"), ev.lift, { decimals: 2, suffix: "×", duration: 900 });
    countTo($("#kN"), ev.n_with_feature, { compact: true, duration: 900 });
    const p = (v) => (v * 100).toFixed(1) + "%";
    $("#bayesEq").innerHTML = `
      P(F) = P(F|D)·P(D) + P(F|¬D)·P(¬D)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <b>${p(ev.likelihood_d)}</b> × <b>${p(prior)}</b> + <b>${p(ev.likelihood_not_d)}</b> × <b>${p(1 - prior)}</b> = <span class="hl2">${p(ev.evidence)}</span><br>
      P(D|F) = <span class="frac"><span>P(F|D) · P(D)</span><span>P(F)</span></span> =
      <span class="frac"><span>${p(ev.likelihood_d)} × ${p(prior)}</span><span>${p(ev.evidence)}</span></span> = <span class="hl">${p(ev.posterior)}</span><br>
      <span class="subtle">Counted directly from people with the fact: ${p(ev.direct_posterior)} ✓</span>`;
  };
  select(0);

  // joint table
  const j = prob.joint_highbp;
  const J = j.joint, pa = j.p_a, pb = j.p_b;
  const cells = [
    ["", "No diabetes", "Diabetes", "P(BP)"],
    ["BP normal", J[0][0], J[0][1], pa[0]],
    ["High BP", J[1][0], J[1][1], pa[1]],
    ["P(D)", pb[0], pb[1], 1],
  ];
  const tbl = $("#jointTable");
  cells.forEach((row, r) => row.forEach((v, c) => {
    const d = document.createElement("div");
    if (r === 0 || c === 0) { d.className = "h"; d.textContent = v; }
    else if (r === 3 || c === 3) { d.className = "marg"; d.textContent = (v * 100).toFixed(1) + "%"; d.dataset.r = r; d.dataset.c = c; }
    else { d.className = "cell"; d.textContent = (v * 100).toFixed(1) + "%"; d.dataset.r = r; d.dataset.c = c; }
    tbl.appendChild(d);
  }));
  tbl.addEventListener("pointerover", (e) => {
    const t = e.target.closest(".cell");
    $$(".lit", tbl).forEach((x) => x.classList.remove("lit"));
    if (!t) return;
    $$(`[data-r="${t.dataset.r}"][data-c="3"], [data-c="${t.dataset.c}"][data-r="3"]`, tbl).forEach((x) => x.classList.add("lit"));
  });
}

function spiral(k) { // distance-from-centre ordering so dots light up in a ripple
  const r = Math.floor(k / 10), c = k % 10;
  return Math.round(Math.hypot(r - 4.5, c - 4.5) * 10);
}

// ---------------------------------------------------------------- 03 density
function initDensity(bmi) {
  const h = bmi.histogram;
  const mids = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
  const pdfAt = (cls, x) => {
    const g = h.grid, y = h[cls].lognormal_pdf;
    let i = g.findIndex((v) => v >= x);
    if (i <= 0) return y[0];
    const t = (x - g[i - 1]) / (g[i] - g[i - 1]);
    return y[i - 1] + t * (y[i] - y[i - 1]);
  };
  const q = (cls, p) => bmi.quantiles[cls][p];
  const lineFor = (cls, p) => ({ x: [q(cls, p), q(cls, p)], y: [0, pdfAt(cls, q(cls, p))] });
  const data = [
    { type: "bar", x: mids, y: h["0"].density, name: "No diabetes", marker: { color: C.teal, opacity: 0.3 }, hovertemplate: "BMI %{x}: %{y:.4f}<extra></extra>" },
    { type: "bar", x: mids, y: h["1"].density, name: "Diabetes / prediabetes", marker: { color: C.coral, opacity: 0.3 }, hovertemplate: "BMI %{x}: %{y:.4f}<extra></extra>" },
    { type: "scatter", mode: "lines", x: h.grid, y: h["0"].lognormal_pdf, name: "log-normal fit", line: { color: C.teal, width: 2.5 }, hovertemplate: "%{y:.4f}<extra>fit, no diabetes</extra>" },
    { type: "scatter", mode: "lines", x: h.grid, y: h["1"].lognormal_pdf, name: "log-normal fit", line: { color: C.coral, width: 2.5 }, showlegend: false, hovertemplate: "%{y:.4f}<extra>fit, diabetes</extra>" },
    { type: "scatter", mode: "lines+markers", ...lineFor("0", 50), line: { color: C.teal, width: 2, dash: "dot" }, marker: { size: [0, 10], color: C.teal }, showlegend: false, hoverinfo: "skip" },
    { type: "scatter", mode: "lines+markers", ...lineFor("1", 50), line: { color: C.coral, width: 2, dash: "dot" }, marker: { size: [0, 10], color: C.coral }, showlegend: false, hoverinfo: "skip" },
  ];
  const el = $("#densityPlot");
  const lay = layout({ barmode: "overlay", bargap: 0.02, xaxis: axis("Body-mass index (kg/m²)", { range: [12, 62] }), yaxis: axis("Probability density") });
  whenVisible(el, async () => {
    await render(el, { data, layout: lay });
    update(+$("#qSlider").value);
  });

  const slider = $("#qSlider");
  const update = (p) => {
    $("#qVal").textContent = p + "%";
    $("#q0").textContent = q("0", p).toFixed(1);
    $("#q1").textContent = q("1", p).toFixed(1);
    setRangeFill(slider);
    if (!el.data) return;
    const l0 = lineFor("0", p), l1 = lineFor("1", p);
    // restyle, not animate: Plotly v4 drops bar traces when a subset of traces is animated
    Plotly.restyle(el, { x: [l0.x, l1.x], y: [l0.y, l1.y] }, [4, 5]);
  };
  slider.addEventListener("input", () => update(+slider.value));
  setRangeFill(slider);
  $("#q0").textContent = q("0", 50).toFixed(1);
  $("#q1").textContent = q("1", 50).toFixed(1);

  const a = bmi.by_class["0"], b = bmi.by_class["1"];
  $("#moments").innerHTML = [["Mean", "mean", 1], ["Variance", "var", 1], ["Std dev", "std", 2], ["Skewness", "skewness", 2]]
    .map(([l, k, d]) => `<div class="kpi"><b class="tnum" style="font-size:18px"><span style="color:var(--teal)">${a[k].toFixed(d)}</span> · <span style="color:var(--coral)">${b[k].toFixed(d)}</span></b><span>${l}</span></div>`).join("");
  const te = bmi.total_expectation;
  $("#totalExp").innerHTML = `Law of total expectation<br>E[BMI] = P(¬D)·E[BMI|¬D] + P(D)·E[BMI|D]<br>
    = <b>${te.p[0].toFixed(3)}</b> × <b>${te.cond_mean[0].toFixed(2)}</b> + <b>${te.p[1].toFixed(3)}</b> × <b>${te.cond_mean[1].toFixed(2)}</b>
    = <span class="hl2">${te.e_total.toFixed(3)}</span><br><span class="subtle">Direct mean of all 253,680 BMIs: ${te.e_direct.toFixed(3)} ✓</span>`;
}

// ---------------------------------------------------------------- 04 curve fitting
function initFit() {
  const sizes = [["300", "300"], ["1000", "1k"], ["3000", "3k"], ["10000", "10k"], ["30000", "30k"], ["all", "177k"]];
  let size = "1000";
  const seg = $("#sizeSeg");
  sizes.forEach(([k, l]) => {
    const b = document.createElement("button");
    b.textContent = l;
    b.classList.toggle("active", k === size);
    b.onclick = () => { size = k; $$("button", seg).forEach((x) => x.classList.toggle("active", x === b)); refresh(true); };
    seg.appendChild(b);
  });
  const mS = $("#mSlider"), lS = $("#lSlider"), el = $("#fitPlot");
  let first = true, pending = null, busy = false;
  const lay = layout({ xaxis: axis("Body-mass index (kg/m²)", { range: [14, 61] }),
    yaxis: axis("Fraction with diabetes / prediabetes", { range: [-0.15, 0.75], tickformat: ".0%" }) });

  const draw = async (r, newPoints) => {
    const traces = [
      { type: "scatter", mode: "lines", x: r.test.x, y: r.test.t, name: "Test rate (38,052 people)", line: { color: C.muted, width: 2 }, hovertemplate: "BMI %{x}: %{y:.1%}<extra>test</extra>" },
      { type: "scatter", mode: "markers", x: r.train.x, y: r.train.t, name: "Training points", marker: { color: C.amber, size: 8, line: { color: C.surface, width: 2 } }, customdata: r.train.n, hovertemplate: "BMI %{x}: %{y:.1%} of %{customdata} people<extra>train</extra>" },
      { type: "scatter", mode: "lines", x: r.grid, y: r.curve, name: `Polynomial, M = ${r.degree}`, line: { color: C.coral, width: 3 }, hovertemplate: "BMI %{x:.1f}: %{y:.1%}<extra>fit</extra>" },
    ];
    if (first) {
      first = false;
      await render(el, { data: traces, layout: lay });
    } else if (newPoints) {
      await Plotly.react(el, traces, lay, CONFIG);
    } else {
      await Plotly.animate(el, { data: [{ y: r.curve, name: traces[2].name }], traces: [2] },
        { transition: { duration: 420, easing: "cubic-out" }, frame: { duration: 420, redraw: false } });
    }
  };

  const refresh = async (newPoints = false) => {
    if (busy) { pending = newPoints || pending || false; return; }
    busy = true;
    const M = +mS.value, L = +lS.value;
    $("#mVal").textContent = M;
    $("#lVal").textContent = L <= -21 ? "off" : L.toFixed(1);
    setRangeFill(mS); setRangeFill(lS);
    const q = new URLSearchParams({ degree: M, size });
    if (L > -21) q.set("log_lambda", L);
    try {
      const r = await api("/api/polyfit?" + q);
      await draw(r, newPoints);
      updateReadout(r);
    } finally {
      busy = false;
      if (pending !== null) { const p = pending; pending = null; refresh(p); }
    }
  };

  const updateReadout = (r) => {
    countTo($("#trainRms"), r.train_rms, { decimals: 4, duration: 500 });
    countTo($("#testRms"), r.test_rms, { decimals: 4, duration: 500 });
    const scale = (v) => Math.min(100, (Math.log10(v * 1000 + 1) / Math.log10(1000 * 2 + 1)) * 100);
    $("#trainBar").style.width = scale(r.train_rms) + "%";
    $("#testBar").style.width = scale(r.test_rms) + "%";
    $("#maxW").textContent = r.max_weight >= 1000 ? r.max_weight.toExponential(1) : r.max_weight.toFixed(2);
    const st = $("#fitStatus");
    const ratio = r.test_rms / Math.max(r.train_rms, 1e-6);
    let cls = "good", txt = "Good fit";
    if (r.degree <= 1 && r.test_rms > 0.05) { cls = "under"; txt = "Underfitting: too simple"; }
    if (r.test_rms > 0.09 || ratio > 1.6) { cls = "over"; txt = "Overfitting: chasing noise"; }
    st.className = "status " + cls;
    $("span", st).textContent = txt;
    const W = $("#weights");
    const n = r.weights.length;
    while (W.children.length < n) W.appendChild(document.createElement("i"));
    while (W.children.length > n) W.lastChild.remove();
    const mx = Math.log10(Math.max(...r.weights.map((w) => Math.abs(w))) + 1) || 1;
    [...W.children].forEach((bar, i) => {
      const w = r.weights[i];
      bar.style.height = Math.max(3, (Math.log10(Math.abs(w) + 1) / Math.max(mx, 1)) * 100) + "%";
      bar.classList.toggle("neg", w < 0);
      bar.title = `w${i} = ${w.toPrecision(4)}`;
    });
  };

  mS.addEventListener("input", () => refresh(false));
  lS.addEventListener("input", () => refresh(false));
  whenVisible(el, () => refresh(true));
}

// ---------------------------------------------------------------- 05 screener
function initScreener(meta, overview) {
  const feats = meta.features;
  const answers = { ...meta.defaults };
  const controls = {};
  const groups = ["Body", "Medical history", "Lifestyle", "About you"];
  const form = $("#form");
  const dayText = (v) => (v === 30 ? "every day" : v === 1 ? "1 day" : `${v} days`);

  groups.forEach((g, gi) => {
    const wrap = document.createElement("div");
    wrap.className = "qgroup" + (gi > 1 ? " collapsed" : "");
    wrap.innerHTML = `<div class="qgroup__head"><h4>${g}</h4><span class="tag">▾ ${feats.filter((f) => f.group === g).length} questions</span></div><div class="qgroup__body"></div>`;
    $(".qgroup__head", wrap).addEventListener("click", () => wrap.classList.toggle("collapsed"));
    const body = $(".qgroup__body", wrap);
    feats.filter((f) => f.group === g).forEach((f) => body.appendChild(buildControl(f)));
    form.appendChild(wrap);
  });

  function buildControl(f) {
    const q = document.createElement("div");
    q.className = "q";
    const label = document.createElement("div");
    label.className = "q__label";
    label.textContent = f.question;
    q.appendChild(label);
    if (f.kind === "binary") {
      const t = document.createElement("div");
      t.className = "toggle";
      t.setAttribute("role", "switch");
      t.tabIndex = 0;
      t.innerHTML = `<i class="knob"></i><span>${f.options[0]}</span><span>${f.options[1]}</span>`;
      const set = (v) => { t.classList.toggle("on", !!v); t.classList.toggle("off", !v); t.setAttribute("aria-checked", !!v); };
      const flip = () => { answers[f.key] = answers[f.key] ? 0 : 1; set(answers[f.key]); changed(); };
      t.addEventListener("click", flip);
      t.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
      set(answers[f.key]);
      controls[f.key] = { set };
      q.appendChild(t);
    } else if (f.key === "GenHlth") {
      q.classList.add("wide");
      const s = document.createElement("div");
      s.className = "seg";
      f.options.forEach((o, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = o;
        b.onclick = () => { answers[f.key] = f.levels[i]; set(answers[f.key]); changed(); };
        s.appendChild(b);
      });
      const set = (v) => $$("button", s).forEach((b, i) => b.classList.toggle("active", f.levels[i] === v));
      set(answers[f.key]);
      controls[f.key] = { set };
      q.appendChild(s);
    } else if (f.kind === "ordinal" && f.key !== "Age") {
      const s = document.createElement("select");
      s.className = "sel";
      s.setAttribute("aria-label", f.question);
      f.options.forEach((o, i) => { const op = document.createElement("option"); op.value = f.levels[i]; op.textContent = o; s.appendChild(op); });
      s.value = answers[f.key];
      s.onchange = () => { answers[f.key] = +s.value; changed(); };
      controls[f.key] = { set: (v) => { s.value = v; } };
      q.appendChild(s);
    } else {
      // sliders: Age (ordinal), BMI (continuous), days
      const isBMI = f.kind === "continuous", isAge = f.key === "Age";
      if (isBMI) q.classList.add("wide");
      const min = isBMI ? 12 : isAge ? 1 : 0, max = isBMI ? 60 : isAge ? 13 : 30;
      const val = document.createElement("div");
      val.className = "slider-val";
      const r = document.createElement("input");
      r.type = "range"; r.min = min; r.max = max; r.step = 1; r.value = answers[f.key];
      r.setAttribute("aria-label", f.question);
      const show = (v) => { val.textContent = isAge ? f.options[v - 1] : isBMI ? `${v} kg/m²` : dayText(v); setRangeFill(r); };
      r.addEventListener("input", () => { answers[f.key] = +r.value; show(+r.value); changed(); });
      show(+r.value);
      controls[f.key] = { set: (v) => { r.value = v; show(v); }, el: r };
      q.append(val, r);
      if (isBMI) {
        const calc = document.createElement("div");
        calc.className = "bmi-calc";
        calc.innerHTML = `<input type="number" placeholder="Height (cm)" min="100" max="230" aria-label="Height in cm"><input type="number" placeholder="Weight (kg)" min="25" max="250" aria-label="Weight in kg">`;
        const [hI, wI] = $$("input", calc);
        const upd = () => {
          const hh = +hI.value / 100, ww = +wI.value;
          if (hh > 0.9 && ww > 20) {
            const b = Math.max(12, Math.min(60, Math.round(ww / (hh * hh))));
            answers.BMI = b; controls.BMI.set(b); changed();
          }
        };
        hI.oninput = upd; wI.oninput = upd;
        q.appendChild(calc);
      }
    }
    return q;
  }

  // result widgets
  const recS = $("#recSlider");
  let lastRisk = 0;
  const predict = debounce(async () => {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers, target_recall: +recS.value / 100 }) });
    showResult(r);
  }, 110);
  function changed() { predict(); }
  recS.addEventListener("input", () => { $("#recVal").textContent = recS.value + "%"; setRangeFill(recS); predict(); });
  setRangeFill(recS);

  const wavePath = (level, amp, phase) => {
    const y0 = 400 * (1 - level);
    let d = `M0 ${y0}`;
    for (let x = 0; x <= 800; x += 10) d += ` L${x} ${(y0 + Math.sin((x / 200) * Math.PI * 2 + phase) * amp).toFixed(1)}`;
    return d + " L800 400 L0 400 Z";
  };
  let level = 0;
  const setLevel = (l) => { $("#w1").setAttribute("d", wavePath(l, 16, 1.3)); $("#w2").setAttribute("d", wavePath(l, 12, 0)); };
  setLevel(0);

  function showResult(r) {
    const risk = r.risk;
    const thr = r.operating_point.threshold_calibrated;
    const col = r.flag ? (risk > 2 * thr ? "#ff7a45" : "#f2b441") : "#2de2c4";
    const orb = $("#orb");
    orb.style.setProperty("--orb-glow", col + "88");
    document.documentElement.style.setProperty("--orb-c", col);
    $("#w1").setAttribute("fill", col + "55");
    $("#w2").setAttribute("fill", col + "bb");
    const target = Math.min(0.94, 0.06 + (risk / 0.6) * 0.88);
    tween(level, target, 1100, (v) => { level = v; setLevel(v); });
    tween(lastRisk, risk, 1100, (v) => { $("#riskNum").textContent = (v * 100).toFixed(1) + "%"; });
    lastRisk = risk;
    const v = $("#verdict");
    v.className = "verdict " + (r.flag ? "flag" : "clear");
    v.innerHTML = r.flag
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 3l9 16H3l9-16z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg> Flagged: recommend a blood-sugar test`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2"/><path d="M8 12l3 3 5-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> Below the screening threshold`;
    $("#relText").textContent = `${r.relative_risk.toFixed(2)}× the average adult's risk (${(r.base_rate * 100).toFixed(1)}%) · threshold ${(thr * 100).toFixed(1)}%`;
    const op = r.operating_point;
    $("#opText").innerHTML = `At this setting the screener catches <b>${(op.recall * 100).toFixed(0)}%</b> of real cases on the test set, sends <b>${(op.flagged_rate * 100).toFixed(0)}%</b> of people for a blood test, and <b>${(op.precision * 100).toFixed(0)}%</b> of those flagged truly have diabetes or prediabetes.`;
    const ev = $("#evidence");
    const top = r.contributions.slice(0, 8);
    const rows = $$(".ev", ev);
    top.forEach((c, i) => {
      let row = rows[i];
      if (!row) {
        row = document.createElement("div");
        row.className = "ev";
        row.innerHTML = `<span class="ev__name"></span><span class="ev__track"><i class="ev__bar"></i></span><span class="ev__val"></span>`;
        ev.appendChild(row);
      }
      const f = feats.find((x) => x.key === c.key);
      $(".ev__name", row).textContent = `${c.label}: ${answerText(f, c.value)}`;
      $(".ev__name", row).title = $(".ev__name", row).textContent;
      const w = Math.min(50, (Math.abs(c.log_lr) / 1.6) * 50);
      const bar = $(".ev__bar", row);
      bar.style.left = c.log_lr >= 0 ? "50%" : 50 - w + "%";
      bar.style.width = w + "%";
      bar.style.background = c.log_lr >= 0 ? "var(--coral)" : "var(--teal-deep)";
      $(".ev__val", row).textContent = "×" + c.odds_multiplier.toFixed(2);
    });
    $("#popText").innerHTML = `Population curve (the trained polynomial): at BMI ${answers.BMI}, <b>${(r.population_rate_at_bmi * 100).toFixed(1)}%</b> of surveyed adults have diabetes or prediabetes.`;
  }
  function answerText(f, v) {
    if (f.kind === "binary") return f.options[v];
    if (f.kind === "ordinal") return f.options[f.levels.indexOf(v)];
    if (f.kind === "days") return dayText(v);
    return String(v);
  }

  const presets = {
    young: { ...meta.defaults, Age: 1, BMI: 22, GenHlth: 1, PhysActivity: 1, HighBP: 0, HighChol: 0, Income: 5, Education: 5, CholCheck: 0 },
    mid: { ...meta.defaults, Age: 8, BMI: 29, GenHlth: 3, HighBP: 1, HighChol: 1, PhysActivity: 1, Sex: 1, Income: 6 },
    high: { ...meta.defaults, Age: 11, BMI: 37, GenHlth: 4, HighBP: 1, HighChol: 1, HeartDiseaseorAttack: 1, DiffWalk: 1,
      PhysActivity: 0, PhysHlth: 15, Fruits: 0, Veggies: 0, Income: 3, Education: 4, Sex: 1 },
  };
  $$("#presets button").forEach((b) => b.addEventListener("click", () => {
    const p = presets[b.dataset.preset];
    Object.entries(p).forEach(([k, v]) => {
      const from = answers[k];
      answers[k] = v;
      const c = controls[k];
      if (c?.el && from !== v) tween(from, v, 700, (x) => c.set(Math.round(x)));
      else c?.set(v);
    });
    $$(".qgroup").forEach((g) => g.classList.remove("collapsed"));
    predict();
    toast(`Loaded "${b.textContent}" profile`);
  }));
  predict();
}

// ---------------------------------------------------------------- 06 report
function initReport(o) {
  const t = o.test;
  const cv = o.cv.map((c) => c.auc);
  const mean = cv.reduce((a, b) => a + b, 0) / cv.length;
  const sd = Math.sqrt(cv.reduce((a, b) => a + (b - mean) ** 2, 0) / cv.length);
  $("#metrics").innerHTML = `
    <div class="metric" style="--glow:rgba(45,226,196,.3)"><div class="metric__v" data-count="${t.roc_auc}" data-decimals="3">0</div><div class="metric__l">ROC AUC (test)</div><div class="metric__s">5-fold CV ${mean.toFixed(3)} ± ${sd.toFixed(3)}</div></div>
    <div class="metric" style="--glow:rgba(255,122,69,.3)"><div class="metric__v" data-count="${t.recall * 100}" data-decimals="1" data-suffix="%">0</div><div class="metric__l">Recall (sensitivity)</div><div class="metric__s">${t.tp.toLocaleString()} of ${(t.tp + t.fn).toLocaleString()} cases caught</div></div>
    <div class="metric" style="--glow:rgba(139,124,240,.3)"><div class="metric__v" data-count="${t.precision * 100}" data-decimals="1" data-suffix="%">0</div><div class="metric__l">Precision</div><div class="metric__s">${(t.precision / t.base_rate).toFixed(1)}× the ${(t.base_rate * 100).toFixed(1)}% base rate</div></div>
    <div class="metric" style="--glow:rgba(242,180,65,.3)"><div class="metric__v">${o.sklearn.max_abs_diff_categorical.toExponential(1)}</div><div class="metric__l">max |ours − scikit-learn|</div><div class="metric__s">same probabilities, to machine precision</div></div>`;
  $$("#metrics .metric").forEach((m) => tilt(m, 8));
  initReveals($("#metrics").parentElement);

  const cells = [["", "Pred. no", "Pred. yes"], ["Actual no", [t.tn, "true negatives", "var(--surface-3)"], [t.fp, "false alarms", "rgba(242,180,65,.18)"]],
    ["Actual yes", [t.fn, "missed", "rgba(208,59,59,.2)"], [t.tp, "caught", "rgba(255,122,69,.25)"]]];
  const cm = $("#cm");
  cells.forEach((row, r) => row.forEach((c, j) => {
    const d = document.createElement("div");
    if (r === 0 || j === 0) { d.className = "h"; d.textContent = c; }
    else { d.className = "c"; d.style.background = c[2]; d.style.transitionDelay = (r * 2 + j) * 120 + "ms"; d.innerHTML = `<b>${c[0].toLocaleString()}</b><span>${c[1]}</span>`; }
    cm.appendChild(d);
  }));
  $("#cmSub").textContent = `Calibrated threshold ${(o.threshold_calibrated * 100).toFixed(1)}% · ${(t.tp + t.fp + t.tn + t.fn).toLocaleString()} test people`;

  const lazy = (el, id) => {
    whenVisible(el, async () => {
      const g = await api("/api/graphs/" + id);
      g.figure.layout.height = 320;
      g.figure.layout.legend = { orientation: "v", x: 1, xanchor: "right", y: 0.02, yanchor: "bottom",
        bgcolor: "rgba(12,17,34,.85)", bordercolor: "rgba(150,172,255,.18)", borderwidth: 1, font: { size: 10.5, color: "#b4bdd3" } };
      g.figure.layout.margin = { l: 50, r: 10, t: 10, b: 44 };
      render(el, g.figure);
    });
  };
  lazy($("#rocPlot"), "roc");
  lazy($("#calPlot"), "calibration");
}
