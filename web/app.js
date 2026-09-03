(() => {
  "use strict";

  const W = 96;
  const H = 96;
  const CHANNELS = 3;
  const TILE = 6;
  const FLOOR = [0.012, 0.027, 0.040];
  const bodyCanvas = document.getElementById("bodyCanvas");
  const bodyCtx = bodyCanvas.getContext("2d");
  const fxCanvas = document.getElementById("fxCanvas");
  const fxCtx = fxCanvas.getContext("2d");
  const pulseCanvas = document.getElementById("pulseCanvas");
  const pulseCtx = pulseCanvas.getContext("2d");
  const lowCanvas = document.createElement("canvas");
  lowCanvas.width = W;
  lowCanvas.height = H;
  const lowCtx = lowCanvas.getContext("2d", { willReadFrequently: true });

  const els = {
    wrap: document.getElementById("canvasWrap"),
    callout: document.getElementById("canvasCallout"),
    mode: document.getElementById("canvasMode"),
    health: document.getElementById("healthValue"),
    pulseCount: document.getElementById("pulseCount"),
    budget: document.getElementById("budgetInput"),
    budgetOut: document.getElementById("budgetOutput"),
    agents: document.getElementById("agentsInput"),
    agentsOut: document.getElementById("agentsOutput"),
    wound: document.getElementById("woundButton"),
    run: document.getElementById("runButton"),
    reset: document.getElementById("resetButton"),
    reveal: document.getElementById("revealButton"),
    upload: document.getElementById("uploadInput"),
    eventLog: document.getElementById("eventLog"),
    logState: document.getElementById("logState"),
    activeBar: document.getElementById("activeBar"),
    randomBar: document.getElementById("randomBar"),
    oracleBar: document.getElementById("oracleBar"),
    activeScore: document.getElementById("activeScore"),
    randomScore: document.getElementById("randomScore"),
    oracleScore: document.getElementById("oracleScore"),
    comparisonNote: document.getElementById("comparisonNote"),
    how: document.getElementById("howButton"),
    drawer: document.getElementById("methodDrawer"),
    share: document.getElementById("shareButton"),
  };

  let target = makeBody();
  let state = target.slice();
  let woundSnapshot = null;
  let revealed = false;
  let running = false;
  let drawing = false;
  let pulseCalls = 0;
  let pulseHistory = [100];
  let initialDamage = 0;
  let targetEnergy = energy(target);
  let runToken = 0;
  let probeMarks = [];
  let lastPointer = null;

  function idx(x, y, c = 0) {
    return (y * W + x) * CHANNELS + c;
  }

  function clamp(value, lo, hi) {
    return Math.max(lo, Math.min(hi, value));
  }

  function makeBody() {
    const out = new Float32Array(W * H * CHANNELS);
    for (let y = 0; y < H; y += 1) {
      const yy = -1 + (2 * y) / (H - 1);
      for (let x = 0; x < W; x += 1) {
        const xx = -1 + (2 * x) / (W - 1);
        const ax = Math.abs(xx);
        const base = idx(x, y);
        let r = FLOOR[0];
        let g = FLOOR[1];
        let b = FLOOR[2];

        const body = Math.exp(-2.2 * ((xx / 0.14) ** 2 + (yy / 0.70) ** 2));
        const head = Math.exp(-2.0 * ((xx / 0.22) ** 2 + ((yy + 0.61) / 0.20) ** 2));
        const wingShape = (ax / 0.86) ** 1.65 + ((yy + 0.02) / 0.72) ** 2 < 1;
        const shoulderCut = ax > 0.12 + 0.18 * clamp(yy + 0.7, 0, 1.4);
        const wing = wingShape && shoulderCut ? 1 : 0;
        const fade = clamp(1 - (ax / 0.92) ** 2 - ((yy + 0.02) / 0.77) ** 2, 0, 1) * wing;
        const veins = Math.exp(-((Math.sin(18 * ax + 4.5 * yy) * 0.55) ** 2) / 0.025)
          * Math.exp(-(((ax - 0.52) / 0.38) ** 2)) * wing;
        const ribs = Math.exp(-((Math.sin(14 * yy - 5 * ax) * 0.55) ** 2) / 0.035)
          * Math.exp(-(((ax - 0.48) / 0.42) ** 2)) * wing;

        r += 0.06 * fade + 0.20 * veins + 0.05 * ribs + 0.20 * body + 0.25 * head;
        g += 0.20 * fade + 0.80 * veins + 0.34 * ribs + 0.64 * body + 0.58 * head;
        b += 0.27 * fade + 0.88 * veins + 0.80 * ribs + 0.70 * body + 0.46 * head;

        for (const eyeX of [-0.085, 0.085]) {
          const eye = Math.exp(-2 * (((xx - eyeX) / 0.028) ** 2 + ((yy + 0.64) / 0.032) ** 2));
          r += 0.92 * eye;
          g += 0.84 * eye;
          b += 0.32 * eye;
        }
        out[base] = clamp(r, 0, 1);
        out[base + 1] = clamp(g, 0, 1);
        out[base + 2] = clamp(b, 0, 1);
      }
    }
    return out;
  }

  function bodyFromImage(image) {
    const temp = document.createElement("canvas");
    temp.width = W;
    temp.height = H;
    const ctx = temp.getContext("2d");
    ctx.fillStyle = "#061014";
    ctx.fillRect(0, 0, W, H);
    const scale = Math.max(W / image.width, H / image.height);
    const dw = image.width * scale;
    const dh = image.height * scale;
    ctx.drawImage(image, (W - dw) / 2, (H - dh) / 2, dw, dh);
    const pixels = ctx.getImageData(0, 0, W, H).data;
    const out = new Float32Array(W * H * CHANNELS);
    for (let y = 0; y < H; y += 1) {
      for (let x = 0; x < W / 2; x += 1) {
        const p = (y * W + x) * 4;
        const left = idx(x, y);
        const right = idx(W - 1 - x, y);
        for (let c = 0; c < 3; c += 1) {
          const value = 0.02 + 0.92 * (pixels[p + c] / 255);
          out[left + c] = value;
          out[right + c] = value;
        }
      }
    }
    return out;
  }

  function energy(array) {
    let total = 0;
    for (let i = 0; i < array.length; i += 1) total += array[i] * array[i];
    return total;
  }

  function rawPulse(candidate) {
    let error = 0;
    for (let i = 0; i < candidate.length; i += 1) {
      const d = candidate[i] - target[i];
      error += d * d;
    }
    return -error;
  }

  function measuredPulse(candidate) {
    pulseCalls += 1;
    els.pulseCount.textContent = String(pulseCalls);
    const score = rawPulse(candidate);
    const relative = initialDamage > 0 ? 100 * (1 + score / initialDamage) : 100;
    pulseHistory.push(clamp(relative, -8, 105));
    if (pulseHistory.length > 70) pulseHistory.shift();
    drawPulse();
    return score;
  }

  function health(candidate = state) {
    const error = -rawPulse(candidate);
    const displayScale = Math.max(targetEnergy * 0.08, initialDamage || 1);
    return 100 * clamp(1 - error / displayScale, 0, 1);
  }

  function updateHealth() {
    els.health.textContent = health().toFixed(1);
    drawPulse();
  }

  function toImageData(array) {
    const data = lowCtx.createImageData(W, H);
    for (let p = 0, q = 0; p < array.length; p += 3, q += 4) {
      data.data[q] = Math.round(255 * clamp(array[p], 0, 1));
      data.data[q + 1] = Math.round(255 * clamp(array[p + 1], 0, 1));
      data.data[q + 2] = Math.round(255 * clamp(array[p + 2], 0, 1));
      data.data[q + 3] = 255;
    }
    return data;
  }

  function renderBody() {
    lowCtx.putImageData(toImageData(state), 0, 0);
    bodyCtx.imageSmoothingEnabled = true;
    bodyCtx.clearRect(0, 0, bodyCanvas.width, bodyCanvas.height);
    bodyCtx.drawImage(lowCanvas, 0, 0, bodyCanvas.width, bodyCanvas.height);
    bodyCtx.save();
    bodyCtx.globalCompositeOperation = "screen";
    bodyCtx.globalAlpha = 0.13;
    bodyCtx.filter = "blur(15px)";
    bodyCtx.drawImage(lowCanvas, 0, 0, bodyCanvas.width, bodyCanvas.height);
    bodyCtx.restore();
    if (revealed) drawWoundMap();
  }

  function drawWoundMap() {
    const sx = bodyCanvas.width / W;
    const sy = bodyCanvas.height / H;
    bodyCtx.save();
    bodyCtx.globalCompositeOperation = "screen";
    for (let y = 0; y < H; y += 2) {
      for (let x = W / 2; x < W; x += 2) {
        let d = 0;
        const base = idx(x, y);
        for (let c = 0; c < 3; c += 1) d += Math.abs(target[base + c] - state[base + c]);
        if (d > 0.08) {
          bodyCtx.fillStyle = `rgba(255,84,127,${clamp(d * 0.65, 0.08, 0.72)})`;
          bodyCtx.fillRect(x * sx, y * sy, sx * 2.1, sy * 2.1);
        }
      }
    }
    bodyCtx.restore();
  }

  function drawPulse() {
    const w = pulseCanvas.width;
    const h = pulseCanvas.height;
    pulseCtx.clearRect(0, 0, w, h);
    pulseCtx.strokeStyle = "rgba(121,170,159,.13)";
    pulseCtx.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) {
      pulseCtx.beginPath();
      pulseCtx.moveTo(0, (h * i) / 4);
      pulseCtx.lineTo(w, (h * i) / 4);
      pulseCtx.stroke();
    }
    if (pulseHistory.length < 2) return;
    const grad = pulseCtx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "#3d786d");
    grad.addColorStop(0.7, "#66fbd1");
    grad.addColorStop(1, "#c7ff57");
    pulseCtx.strokeStyle = grad;
    pulseCtx.lineWidth = 2.5;
    pulseCtx.shadowColor = "#66fbd1";
    pulseCtx.shadowBlur = 9;
    pulseCtx.beginPath();
    pulseHistory.forEach((value, i) => {
      const x = (i / Math.max(pulseHistory.length - 1, 1)) * w;
      const y = h - clamp(value / 100, 0.04, 0.96) * h;
      if (i === 0) pulseCtx.moveTo(x, y);
      else pulseCtx.lineTo(x, y);
    });
    pulseCtx.stroke();
    pulseCtx.shadowBlur = 0;
  }

  function woundAt(px, py, radius = 9, strength = 1) {
    if (px < W / 2 + 2) return;
    for (let y = Math.max(0, Math.floor(py - radius)); y < Math.min(H, Math.ceil(py + radius)); y += 1) {
      for (let x = Math.max(W / 2, Math.floor(px - radius)); x < Math.min(W, Math.ceil(px + radius)); x += 1) {
        const distance = Math.hypot(x - px, y - py);
        const mask = clamp((radius - distance) / Math.max(radius * 0.28, 0.1), 0, 1) * strength;
        if (mask <= 0) continue;
        const base = idx(x, y);
        for (let c = 0; c < 3; c += 1) state[base + c] = state[base + c] * (1 - mask) + FLOOR[c] * mask;
      }
    }
    woundSnapshot = state.slice();
    initialDamage = -rawPulse(state);
    pulseHistory = [100, 0];
    resetComparison();
    renderBody();
    updateHealth();
    setStage("wound");
    els.callout.style.opacity = "0";
    log("Wound added. Spatial error map sealed.");
  }

  function pointerToGrid(event) {
    const rect = els.wrap.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * W,
      y: ((event.clientY - rect.top) / rect.height) * H,
    };
  }

  function region(x0, y0, x1, y1) {
    return { x0, y0, x1, y1, get width() { return this.x1 - this.x0; }, get height() { return this.y1 - this.y0; } };
  }

  function isLeaf(r) {
    return r.width <= TILE && r.height <= TILE;
  }

  function splitRegion(r) {
    if (r.width >= r.height && r.width > 1) {
      const xm = r.x0 + Math.floor(r.width / 2);
      return [region(r.x0, r.y0, xm, r.y1), region(xm, r.y0, r.x1, r.y1)];
    }
    const ym = r.y0 + Math.floor(r.height / 2);
    return [region(r.x0, r.y0, r.x1, ym), region(r.x0, ym, r.x1, r.y1)];
  }

  function mirrorTrial(input, r, alpha = 1) {
    const out = input.slice();
    for (let y = r.y0; y < r.y1; y += 1) {
      for (let x = r.x0; x < r.x1; x += 1) {
        const sourceX = W - 1 - x;
        const source = idx(sourceX, y);
        const destination = idx(x, y);
        for (let c = 0; c < 3; c += 1) out[destination + c] = (1 - alpha) * out[destination + c] + alpha * input[source + c];
      }
    }
    return out;
  }

  function probeGain(input, baseline, r, measure = measuredPulse) {
    return measure(mirrorTrial(input, r, 0.35)) - baseline;
  }

  async function adaptiveSearch(input, budget, repairSlots, token) {
    pulseCalls = 0;
    pulseHistory = [0];
    const baseline = measuredPulse(input);
    const root = region(W / 2, 0, W, H);
    const probes = [];
    const leaves = [];
    const queue = [];

    const rootGain = probeGain(input, baseline, root);
    const rootProbe = { region: root, gain: rootGain, depth: 0 };
    probes.push(rootProbe);
    queue.push(rootProbe);
    await showProbe(rootProbe, token);

    while (queue.length && pulseCalls < budget && token === runToken) {
      queue.sort((a, b) => b.gain - a.gain || a.region.width * a.region.height - b.region.width * b.region.height);
      const parent = queue.shift();
      if (parent.gain <= 0) continue;
      if (isLeaf(parent.region)) {
        leaves.push(parent);
        continue;
      }
      for (const child of splitRegion(parent.region)) {
        if (pulseCalls >= budget || token !== runToken) break;
        const gain = probeGain(input, baseline, child);
        const item = { region: child, gain, depth: parent.depth + 1 };
        probes.push(item);
        if (isLeaf(child)) {
          if (gain > 0) leaves.push(item);
        } else if (gain > 0) {
          queue.push(item);
        }
        await showProbe(item, token);
      }
    }
    leaves.sort((a, b) => b.gain - a.gain);
    return { selected: leaves.slice(0, repairSlots), probes, pulseCalls, baseline };
  }

  async function showProbe(item, token) {
    if (token !== runToken) return;
    probeMarks.push(item);
    if (probeMarks.length > 17) probeMarks.shift();
    drawProbeMarks(item);
    const sign = item.gain >= 0 ? "+" : "";
    log(`Mask d${item.depth}: Δpulse ${sign}${normalizedGain(item.gain).toFixed(2)}`);
    await sleep(window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 4 : 58);
  }

  function normalizedGain(gain) {
    return initialDamage > 0 ? (100 * gain) / initialDamage : 0;
  }

  function drawProbeMarks(current) {
    fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
    const sx = fxCanvas.width / W;
    const sy = fxCanvas.height / H;
    probeMarks.forEach((item, index) => {
      const age = 1 - index / Math.max(probeMarks.length, 1);
      const positive = item.gain > 0;
      fxCtx.strokeStyle = positive ? `rgba(102,251,209,${0.16 + 0.25 * age})` : `rgba(255,84,127,${0.10 + 0.18 * age})`;
      fxCtx.lineWidth = item === current ? 3 : 1;
      fxCtx.strokeRect(item.region.x0 * sx + 1, item.region.y0 * sy + 1, item.region.width * sx - 2, item.region.height * sy - 2);
    });
    if (current) {
      fxCtx.fillStyle = current.gain > 0 ? "rgba(102,251,209,.11)" : "rgba(255,84,127,.08)";
      fxCtx.fillRect(current.region.x0 * sx, current.region.y0 * sy, current.region.width * sx, current.region.height * sy);
    }
  }

  async function animateRepairers(selected, token) {
    if (!selected.length) return;
    const sx = fxCanvas.width / W;
    const sy = fxCanvas.height / H;
    const agents = selected.map((item, i) => {
      const endX = (item.region.x0 + item.region.x1) / 2;
      const endY = (item.region.y0 + item.region.y1) / 2;
      const startX = W - 1 - endX;
      const bend = (i % 2 ? -1 : 1) * (8 + i * 1.7);
      return {
        item,
        startX,
        startY: endY,
        endX,
        endY,
        controlX: W / 2,
        controlY: clamp(endY + bend, 4, H - 4),
        delay: i * 130,
        committed: false,
      };
    });

    const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 80 : 1050;
    const start = performance.now();
    await new Promise((resolve) => {
      function frame(now) {
        if (token !== runToken) return resolve();
        const elapsed = now - start;
        fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
        let allDone = true;
        for (const [i, agent] of agents.entries()) {
          const t = clamp((elapsed - agent.delay) / duration, 0, 1);
          if (t < 1) allDone = false;
          const eased = t * t * (3 - 2 * t);
          const omt = 1 - eased;
          const x = omt * omt * agent.startX + 2 * omt * eased * agent.controlX + eased * eased * agent.endX;
          const y = omt * omt * agent.startY + 2 * omt * eased * agent.controlY + eased * eased * agent.endY;

          fxCtx.beginPath();
          fxCtx.moveTo(agent.startX * sx, agent.startY * sy);
          fxCtx.quadraticCurveTo(agent.controlX * sx, agent.controlY * sy, agent.endX * sx, agent.endY * sy);
          fxCtx.strokeStyle = `rgba(102,251,209,${0.15 + 0.22 * (1 - i / agents.length)})`;
          fxCtx.lineWidth = 1.2;
          fxCtx.stroke();

          fxCtx.beginPath();
          fxCtx.arc(x * sx, y * sy, 4.5 + 2 * Math.sin((elapsed + i * 90) / 90), 0, Math.PI * 2);
          fxCtx.fillStyle = i % 2 ? "#c7ff57" : "#66fbd1";
          fxCtx.shadowColor = fxCtx.fillStyle;
          fxCtx.shadowBlur = 16;
          fxCtx.fill();
          fxCtx.shadowBlur = 0;

          if (t >= 1 && !agent.committed) {
            state = mirrorTrial(state, agent.item.region, 1);
            agent.committed = true;
            renderBody();
            updateHealth();
            log(`Observer ${i + 1} committed local write.`);
          }
        }
        if (allDone) resolve();
        else requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  function leafGrid() {
    const out = [];
    for (let y = 0; y < H; y += TILE) {
      for (let x = W / 2; x < W; x += TILE) out.push(region(x, y, Math.min(x + TILE, W), Math.min(y + TILE, H)));
    }
    return out;
  }

  function seededRandom(seed) {
    let value = seed >>> 0;
    return () => {
      value += 0x6D2B79F5;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(array, random) {
    const out = array.slice();
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  function selectRandom(input, budget, repairSlots, seed) {
    const baseline = rawPulse(input);
    const candidates = shuffle(leafGrid(), seededRandom(seed)).slice(0, Math.max(0, budget - 1));
    return candidates
      .map((r) => ({ region: r, gain: probeGain(input, baseline, r, rawPulse) }))
      .filter((item) => item.gain > 0)
      .sort((a, b) => b.gain - a.gain)
      .slice(0, repairSlots);
  }

  function selectExhaustive(input, repairSlots) {
    const baseline = rawPulse(input);
    return leafGrid()
      .map((r) => ({ region: r, gain: probeGain(input, baseline, r, rawPulse) }))
      .filter((item) => item.gain > 0)
      .sort((a, b) => b.gain - a.gain)
      .slice(0, repairSlots);
  }

  function applyRepairs(input, selected) {
    let out = input.slice();
    for (const item of selected) out = mirrorTrial(out, item.region, 1);
    return out;
  }

  function recovered(input, repaired) {
    const deficit = Math.max(-rawPulse(input), 1e-9);
    return clamp((rawPulse(repaired) - rawPulse(input)) / deficit, 0, 1);
  }

  function counterfactuals(input, activeSelected, budget, repairSlots) {
    const active = recovered(input, applyRepairs(input, activeSelected));
    let random = 0;
    const trials = 128;
    for (let seed = 0; seed < trials; seed += 1) random += recovered(input, applyRepairs(input, selectRandom(input, budget, repairSlots, 9301 + seed * 13)));
    random /= trials;
    const exhaustive = recovered(input, applyRepairs(input, selectExhaustive(input, repairSlots)));
    return { active, random, exhaustive, trials };
  }

  async function runRepair() {
    if (running) return;
    if (!woundSnapshot || initialDamage < 1e-6) {
      randomWound();
      await sleep(180);
    }
    running = true;
    const token = ++runToken;
    setControls(true);
    state = woundSnapshot.slice();
    renderBody();
    probeMarks = [];
    resetComparison();
    setStage("probe");
    els.mode.textContent = "SCALAR PROBES · TARGET SEALED";
    els.logState.textContent = "SCANNING";
    log("Baseline pulse cached. No spatial map returned.");

    const budget = Number(els.budget.value);
    const repairSlots = Number(els.agents.value);
    const result = await adaptiveSearch(woundSnapshot, budget, repairSlots, token);
    if (token !== runToken) return;
    log(`${result.selected.length} positive local destinations isolated.`);
    setStage("carry");
    els.mode.textContent = "LOCAL FACTS IN TRANSIT";
    els.logState.textContent = "CARRYING";
    await animateRepairers(result.selected, token);
    if (token !== runToken) return;

    setStage("repair");
    fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
    els.mode.textContent = "REPAIR COMMITTED";
    els.logState.textContent = "COMPLETE";
    const metrics = counterfactuals(woundSnapshot, result.selected, budget, repairSlots);
    showComparison(metrics);
    log(`Recovered ${(metrics.active * 100).toFixed(1)}% of scalar deficit.`);
    running = false;
    setControls(false);
  }

  function showComparison(metrics) {
    const pct = (x) => `${(100 * x).toFixed(1)}%`;
    els.activeBar.style.width = pct(metrics.active);
    els.randomBar.style.width = pct(metrics.random);
    els.oracleBar.style.width = pct(metrics.exhaustive);
    els.activeScore.textContent = pct(metrics.active);
    els.randomScore.textContent = pct(metrics.random);
    els.oracleScore.textContent = pct(metrics.exhaustive);
    els.comparisonNote.textContent = `Random is the mean of ${metrics.trials} paired runs. Full tile scan costs 129 pulses.`;
  }

  function resetComparison() {
    for (const bar of [els.activeBar, els.randomBar, els.oracleBar]) bar.style.width = "0";
    for (const score of [els.activeScore, els.randomScore, els.oracleScore]) score.textContent = "—";
    els.comparisonNote.textContent = "Run repair to generate a paired counterfactual.";
  }

  function randomWound() {
    if (running) return;
    const x = W * (0.65 + Math.random() * 0.19);
    const y = H * (0.25 + Math.random() * 0.50);
    woundAt(x, y, 7 + Math.random() * 4);
    if (Math.random() > 0.55) woundAt(clamp(x + (Math.random() - 0.5) * 18, W * 0.56, W * 0.91), clamp(y + (Math.random() - 0.5) * 24, 12, H - 12), 5 + Math.random() * 3);
  }

  function reset() {
    runToken += 1;
    running = false;
    drawing = false;
    state = target.slice();
    woundSnapshot = null;
    initialDamage = 0;
    pulseCalls = 0;
    pulseHistory = [100, 100];
    probeMarks = [];
    fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height);
    els.pulseCount.textContent = "0";
    els.callout.style.opacity = "1";
    els.mode.textContent = "DRAG TO CUT RIGHT WING";
    els.logState.textContent = "WAITING";
    els.eventLog.innerHTML = "<li><time>00</time><span>Body intact. Add a wound.</span></li>";
    setStage("wound");
    setControls(false);
    resetComparison();
    renderBody();
    updateHealth();
  }

  function setStage(stage) {
    document.querySelectorAll("[data-stage]").forEach((node) => node.classList.toggle("active", node.dataset.stage === stage));
  }

  function setControls(disabled) {
    for (const control of [els.wound, els.run, els.budget, els.agents, els.upload]) control.disabled = disabled;
    els.reset.disabled = false;
  }

  function log(message) {
    const item = document.createElement("li");
    const time = String(pulseCalls).padStart(2, "0");
    item.innerHTML = `<time>${time}</time><span></span>`;
    item.querySelector("span").textContent = message;
    els.eventLog.prepend(item);
    while (els.eventLog.children.length > 6) els.eventLog.removeChild(els.eventLog.lastChild);
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  els.wrap.addEventListener("pointerdown", (event) => {
    if (running) return;
    drawing = true;
    els.wrap.setPointerCapture(event.pointerId);
    lastPointer = pointerToGrid(event);
    woundAt(lastPointer.x, lastPointer.y, 6.5);
  });
  els.wrap.addEventListener("pointermove", (event) => {
    if (!drawing || running) return;
    const point = pointerToGrid(event);
    if (!lastPointer || Math.hypot(point.x - lastPointer.x, point.y - lastPointer.y) > 2.3) {
      woundAt(point.x, point.y, 5.8);
      lastPointer = point;
    }
  });
  const stopDrawing = () => { drawing = false; lastPointer = null; };
  els.wrap.addEventListener("pointerup", stopDrawing);
  els.wrap.addEventListener("pointercancel", stopDrawing);

  els.wound.addEventListener("click", randomWound);
  els.run.addEventListener("click", runRepair);
  els.reset.addEventListener("click", reset);
  els.budget.addEventListener("input", () => { els.budgetOut.textContent = els.budget.value; });
  els.agents.addEventListener("input", () => { els.agentsOut.textContent = els.agents.value; });
  els.reveal.addEventListener("click", () => {
    revealed = !revealed;
    els.reveal.classList.toggle("on", revealed);
    els.reveal.textContent = revealed ? "HIDE WOUND MAP" : "REVEAL WOUND MAP";
    renderBody();
  });
  els.how.addEventListener("click", () => {
    const open = els.drawer.hidden;
    els.drawer.hidden = !open;
    els.how.setAttribute("aria-expanded", String(open));
    els.how.querySelector("span").textContent = open ? "−" : "＋";
  });
  els.upload.addEventListener("change", () => {
    const [file] = els.upload.files;
    if (!file) return;
    const image = new Image();
    image.onload = () => {
      target = bodyFromImage(image);
      targetEnergy = energy(target);
      reset();
      log("Local image loaded. Left half is the repair blueprint.");
      URL.revokeObjectURL(image.src);
    };
    image.src = URL.createObjectURL(file);
  });
  els.share.addEventListener("click", async () => {
    const data = {
      title: "Pulse Repair — the one-number organism",
      text: "Cut a digital organism. Its local repairers get one global number and no wound map.",
      url: window.location.href,
    };
    try {
      if (navigator.share) await navigator.share(data);
      else {
        await navigator.clipboard.writeText(`${data.text} ${data.url}`);
        els.share.textContent = "LINK COPIED ✓";
        setTimeout(() => { els.share.textContent = "SHARE THE WOUND ↗"; }, 1800);
      }
    } catch (_) { /* user cancelled */ }
  });

  renderBody();
  drawPulse();
})();
