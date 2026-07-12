(() => {
  const $ = (id) => document.getElementById(id);

  let current = null;

  function heatColor(p) {
    // p = P(B); near 0.5 = uncertain (amber), high B = ink, high I = muted
    const uncertainty = 1 - Math.abs(p - 0.5) * 2;
    const r = Math.round(196 + uncertainty * 40);
    const g = Math.round(92 + (1 - uncertainty) * 80);
    const b = Math.round(38 + (1 - p) * 90);
    return `rgba(${r}, ${g}, ${b}, ${0.25 + uncertainty * 0.45})`;
  }

  function renderResult(el, data, { showGold } = {}) {
    const chips = (data.segments || [])
      .map((s, i) => `<span class="chip" style="animation-delay:${i * 0.02}s">${escapeHtml(s)}</span>`)
      .join("");

    let heat = "";
    if (data.char_probs && data.char_probs.length === data.text.length) {
      heat = `<div class="heat" title="P(boundary) per character">${[...data.text]
        .map((ch, i) => {
          const p = data.char_probs[i];
          return `<span style="background:${heatColor(p)}" title="P(B)=${p.toFixed(3)}">${escapeHtml(ch)}</span>`;
        })
        .join("")}</div>`;
    }

    const metrics = [
      `confidence ${Number(data.confidence).toFixed(4)}`,
      `source ${data.source}`,
      data.query_score != null ? `query ${Number(data.query_score).toFixed(4)}` : null,
      data.mean_entropy != null ? `entropy ${Number(data.mean_entropy).toFixed(4)}` : null,
      data.joined_ok === false ? "WARN: join mismatch" : null,
    ]
      .filter(Boolean)
      .map((m) => `<span>${m}</span>`)
      .join("");

    el.classList.remove("empty");
    el.innerHTML = `
      <div class="seg-row">${chips || "<em>no segments</em>"}</div>
      <div class="metrics">${metrics}</div>
      ${heat}
      ${showGold && data.gold_segments ? `<div class="toast">gold: ${escapeHtml(data.gold_segments.join(" | "))}</div>` : ""}
    `;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function api(path, opts) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(opts && opts.headers) },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText || "request failed");
    return data;
  }

  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      const s = await api("/api/active/stats");
      $("healthMeta").textContent = `${h.model}\npool ${h.pool_size} · reviewed ${h.reviewed}`;
      $("activeStats").textContent =
        `remaining ${s.remaining}\n` +
        `accept ${s.actions.accept} · correct ${s.actions.correct} · skip ${s.actions.skip}`;
    } catch (e) {
      $("healthMeta").textContent = String(e.message || e);
    }
  }

  function setActiveEnabled(on) {
    ["btnAccept", "btnCorrect", "btnSkip", "btnReveal"].forEach((id) => {
      $(id).disabled = !on;
    });
    $("correctBox").classList.toggle("hidden", !on);
  }

  async function doSplit() {
    const text = $("inputText").value.trim();
    if (!text) return;
    const out = $("splitOut");
    out.textContent = "splitting…";
    out.classList.remove("empty");
    try {
      const data = await api("/api/split", {
        method: "POST",
        body: JSON.stringify({
          text,
          fallback: $("fallback").checked,
          min_confidence: Number($("minConf").value || 0.96),
          model: $("modelSel") ? $("modelSel").value : "default",
        }),
      });
      renderResult(out, data);
    } catch (e) {
      out.textContent = String(e.message || e);
    }
  }

  async function nextActive() {
    const out = $("activeOut");
    out.textContent = "scoring uncertain examples…";
    out.classList.remove("empty");
    try {
      const data = await api("/api/active/next?limit=1");
      if (!data.items || !data.items.length) {
        current = null;
        setActiveEnabled(false);
        out.textContent = data.message || "No more items.";
        await refreshHealth();
        return;
      }
      current = data.items[0];
      renderResult(out, current);
      $("correctEdit").value = (current.segments || []).join(" ");
      setActiveEnabled(true);
      await refreshHealth();
    } catch (e) {
      out.textContent = String(e.message || e);
      setActiveEnabled(false);
    }
  }

  async function sendFeedback(action) {
    if (!current) return;
    let corrected = null;
    if (action === "correct") {
      corrected = $("correctEdit").value
        .trim()
        .split(/\s+/)
        .filter(Boolean);
      if (corrected.join("") !== current.text) {
        alert("Correction must concatenate to the original text (spaces mark boundaries only).");
        return;
      }
    }
    try {
      await api("/api/active/feedback", {
        method: "POST",
        body: JSON.stringify({
          text: current.text,
          model_segments: current.segments,
          model_confidence: current.confidence,
          action,
          corrected_segments: corrected,
        }),
      });
      current = null;
      setActiveEnabled(false);
      $("activeOut").innerHTML = `<div class="toast">Saved «${action}». Fetching next…</div>`;
      await refreshHealth();
      await nextActive();
    } catch (e) {
      alert(String(e.message || e));
    }
  }

  async function revealGold() {
    if (!current) return;
    try {
      const g = await api(`/api/active/reveal-gold?text=${encodeURIComponent(current.text)}`);
      current.gold_segments = g.gold_segments;
      renderResult($("activeOut"), current, { showGold: true });
      $("correctEdit").value = g.gold_segments.join(" ");
    } catch (e) {
      alert(String(e.message || e));
    }
  }

  $("btnSplit").addEventListener("click", doSplit);
  $("btnNext").addEventListener("click", nextActive);
  $("btnAccept").addEventListener("click", () => sendFeedback("accept"));
  $("btnCorrect").addEventListener("click", () => sendFeedback("correct"));
  $("btnSkip").addEventListener("click", () => sendFeedback("skip"));
  $("btnReveal").addEventListener("click", revealGold);
  $("inputText").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") doSplit();
  });

  setActiveEnabled(false);
  refreshHealth();
  doSplit();
})();
