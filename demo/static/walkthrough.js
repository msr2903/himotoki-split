(() => {
  const fmt = (x, d = 4) => (x == null ? "—" : Number(x).toFixed(d));
  const pct = (x) => (x == null ? "—" : `${(Number(x) * 100).toFixed(1)}%`);

  function row(title, bodyHtml) {
    return `<div class="live-row"><div class="k">${title}</div><div class="v">${bodyHtml}</div></div>`;
  }

  async function load() {
    const board = document.getElementById("liveBoard");
    try {
      const res = await fetch("/api/walkthrough");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "failed");

      const mild = data.active_mild || {};
      const pre = data.pre_active || {};
      const phaseB = data.phase_b || {};
      const selected = data.selected || "active_mild";

      const cleanF1 = mild.clean?.boundary_f1 ?? phaseB.holdout_clean_800?.onnx_f1;
      const cleanExact = mild.clean?.exact_seg ?? phaseB.holdout_clean_800?.onnx_exact;
      const holdF1 = mild.boundary_f1 ?? phaseB.holdout_5k?.onnx_f1;
      const holdExact = mild.exact_seg ?? phaseB.holdout_5k?.onnx_exact;

      const f1El = document.getElementById("liveF1");
      const exEl = document.getElementById("liveExact");
      if (f1El) f1El.textContent = fmt(cleanF1);
      if (exEl) exEl.textContent = pct(cleanExact);

      board.innerHTML = [
        row(
          "Shipped model",
          `<strong>${selected}</strong> → <code>himotoki_split/models/default.onnx</code>`
        ),
        row(
          "Clean eval (~800)",
          `<div class="metric-pills">
            <span class="pill">F1 ${fmt(cleanF1)}</span>
            <span class="pill">exact ${pct(cleanExact)}</span>
           </div>
           <div class="note" style="margin-top:0.4rem">Best number to quote for “how good does it feel?”</div>`
        ),
        row(
          "Full holdout (5k)",
          `<div class="metric-pills">
            <span class="pill">F1 ${fmt(holdF1)}</span>
            <span class="pill">exact ${pct(holdExact)}</span>
           </div>
           <div class="note" style="margin-top:0.4rem">Noisier wiki text — exact looks lower.</div>`
        ),
        row(
          "Before active query",
          `<div class="metric-pills">
            <span class="pill">clean F1 ${fmt(pre.clean?.boundary_f1)}</span>
            <span class="pill">clean exact ${pct(pre.clean?.exact_seg)}</span>
           </div>`
        ),
        row(
          "Active query batch",
          data.active_query
            ? `Scored ${data.active_query.sample_n}, labeled top ${data.active_query.query_k}
               (${data.active_query.n_correct} corrections / ${data.active_query.n_accept} accepts).`
            : "—"
        ),
        row(
          "Phase B bar",
          phaseB.success_bar_met
            ? "Met (exact ≥ 20% or F1 ≥ 0.94 on holdout)."
            : "Not recorded."
        ),
      ].join("");

      if (data.train_log_sample) {
        document.getElementById("logSample").textContent = data.train_log_sample;
      }
    } catch (e) {
      board.textContent = `Could not load live metrics: ${e.message || e}`;
    }
  }

  load();
})();
