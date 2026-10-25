/* =========================================================================
   dashboard.js — dashboard page logic
   Loads model metrics and prediction history, renders the table, supports
   clearing history, and shares the dark-mode toggle.
   ========================================================================= */

(() => {
  "use strict";

  // ---------------- Dark mode (shared behavior) ---------------- //
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const savedTheme = localStorage.getItem("theme") || "light";
  root.setAttribute("data-theme", savedTheme);
  themeToggle.textContent = savedTheme === "dark" ? "☀️" : "🌙";
  themeToggle.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    themeToggle.textContent = next === "dark" ? "☀️" : "🌙";
  });

  // ---------------- Metrics ---------------- //
  async function loadMetrics() {
    try {
      const res = await fetch("/metrics");
      if (!res.ok) return;
      const m = await res.json();
      if (m.test_accuracy != null)
        document.getElementById("mAccuracy").textContent =
          (m.test_accuracy * 100).toFixed(2) + "%";
      if (m.test_loss != null)
        document.getElementById("mLoss").textContent = m.test_loss.toFixed(4);
      if (m.epochs_trained != null)
        document.getElementById("mEpochs").textContent = m.epochs_trained;
    } catch (_) {
      /* metrics optional */
    }
  }

  // ---------------- History ---------------- //
  async function loadHistory() {
    const body = document.getElementById("historyBody");
    try {
      const res = await fetch("/history");
      const rows = await res.json();
      if (!rows.length) {
        body.innerHTML =
          '<tr><td colspan="6" class="text-center text-muted">No predictions yet.</td></tr>';
        return;
      }
      body.innerHTML = rows
        .map(
          (r) => `
        <tr>
          <td>${r.id}</td>
          <td><img class="thumb" src="${r.image_path}" alt="digit ${r.predicted_digit}" /></td>
          <td><strong>${r.predicted_digit}</strong></td>
          <td>${(r.confidence * 100).toFixed(1)}%</td>
          <td>${r.source}</td>
          <td>${r.timestamp}</td>
        </tr>`
        )
        .join("");
    } catch (err) {
      body.innerHTML =
        '<tr><td colspan="6" class="text-center text-danger">Failed to load history.</td></tr>';
    }
  }

  document.getElementById("clearHistoryBtn").addEventListener("click", async () => {
    if (!confirm("Clear all prediction history? This cannot be undone.")) return;
    try {
      await fetch("/history/clear", { method: "POST" });
      loadHistory();
    } catch (err) {
      alert("Failed to clear history.");
    }
  });

  // ---------------- Init ---------------- //
  loadMetrics();
  loadHistory();
})();
