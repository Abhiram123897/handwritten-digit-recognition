/* =========================================================================
   script.js — index page logic
   Canvas drawing (mouse + touch), debounced real-time prediction, upload with
   drag-and-drop + preview, Chart.js probability chart, dark-mode toggle.
   ========================================================================= */

(() => {
  "use strict";

  // ---------------- Dark mode ---------------- //
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

  // ---------------- Canvas setup ---------------- //
  const canvas = document.getElementById("drawCanvas");
  const ctx = canvas.getContext("2d");
  let drawing = false;

  function initCanvas() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }
  initCanvas();

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  }

  function startDraw(e) {
    e.preventDefault();
    drawing = true;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  }
  function moveDraw(e) {
    if (!drawing) return;
    e.preventDefault();
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  }
  function endDraw() {
    if (!drawing) return;
    drawing = false;
    debouncedPredict(); // real-time prediction after each stroke
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  canvas.addEventListener("mouseup", endDraw);
  canvas.addEventListener("mouseleave", endDraw);
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", moveDraw, { passive: false });
  canvas.addEventListener("touchend", endDraw);

  // ---------------- Chart.js probability chart ---------------- //
  const probCtx = document.getElementById("probChart").getContext("2d");
  const probChart = new Chart(probCtx, {
    type: "bar",
    data: {
      labels: [...Array(10).keys()].map(String),
      datasets: [
        {
          label: "Probability",
          data: new Array(10).fill(0),
          backgroundColor: "rgba(99, 102, 241, 0.7)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true, max: 1 } },
      plugins: { legend: { display: false } },
    },
  });

  // ---------------- Helpers ---------------- //
  function debounce(fn, wait) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), wait);
    };
  }

  function renderResult(data) {
    document.getElementById("resultDigit").textContent = data.digit;
    const pct = Math.round(data.confidence * 100);
    document.getElementById("confidenceText").textContent = `${pct}%`;
    const bar = document.getElementById("confidenceBar");
    bar.style.width = `${pct}%`;
    bar.textContent = `${pct}%`;
    probChart.data.datasets[0].data = data.probabilities;
    probChart.update();
  }

  function showLoading(id, on) {
    document.getElementById(id).classList.toggle("active", on);
  }

  async function postPredict(payload, isForm, loadingId) {
    showLoading(loadingId, true);
    try {
      const opts = isForm
        ? { method: "POST", body: payload }
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          };
      const res = await fetch("/predict", opts);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      renderResult(data);
    } catch (err) {
      alert(err.message);
    } finally {
      showLoading(loadingId, false);
    }
  }

  // ---------------- Canvas predict ---------------- //
  function canvasIsBlank() {
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < pixels.length; i += 4) {
      // any non-black pixel means something was drawn
      if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 10) return false;
    }
    return true;
  }

  function predictCanvas() {
    if (canvasIsBlank()) return;
    const dataUrl = canvas.toDataURL("image/png");
    postPredict({ image: dataUrl, source: "canvas" }, false, "loadingDraw");
  }
  const debouncedPredict = debounce(predictCanvas, 500);

  document.getElementById("predictBtn").addEventListener("click", predictCanvas);
  document.getElementById("clearBtn").addEventListener("click", () => {
    initCanvas();
    document.getElementById("resultDigit").textContent = "–";
    document.getElementById("confidenceText").textContent = "0%";
    const bar = document.getElementById("confidenceBar");
    bar.style.width = "0%";
    bar.textContent = "";
    probChart.data.datasets[0].data = new Array(10).fill(0);
    probChart.update();
  });

  // ---------------- Upload + drag & drop ---------------- //
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const uploadBtn = document.getElementById("uploadPredictBtn");
  let selectedFile = null;

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please choose an image file.");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.style.display = "inline-block";
    };
    reader.readAsDataURL(file);
    uploadBtn.disabled = false;
  }

  uploadBtn.addEventListener("click", () => {
    if (!selectedFile) return;
    const fd = new FormData();
    fd.append("image", selectedFile);
    fd.append("source", "upload");
    postPredict(fd, true, "loadingUpload");
  });
})();
