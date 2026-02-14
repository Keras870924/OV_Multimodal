// =========================================================================
// Renderer: UI interactions, DICOM preview, and server communication
// =========================================================================

let serverUrl = "http://127.0.0.1:8000";
let serverReady = false;
let selectedFilePath = null;
let pendingDicomPreview = null; // file path awaiting server readiness

// DOM references
const dropZone = document.getElementById("drop-zone");
const dropZoneContent = document.getElementById("drop-zone-content");
const imagePreview = document.getElementById("image-preview");
const imageLoading = document.getElementById("image-loading");
const fileNameEl = document.getElementById("file-name");
const ageInput = document.getElementById("age");
const transducerSelect = document.getElementById("transducer");
const reportTextarea = document.getElementById("report");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const resultsSection = document.getElementById("results-section");
const resultLabel = document.getElementById("result-label");
const barBenign = document.getElementById("bar-benign");
const barCancer = document.getElementById("bar-cancer");
const probBenign = document.getElementById("prob-benign");
const probCancer = document.getElementById("prob-cancer");
const foldContainer = document.getElementById("fold-container");
const serverIndicator = document.getElementById("server-indicator");
const serverStatusText = document.getElementById("server-status-text");

// -------------------------------------------------------------------------
// Server status
// -------------------------------------------------------------------------

window.electronAPI.onServerStatus((status, data) => {
  serverIndicator.className = "status-badge " + status;

  if (status === "ready") {
    serverReady = true;
    serverStatusText.textContent = `Ready (${data.models_loaded} models)`;
    updateAnalyzeButton();
    // If a DICOM file was selected before the server was ready, preview it now
    if (pendingDicomPreview) {
      requestDicomPreview(pendingDicomPreview);
      pendingDicomPreview = null;
    }
  } else if (status === "loading") {
    serverReady = false;
    serverStatusText.textContent =
      typeof data === "string" ? data : "Loading models...";
  } else if (status === "error") {
    serverReady = false;
    serverStatusText.textContent =
      typeof data === "string" ? data : "Server error";
  }
});

window.electronAPI.getServerUrl().then((url) => {
  serverUrl = url;
});

// -------------------------------------------------------------------------
// Image selection
// -------------------------------------------------------------------------

dropZone.addEventListener("click", async () => {
  const filePath = await window.electronAPI.selectImage();
  if (filePath) handleImageSelected(filePath);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) {
    handleImageSelected(e.dataTransfer.files[0].path);
  }
});

async function handleImageSelected(filePath) {
  selectedFilePath = filePath;
  const fileName = filePath.split(/[\\/]/).pop();

  // Show file badge
  fileNameEl.textContent = fileName;
  fileNameEl.classList.add("visible");

  const ext = fileName.split(".").pop().toLowerCase();

  if (["png", "jpg", "jpeg", "bmp", "tif", "tiff"].includes(ext)) {
    // Standard image: read via IPC and display as data URI
    const file = await window.electronAPI.readFile(filePath);
    if (file) {
      const mimeMap = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", bmp: "image/bmp", tif: "image/tiff", tiff: "image/tiff" };
      const mime = mimeMap[ext] || "image/png";
      showPreviewImage(`data:${mime};base64,${file.data}`);
    }
  } else {
    // DICOM file: request server-side conversion
    if (serverReady) {
      requestDicomPreview(filePath);
    } else {
      // Show loading placeholder; will retry when server is ready
      pendingDicomPreview = filePath;
      showLoadingState();
    }
  }

  updateAnalyzeButton();
}

function showPreviewImage(src) {
  imageLoading.classList.add("hidden");
  dropZoneContent.classList.add("hidden");
  imagePreview.src = src;
  imagePreview.classList.remove("hidden");
  dropZone.classList.add("has-image");
}

function showLoadingState() {
  dropZoneContent.classList.add("hidden");
  imagePreview.classList.add("hidden");
  imageLoading.classList.remove("hidden");
}

async function requestDicomPreview(filePath) {
  showLoadingState();
  try {
    const file = await window.electronAPI.readFile(filePath);
    if (!file) throw new Error("Could not read file");

    const byteArray = Uint8Array.from(atob(file.data), (c) => c.charCodeAt(0));
    const blob = new Blob([byteArray], { type: "application/octet-stream" });
    const fileName = filePath.split(/[\\/]/).pop();

    const formData = new FormData();
    formData.append("image", blob, fileName);

    const res = await fetch(`${serverUrl}/preview`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error("Preview request failed");

    const data = await res.json();
    showPreviewImage(data.image);
  } catch (err) {
    console.error("DICOM preview error:", err);
    imageLoading.classList.add("hidden");
    dropZoneContent.classList.remove("hidden");
    dropZoneContent.querySelector(".drop-text").textContent =
      filePath.split(/[\\/]/).pop();
    dropZoneContent.querySelector(".drop-hint").textContent =
      "DICOM selected (preview unavailable)";
    dropZone.classList.add("has-image");
  }
}

// -------------------------------------------------------------------------
// Analyze button
// -------------------------------------------------------------------------

function updateAnalyzeButton() {
  const ok =
    serverReady &&
    selectedFilePath !== null &&
    ageInput.value.trim() !== "" &&
    transducerSelect.value !== "";
  analyzeBtn.disabled = !ok;
}

ageInput.addEventListener("input", updateAnalyzeButton);
transducerSelect.addEventListener("change", updateAnalyzeButton);
analyzeBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  if (!serverReady || !selectedFilePath) return;

  analyzeBtn.disabled = true;
  btnText.textContent = "Analyzing...";
  btnSpinner.classList.remove("hidden");
  resultsSection.classList.add("hidden");

  try {
    // Read file via IPC (avoids file:// fetch restrictions)
    const file = await window.electronAPI.readFile(selectedFilePath);
    if (!file) throw new Error("Could not read the selected file.");

    const byteArray = Uint8Array.from(atob(file.data), (c) => c.charCodeAt(0));
    const blob = new Blob([byteArray], { type: "application/octet-stream" });
    const fileName = selectedFilePath.split(/[\\/]/).pop();

    const fd = new FormData();
    fd.append("image", blob, fileName);
    fd.append("age", ageInput.value);
    fd.append("transducer_type", transducerSelect.value);
    fd.append("report_text", reportTextarea.value || "");

    const res = await fetch(`${serverUrl}/predict`, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `Server returned ${res.status}`);
    }

    displayResults(await res.json());
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
    btnText.textContent = "Analyze";
    btnSpinner.classList.add("hidden");
    updateAnalyzeButton();
  }
}

// -------------------------------------------------------------------------
// Results display
// -------------------------------------------------------------------------

function displayResults(result) {
  resultsSection.classList.remove("hidden");

  const isBenign = result.prediction === "Benign";
  const pct = (result.confidence * 100).toFixed(1);

  resultLabel.className = "verdict-card " + (isBenign ? "benign" : "cancer");
  resultLabel.innerHTML = `
    <span class="verdict-class">${result.prediction}</span>
    <span class="verdict-confidence">Confidence ${pct}%</span>
  `;

  const benignPct = (result.probabilities.Benign * 100).toFixed(1);
  const cancerPct = (result.probabilities.Cancer * 100).toFixed(1);

  barBenign.style.width = `${benignPct}%`;
  barCancer.style.width = `${cancerPct}%`;
  probBenign.textContent = `${benignPct}%`;
  probCancer.textContent = `${cancerPct}%`;

  // Per-fold cards
  foldContainer.innerHTML = "";
  if (result.fold_predictions) {
    result.fold_predictions.forEach((fold) => {
      const fb = (fold.probabilities.Benign * 100).toFixed(1);
      const fc = (fold.probabilities.Cancer * 100).toFixed(1);
      const card = document.createElement("div");
      card.className = "fold-card";
      card.innerHTML = `
        <h4>Fold ${fold.fold}</h4>
        <p class="fold-val">Benign: <span class="v benign-text">${fb}%</span></p>
        <p class="fold-val">Cancer: <span class="v cancer-text">${fc}%</span></p>
      `;
      foldContainer.appendChild(card);
    });
  }

  resultsSection.scrollIntoView({ behavior: "smooth" });
}
