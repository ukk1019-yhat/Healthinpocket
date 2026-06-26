const API_BASE = "/api/v1";

const DIAGNOSE_URL = `${API_BASE}/diagnose`;

const testGrid = document.getElementById("testGrid");
const uploadSection = document.getElementById("uploadSection");
const uploadZone = document.getElementById("uploadZone");
const uploadContent = document.getElementById("uploadContent");
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const previewImg = document.getElementById("previewImg");
const changeBtn = document.getElementById("changeBtn");
const diagnoseBtn = document.getElementById("diagnoseBtn");
const processingSection = document.getElementById("processingSection");
const resultsSection = document.getElementById("resultsSection");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultFilename = document.getElementById("resultFilename");
const resultTime = document.getElementById("resultTime");
const probBars = document.getElementById("probBars");
const steps = {
  1: document.getElementById("step1"),
  2: document.getElementById("step2"),
  3: document.getElementById("step3"),
  4: document.getElementById("step4"),
};

let selectedFile = null;

// Test selection
testGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".test-card");
  if (!card || card.classList.contains("disabled")) return;
  document.querySelectorAll(".test-card").forEach((c) => c.classList.remove("active"));
  card.classList.add("active");
  uploadSection.scrollIntoView({ behavior: "smooth" });
});

// Upload: click zone
uploadZone.addEventListener("click", () => fileInput.click());

// Upload: drag & drop
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/tiff"];
  if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|tiff|tif)$/i)) {
    alert("Please upload a JPG, PNG, or TIFF image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert("File is too large. Maximum size is 10 MB.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadContent.classList.add("hidden");
    previewArea.classList.remove("hidden");
    diagnoseBtn.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

changeBtn.addEventListener("click", () => {
  selectedFile = null;
  previewArea.classList.add("hidden");
  uploadContent.classList.remove("hidden");
  diagnoseBtn.classList.add("hidden");
  resultsSection.classList.add("hidden");
  fileInput.value = "";
});

// Diagnose
diagnoseBtn.addEventListener("click", runDiagnosis);

async function runDiagnosis() {
  if (!selectedFile) return;
  resultsSection.classList.add("hidden");
  processingSection.classList.remove("hidden");
  resetSteps();

  await animateSteps();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch(DIAGNOSE_URL, { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Server error");
    }
    const data = await resp.json();
    showResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    processingSection.classList.add("hidden");
  }
}

async function animateSteps() {
  const delay = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 1; i <= 4; i++) {
    steps[i].classList.add("active");
    await delay(400 + Math.random() * 300);
    steps[i].classList.remove("active");
    steps[i].classList.add("done");
  }
}

function resetSteps() {
  for (let i = 1; i <= 4; i++) {
    steps[i].classList.remove("done", "active");
  }
}

function showResults(data) {
  const primary = data.primary_diagnosis;
  resultLabel.textContent = primary.label;
  resultConfidence.textContent = (primary.confidence * 100).toFixed(1) + "%";
  resultFilename.textContent = data.filename;
  resultTime.textContent = (data.processing_time_ms / 1000).toFixed(2) + "s";

  // Color code the result
  const card = document.getElementById("resultCard");
  card.style.borderLeftColor = primary.class_id === 0 ? "var(--green)" : primary.class_id <= 2 ? "var(--accent)" : "var(--red)";
  resultConfidence.style.color = primary.class_id === 0 ? "var(--green)" : primary.class_id <= 2 ? "var(--accent)" : "var(--red)";

  // Probability bars
  probBars.innerHTML = "";
  data.predictions.forEach((p, i) => {
    const pct = (p.confidence * 100).toFixed(1);
    const isTop = i === 0;
    const barClass = isTop ? "top" : p.confidence > 0.05 ? "mid" : "low";
    const div = document.createElement("div");
    div.className = "prob-bar-item";
    div.innerHTML = `
      <span class="prob-bar-label">${p.label}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${barClass}" style="width: 0%"></div>
      </div>
      <span class="prob-bar-value">${pct}%</span>
    `;
    probBars.appendChild(div);
    // Animate fill
    requestAnimationFrame(() => {
      div.querySelector(".prob-bar-fill").style.width = pct + "%";
    });
  });

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth" });
}
