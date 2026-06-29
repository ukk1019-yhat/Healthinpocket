const API_BASE = "/api/v1";

let selectedFile = null;
let currentTest = "retinopathy";

const testGrid = document.getElementById("testGrid");
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
const uploadHint = document.getElementById("uploadHint");
const steps = {
  1: document.getElementById("step1"),
  2: document.getElementById("step2"),
  3: document.getElementById("step3"),
  4: document.getElementById("step4"),
};

const TEST_HINTS = {
  retinopathy: "a retinal image",
  skin: "a skin lesion photo",
};

// --- Test selection ---
document.querySelector(".test-card[data-test='retinopathy']").classList.add("active");

testGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".test-card");
  if (!card || card.classList.contains("disabled")) return;
  if (card.dataset.test === currentTest) return;
  document.querySelectorAll(".test-card").forEach((c) => c.classList.remove("active"));
  card.classList.add("active");
  currentTest = card.dataset.test;
  uploadHint.textContent = TEST_HINTS[currentTest] || "a medical image";
  resultsSection.classList.add("hidden");
  changeBtn.click();
  document.getElementById("uploadSection").scrollIntoView({ behavior: "smooth" });
});

// --- Upload ---
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => { e.preventDefault(); uploadZone.classList.remove("dragover"); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (!file.name.match(/\.(jpg|jpeg|png|tiff|tif)$/i)) { alert("Please upload a JPG, PNG, or TIFF image."); return; }
  if (file.size > 10 * 1024 * 1024) { alert("File too large. Max 10 MB."); return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => { previewImg.src = e.target.result; uploadContent.classList.add("hidden"); previewArea.classList.remove("hidden"); diagnoseBtn.classList.remove("hidden"); };
  reader.readAsDataURL(file);
}

changeBtn.addEventListener("click", () => {
  selectedFile = null; previewArea.classList.add("hidden"); uploadContent.classList.remove("hidden");
  diagnoseBtn.classList.add("hidden"); resultsSection.classList.add("hidden"); fileInput.value = "";
});

// --- Diagnose ---
diagnoseBtn.addEventListener("click", runDiagnosis);

async function runDiagnosis() {
  if (!selectedFile) return;
  resultsSection.classList.add("hidden");
  processingSection.classList.remove("hidden");
  resetSteps();
  await animateSteps();

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("test_type", currentTest);

  try {
    const resp = await fetch(`${API_BASE}/diagnose`, { method: "POST", body: formData });
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || "Server error"); }
    const data = await resp.json();
    showResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  }
  processingSection.classList.add("hidden");
}

async function animateSteps() {
  for (let i = 1; i <= 4; i++) {
    steps[i].classList.add("active");
    await new Promise((r) => setTimeout(r, 400 + Math.random() * 300));
    steps[i].classList.remove("active"); steps[i].classList.add("done");
  }
}
function resetSteps() { for (let i = 1; i <= 4; i++) steps[i].classList.remove("done", "active"); }

function showResults(data) {
  const p = data.primary_diagnosis;
  resultLabel.textContent = p.label;
  resultConfidence.textContent = (p.confidence * 100).toFixed(1) + "%";
  resultFilename.textContent = data.filename;
  resultTime.textContent = (data.processing_time_ms / 1000).toFixed(2) + "s";

  const card = document.getElementById("resultCard");
  const isHealthy = data.test_type === "retinopathy" ? p.class_id === 0 : p.class_id === 5 || p.class_id === 6 || p.class_id === 0;
  card.style.borderLeftColor = isHealthy ? "var(--green)" : p.confidence > 0.7 ? "var(--red)" : "var(--accent)";
  resultConfidence.style.color = isHealthy ? "var(--green)" : p.confidence > 0.7 ? "var(--red)" : "var(--accent)";

  probBars.innerHTML = "";
  data.predictions.forEach((pred, i) => {
    const pct = (pred.confidence * 100).toFixed(1);
    const isTop = i === 0;
    const div = document.createElement("div");
    div.className = "prob-bar-item";
    div.innerHTML = `<span class="prob-bar-label">${pred.label}</span><div class="prob-bar-track"><div class="prob-bar-fill ${isTop ? "top" : pred.confidence > 0.05 ? "mid" : "low"}" style="width:0%"></div></div><span class="prob-bar-value">${pct}%</span>`;
    probBars.appendChild(div);
    requestAnimationFrame(() => { div.querySelector(".prob-bar-fill").style.width = pct + "%"; });
  });

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth" });

  fetchExplanation(data);
}

async function fetchExplanation(data) {
  const section = document.getElementById("aiExplanation");
  const loading = document.getElementById("explanationLoading");
  const content = document.getElementById("explanationContent");
  section.classList.remove("hidden");
  loading.classList.remove("hidden");
  content.classList.add("hidden");
  content.innerHTML = "";

  try {
    const resp = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test_type: data.test_type, predictions: data.predictions, primary_diagnosis: data.primary_diagnosis }),
    });
    if (!resp.ok) throw new Error("AI explanation unavailable");
    const result = await resp.json();
    content.innerHTML = result.explanation.replace(/\*\*/g, "").replace(/\*/g, "").replace(/\n/g, "<br>");
    loading.classList.add("hidden");
    content.classList.remove("hidden");
  } catch {
    loading.innerHTML = "<span style='color:var(--gray-500)'>AI explanation unavailable right now</span>";
  }
}
