const API_BASE = "/api/v1";

// State
let selectedFile = null;
let accessToken = localStorage.getItem("hip_token") || null;
let userEmail = localStorage.getItem("hip_email") || null;
let isSignUp = false;

// DOM refs
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
const savedBadge = document.getElementById("savedBadge");
const historySection = document.getElementById("historySection");
const historyList = document.getElementById("historyList");
const authStatus = document.getElementById("authStatus");
const loginLink = document.getElementById("loginLink");
const authModal = document.getElementById("authModal");
const modalClose = document.getElementById("modalClose");
const authTitle = document.getElementById("authTitle");
const authForm = document.getElementById("authForm");
const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");
const authError = document.getElementById("authError");
const authSubmit = document.getElementById("authSubmit");
const authToggleText = document.getElementById("authToggleText");
const authToggleLink = document.getElementById("authToggleLink");
const steps = {
  1: document.getElementById("step1"),
  2: document.getElementById("step2"),
  3: document.getElementById("step3"),
  4: document.getElementById("step4"),
};

// Init
updateAuthUI();
if (accessToken) loadHistory();

// --- Auth ---
function updateAuthUI() {
  if (userEmail) {
    authStatus.innerHTML = `<span>Signed in as <strong>${userEmail}</strong> · <a href="#" id="signoutLink">Sign out</a></span>`;
    document.getElementById("signoutLink")?.addEventListener("click", signOut);
  } else {
    authStatus.innerHTML = `<a href="#" id="loginLink">Sign in</a> to save results`;
    document.getElementById("loginLink")?.addEventListener("click", openAuth);
  }
}

function signOut(e) {
  e.preventDefault();
  accessToken = null;
  userEmail = null;
  localStorage.removeItem("hip_token");
  localStorage.removeItem("hip_email");
  historySection.classList.add("hidden");
  updateAuthUI();
}

// Modal
function openAuth(e) {
  if (e) e.preventDefault();
  isSignUp = false;
  authTitle.textContent = "Sign In";
  authSubmit.textContent = "Sign In";
  authToggleText.textContent = "Don't have an account?";
  authToggleLink.textContent = "Sign Up";
  authError.classList.add("hidden");
  authForm.reset();
  authModal.classList.remove("hidden");
}

loginLink?.addEventListener("click", openAuth);
modalClose?.addEventListener("click", () => authModal.classList.add("hidden"));
authModal?.addEventListener("click", (e) => { if (e.target === authModal) authModal.classList.add("hidden"); });

// Google OAuth
document.getElementById("googleSignIn")?.addEventListener("click", async () => {
  try {
    const resp = await fetch(`${API_BASE}/auth/oauth/google`);
    if (!resp.ok) { const err = await resp.json(); throw new Error(err.detail || "OAuth failed"); }
    const data = await resp.json();
    window.open(data.url, "google-oauth", "width=600,height=700");
  } catch (err) {
    authError.textContent = err.message;
    authError.classList.remove("hidden");
  }
});

// Listen for OAuth callback from popup
window.addEventListener("message", async (e) => {
  if (e.data?.type !== "oauth") return;
  const { token, email } = e.data;
  if (!token) return;
  if (token.length > 200) {
    // Implicit flow — token is access_token
    accessToken = token;
    userEmail = email || "Google user";
  } else {
    // PKCE flow — token is code, exchange it
    try {
      const resp = await fetch(`${API_BASE}/auth/exchange`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: token }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail);
      accessToken = data.access_token;
      userEmail = data.user || "Google user";
    } catch (err) {
      console.error("OAuth exchange failed", err);
      return;
    }
  }
  localStorage.setItem("hip_token", accessToken);
  localStorage.setItem("hip_email", userEmail);
  authModal.classList.add("hidden");
  updateAuthUI();
  loadHistory();
});

authToggleLink?.addEventListener("click", (e) => {
  e.preventDefault();
  isSignUp = !isSignUp;
  authTitle.textContent = isSignUp ? "Sign Up" : "Sign In";
  authSubmit.textContent = isSignUp ? "Sign Up" : "Sign In";
  authToggleText.textContent = isSignUp ? "Already have an account?" : "Don't have an account?";
  authToggleLink.textContent = isSignUp ? "Sign In" : "Sign Up";
  authError.classList.add("hidden");
});

authForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  authError.classList.add("hidden");
  authSubmit.disabled = true;
  authSubmit.textContent = "Please wait...";

  const email = authEmail.value;
  const password = authPassword.value;
  const endpoint = isSignUp ? `${API_BASE}/auth/signup` : `${API_BASE}/auth/signin`;

  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Auth failed");
    if (isSignUp) {
      authError.textContent = "Account created! Check your email to confirm.";
      authError.className = "form-error";
      authError.style.background = "#ecfdf5";
      authError.style.color = "#065f46";
      authError.classList.remove("hidden");
    } else {
      accessToken = data.access_token;
      userEmail = data.user;
      localStorage.setItem("hip_token", accessToken);
      localStorage.setItem("hip_email", userEmail);
      authModal.classList.add("hidden");
      updateAuthUI();
      loadHistory();
    }
  } catch (err) {
    authError.textContent = err.message;
    authError.className = "form-error";
    authError.classList.remove("hidden");
  }
  authSubmit.disabled = false;
  authSubmit.textContent = isSignUp ? "Sign Up" : "Sign In";
});

// --- Test selection ---
testGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".test-card");
  if (!card || card.classList.contains("disabled")) return;
  document.querySelectorAll(".test-card").forEach((c) => c.classList.remove("active"));
  card.classList.add("active");
  uploadSection.scrollIntoView({ behavior: "smooth" });
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
  savedBadge.classList.add("hidden");
  processingSection.classList.remove("hidden");
  resetSteps();
  await animateSteps();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const headers = {};
    if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;
    const resp = await fetch(`${API_BASE}/diagnose`, { method: "POST", body: formData, headers });
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
  card.style.borderLeftColor = p.class_id === 0 ? "var(--green)" : p.class_id <= 2 ? "var(--accent)" : "var(--red)";
  resultConfidence.style.color = p.class_id === 0 ? "var(--green)" : p.class_id <= 2 ? "var(--accent)" : "var(--red)";

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

  if (accessToken) savedBadge.classList.remove("hidden");
  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth" });
  if (accessToken) loadHistory();
}

// --- History ---
async function loadHistory() {
  if (!accessToken) return;
  try {
    const resp = await fetch(`${API_BASE}/history/`, { headers: { Authorization: `Bearer ${accessToken}` } });
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistory(data.screenings || []);
  } catch { /* silent */ }
}

function renderHistory(screenings) {
  if (!screenings.length) { historySection.classList.add("hidden"); return; }
  historySection.classList.remove("hidden");
  historyList.innerHTML = "";
  screenings.slice(0, 10).forEach((s) => {
    const pct = (s.primary_confidence * 100).toFixed(1);
    const div = document.createElement("div");
    div.className = "history-item";
    div.innerHTML = `
      <div class="history-item-left">
        <div class="hi-test">${s.test_type || "retinopathy"}</div>
        <div class="hi-diagnosis">${s.primary_diagnosis}</div>
        <div class="hi-date">${new Date(s.created_at).toLocaleString()}</div>
      </div>
      <div class="history-item-right" style="color:${s.primary_confidence > 0.9 ? "var(--green)" : "var(--accent)"}">${pct}%</div>
    `;
    historyList.appendChild(div);
  });
}
