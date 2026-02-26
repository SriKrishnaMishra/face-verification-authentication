const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCam = document.getElementById('startCam');
const stopCam = document.getElementById('stopCam');
const registerBtn = document.getElementById('registerBtn');
const verifyBtn = document.getElementById('verifyBtn');
const stopVerifyBtn = document.getElementById('stopVerifyBtn');
const userIdInput = document.getElementById('userId');
const resultDiv = document.getElementById('result');
const scoreDiv = document.getElementById('score');

let stream = null;
let verifyInterval = null;

const API_BASE = window.location.origin; // same origin

function logStatus(text) {
  resultDiv.textContent = `Status: ${text}`;
}

function setScore(score) {
  scoreDiv.textContent = `Score: ${score}`;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    logStatus('Camera started');
    startCam.disabled = true;
    stopCam.disabled = false;
    registerBtn.disabled = false;
    verifyBtn.disabled = false;
  } catch (err) {
    console.error(err);
    logStatus('Failed to start camera: ' + err.message);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  startCam.disabled = false;
  stopCam.disabled = true;
  registerBtn.disabled = true;
  verifyBtn.disabled = true;
  stopVerifyBtn.disabled = true;
  logStatus('Camera stopped');
}

function grabFrameAsDataURL() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.92);
}

async function postJSON(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

async function registerFace() {
  const userId = userIdInput.value.trim();
  if (!userId) {
    alert('Enter a user ID first');
    return;
  }
  const image = grabFrameAsDataURL();
  logStatus('Registering...');
  try {
    const data = await postJSON('/register', { user_id: userId, image });
    logStatus(`Registered user: ${data.user_id}`);
  } catch (e) {
    console.error(e);
    logStatus('Register failed');
  }
}

function startVerify() {
  const userId = userIdInput.value.trim();
  if (!userId) {
    alert('Enter a user ID first');
    return;
  }
  if (verifyInterval) return;
  stopVerifyBtn.disabled = false;
  logStatus('Verifying...');
  verifyInterval = setInterval(async () => {
    const image = grabFrameAsDataURL();
    try {
      const data = await postJSON('/verify', { user_id: userId, image });
      logStatus(data.verified ? 'Verified ✅' : 'Not Verified ❌');
      setScore(`${data.score} (thr=${data.threshold})`);
    } catch (e) {
      console.error(e);
      logStatus('Verify error (check console)');
    }
  }, 1500); // every 1.5s
}

function stopVerify() {
  if (verifyInterval) {
    clearInterval(verifyInterval);
    verifyInterval = null;
  }
  stopVerifyBtn.disabled = true;
  logStatus('Verification stopped');
}

startCam.addEventListener('click', startCamera);
stopCam.addEventListener('click', stopCamera);
registerBtn.addEventListener('click', registerFace);
verifyBtn.addEventListener('click', startVerify);
stopVerifyBtn.addEventListener('click', stopVerify);
