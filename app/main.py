import os
import json
import base64
import threading
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None
from passlib.context import CryptContext
try:
    import torch
    from PIL import Image
    from facenet_pytorch import InceptionResnetV1, MTCNN
except Exception:
    torch = None
    Image = None
    InceptionResnetV1 = None
    MTCNN = None
from jose import JWTError, jwt
import pyotp
import qrcode
import re
from fastapi import Request

# Security imports
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("Security modules not available")

# Import optimization modules
try:
    from realtime_optimization import (
        EmbeddingCache, ImagePreprocessor, FaceMatcher,
        RealTimeVerificationPipeline, BatchVerifier
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("Real-time optimization module not available")

# Import advanced real-time engine
try:
    from advanced_realtime import verification_engine, manager
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    logging.warning("Advanced real-time module not available")

# Configuration (overridable via env vars)
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet512")  # alternatives: "ArcFace", "VGG-Face"
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv")  # alternatives: "mtcnn", "retinaface" (downloads weights)
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.7"))  # tweak after testing with your data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
STORAGE_DIR = os.path.join(ROOT_DIR, "storage")
STORAGE_PATH = os.path.join(STORAGE_DIR, "embeddings.json")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

store_lock = threading.Lock()

# Real-time optimization components
embedding_cache = None
realtime_pipeline = None
face_matcher = None

if OPTIMIZATION_AVAILABLE:
    embedding_cache = EmbeddingCache(max_size=128)
    face_matcher = FaceMatcher(threshold=COSINE_THRESHOLD)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = FastAPI(title="Face Verification API", version="0.2.0")

# CORS for local development and simple hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
if SECURITY_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# Mount static files for frontend (for basic demo; React app runs in /web in dev)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image: str = Field(..., description="Base64 data URL or raw base64 of an image frame")
    reset: bool = Field(False, description="If true, clears previous samples before adding this one")


class VerifyRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    image: str = Field(..., description="Base64 data URL or raw base64 of an image frame")


class VerifyResponse(BaseModel):
    verified: bool
    score: float
    threshold: float
    user_id: str
    model: str
    samples_used: int


class AuthRegister(BaseModel):
    username: str = Field(..., min_length=3, description="Username or email address")
    password: str = Field(..., min_length=6)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class TOTPSetupResponse(BaseModel):
    secret: str
    qr_code: str  # Base64 encoded PNG image
    backup_codes: List[str]


class TOTPVerifyRequest(BaseModel):
    username: str
    totp_code: str = Field(..., min_length=6, max_length=6)


class TOTPEnableRequest(BaseModel):
    username: str
    totp_code: str = Field(..., min_length=6, max_length=6)


class TOTPStatusResponse(BaseModel):
    is_enabled: bool
    backup_codes_count: int


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


class VerifyRealtimeResponse(BaseModel):
    verified: bool
    similarity: float
    confidence: float
    threshold: float
    user_id: str
    quality_metrics: Dict
    processing_time_ms: float
    cached: bool
    margin: float


class BatchVerifyRequest(BaseModel):
    image: str = Field(..., description="Base64 image")
    user_ids: List[str] = Field(..., description="List of user IDs to match against")


class BatchVerifyResponse(BaseModel):
    results: List[Dict]
    processing_time_ms: float


class PerformanceStatsResponse(BaseModel):
    total_verifications: int
    success_rate: float
    average_processing_time_ms: float
    average_match_score: float
    cache_stats: Optional[Dict]


class CacheInfoResponse(BaseModel):
    cache_size: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    users_cached: List[str]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Auth config
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-to-a-secure-random-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def validate_password_strength(password: str) -> str:
    """Validate password strength and return error message or empty string."""
    if len(password) < 8:
        return "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return "Password must contain at least one digit"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character"
    return ""


def validate_email_or_username(username: str) -> bool:
    """Validate if input is a valid email or username."""
    # Simple email regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, username)) or len(username) >= 3


def check_account_lockout(username: str) -> bool:
    """Check if account is locked due to too many failed attempts."""
    auth = load_auth_store()
    user = auth.get(username, {})
    failed_attempts = user.get("failed_attempts", 0)
    last_attempt = user.get("last_failed_attempt")
    
    # Reset if enough time has passed (15 minutes)
    if last_attempt:
        try:
            last_attempt_dt = datetime.fromisoformat(last_attempt)
            if (datetime.utcnow() - last_attempt_dt).seconds > 900:
                user["failed_attempts"] = 0
                save_auth_store(auth)
                return False
        except (ValueError, TypeError):
            pass
    
    return failed_attempts >= 5  # Lock after 5 failed attempts


def record_failed_attempt(username: str):
    """Record a failed login attempt."""
    with store_lock:
        auth = load_auth_store()
        if username in auth:
            user = auth[username]
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1
            user["last_failed_attempt"] = datetime.utcnow().isoformat()
            save_auth_store(auth)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def load_auth_store() -> Dict[str, Any]:
    # Store auth data as part of the same JSON file under 'auth' key for simplicity
    store = load_store()
    return store.get("auth", {})


def save_auth_store(auth: Dict[str, Any]) -> None:
    store = load_store()
    store["auth"] = auth
    save_store(store)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    auth = load_auth_store()
    user = auth.get(username)
    if not user:
        return None
    if not verify_password(password, user.get("password_hash", "")):
        return None
    return user


async def get_current_username(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data.username


# --------- TOTP (2FA) Utilities ---------

def generate_totp_secret(username: str) -> str:
    """Generate a random TOTP secret."""
    return pyotp.random_base32()


def generate_qr_code(username: str, secret: str) -> str:
    """Generate a QR code for TOTP setup and return as base64."""
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(username, issuer_name="Face Verification")
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    qr_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    return qr_base64


def generate_backup_codes(count: int = 10) -> List[str]:
    """Generate backup codes for account recovery."""
    import secrets
    codes = [f"{secrets.token_hex(4).upper()}" for _ in range(count)]
    return codes


def verify_totp(secret: str, totp_code: str, window: int = 1) -> bool:
    """Verify a TOTP code with a time window tolerance."""
    totp = pyotp.TOTP(secret)
    return totp.verify(totp_code, valid_window=window)


def use_backup_code(backup_codes: List[str], code: str) -> tuple[bool, List[str]]:
    """Check if a backup code is valid and remove it from the list."""
    if code in backup_codes:
        updated_codes = [c for c in backup_codes if c != code]
        return True, updated_codes
    return False, backup_codes


# --------- Utilities ---------

def decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 string (optionally a data URL) to a BGR OpenCV image."""
    if "," in data and data.strip().startswith("data:"):
        # data URL format: data:image/jpeg;base64,XXXXX
        data = data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image input")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec) + eps
    return vec / norm


def get_embedding(image_bgr: np.ndarray) -> np.ndarray:
    """Return L2-normalized embedding vector for the largest detected face in the image."""
    # Prefer DeepFace if available (keeps backward compatibility)
    if DeepFace is not None:
        try:
            reps = DeepFace.represent(
                img_path=image_bgr,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Face detection/representation failed: {str(e)}")

        if not isinstance(reps, list) or len(reps) == 0:
            raise HTTPException(status_code=422, detail="No face embeddings returned")

        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        return l2_normalize(emb)

    # Fallback to facenet-pytorch if available
    if InceptionResnetV1 is None or MTCNN is None or torch is None:
        raise HTTPException(status_code=500, detail="No embedding backend available. Install deepface or facenet-pytorch.")

    # Convert BGR (OpenCV) to RGB PIL Image
    try:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert image: {str(e)}")

    # Detect and crop face
    global face_detector, face_embedder, torch_device
    if 'face_detector' not in globals() or face_detector is None:
        raise HTTPException(status_code=500, detail="Face detector not initialized on server")
    if 'face_embedder' not in globals() or face_embedder is None:
        raise HTTPException(status_code=500, detail="Embedding model not initialized on server")

    try:
        # MTCNN returns a tensor of shape (3, H, W) or a batch; get single face
        face_t = face_detector(pil)
        if face_t is None:
            raise HTTPException(status_code=422, detail="No face detected")
        if face_t.dim() == 3:
            face_t = face_t.unsqueeze(0)
        face_t = face_t.to(torch_device)
        with torch.no_grad():
            emb_t = face_embedder(face_t)
        emb = emb_t[0].cpu().numpy().astype(np.float32)
        return l2_normalize(emb)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


def load_store() -> Dict[str, Any]:
    if not os.path.exists(STORAGE_PATH):
        return {}
    with open(STORAGE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_store(store: Dict[str, Any]) -> None:
    tmp_path = STORAGE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STORAGE_PATH)


def get_users_map(store: Dict[str, Any]) -> Dict[str, Any]:
    """Support both legacy format and new 'users' container."""
    if "users" in store and isinstance(store["users"], dict):
        return store["users"]
    # Legacy: top-level user_id -> {embedding:..., model:...}
    return {k: v for k, v in store.items() if isinstance(v, dict) and ("embedding" in v or "samples" in v)}


def set_users_map(store: Dict[str, Any], users: Dict[str, Any]) -> Dict[str, Any]:
    store_new = {**store}
    store_new["users"] = users
    # Optionally clean legacy keys
    for k in list(store_new.keys()):
        if k not in ("users",):
            if isinstance(store_new[k], dict) and ("embedding" in store_new[k] or "samples" in store_new[k]):
                del store_new[k]
    return store_new


def compute_mean_embedding(samples: List[List[float]]) -> np.ndarray:
    if not samples:
        raise HTTPException(status_code=400, detail="No samples available")
    arr = np.array(samples, dtype=np.float32)
    # Ensure each sample is normalized
    arr = np.array([l2_normalize(x) for x in arr])
    mean = np.mean(arr, axis=0)
    return l2_normalize(mean)


def user_sample_count(entry: Dict[str, Any]) -> int:
    if entry is None:
        return 0
    if "samples" in entry and isinstance(entry["samples"], list):
        return len(entry["samples"])
    if "embedding" in entry:
        return 1
    return 0


# --------- Startup -----------
@app.on_event("startup")
def warmup_models():
    # pre-load the embedding model to reduce first-hit latency
    try:
        DeepFace.build_model(MODEL_NAME)
    except Exception:
        # let requests attempt to build on demand if this fails
        pass


# --------- Routes -----------
@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "message": "Frontend not found yet. Use the React app in /web during development.",
        "api_endpoints": ["/register", "/verify", "/users", "/health"],
    })


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "detector": DETECTOR_BACKEND}


@app.get("/config")
def config():
    return {"model": MODEL_NAME, "detector": DETECTOR_BACKEND, "threshold": COSINE_THRESHOLD}


@app.get("/users")
def list_users():
    with store_lock:
        store = load_store()
        users = get_users_map(store)
        data = []
        for uid, entry in users.items():
            data.append({
                "user_id": uid,
                "samples": user_sample_count(entry),
                "model": entry.get("model", MODEL_NAME),
            })
    return {"users": data}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    with store_lock:
        store = load_store()
        users = get_users_map(store)
        if user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        entry = users[user_id]
    return {
        "user_id": user_id,
        "samples": user_sample_count(entry),
        "model": entry.get("model", MODEL_NAME),
    }


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    with store_lock:
        store = load_store()
        users = get_users_map(store)
        if user_id in users:
            del users[user_id]
            store = set_users_map(store, users)
            save_store(store)
        else:
            raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "user_id": user_id}


@app.post("/register")
def register_face(req: RegisterRequest):
    logging.info(f"REGISTER user_id={req.user_id} reset={req.reset}")
    img = decode_base64_image(req.image)
    emb = get_embedding(img)
    logging.info(f"REGISTER embedding_dim={emb.shape[0]}")
    emb_list = emb.tolist()

    with store_lock:
        store = load_store()
        users = get_users_map(store)
        entry = users.get(req.user_id)
        if entry is None or req.reset:
            entry = {"samples": [], "model": MODEL_NAME}
        # Backward compatibility: if legacy single embedding exists, move to samples
        if "embedding" in entry and "samples" not in entry:
            entry = {"samples": [entry["embedding"]], "model": entry.get("model", MODEL_NAME)}
        entry.setdefault("samples", [])
        entry.setdefault("model", MODEL_NAME)
        entry["samples"].append(emb_list)
        users[req.user_id] = entry
        store = set_users_map(store, users)
        save_store(store)

    return {"status": "registered", "user_id": req.user_id, "samples": len(entry["samples"]), "model": entry["model"]}


# --------- Auth routes ---------
@app.post("/auth/register", response_model=Dict[str, str])
def auth_register(req: AuthRegister):
    username = req.username
    password = req.password
    
    # Validate username/email format
    if not validate_email_or_username(username):
        raise HTTPException(status_code=400, detail="Invalid username or email format")
    
    # Validate password strength
    password_error = validate_password_strength(password)
    if password_error:
        raise HTTPException(status_code=400, detail=password_error)
    
    with store_lock:
        auth = load_auth_store()
        if username in auth:
            raise HTTPException(status_code=400, detail="User already exists")
        auth[username] = {"password_hash": get_password_hash(password)}
        save_auth_store(auth)
    return {"status": "created", "username": username}


@app.post("/auth/token", response_model=Token)
@limiter.limit("5/minute") if SECURITY_AVAILABLE else lambda x: x
def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Validate username/email format
    if not validate_email_or_username(form_data.username):
        raise HTTPException(status_code=400, detail="Invalid username or email format")
    
    # Check account lockout
    if check_account_lockout(form_data.username):
        raise HTTPException(status_code=429, detail="Account temporarily locked due to too many failed attempts")
    
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        record_failed_attempt(form_data.username)
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # Check if 2FA is enabled
    if user.get("totp_enabled", False):
        raise HTTPException(status_code=403, detail="2FA required", headers={"X-2FA-Required": "true"})
    
    # Reset failed attempts on successful login
    with store_lock:
        auth = load_auth_store()
        if form_data.username in auth:
            auth[form_data.username]["failed_attempts"] = 0
            save_auth_store(auth)
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/token-with-2fa", response_model=Token)
def login_with_2fa(form_data: OAuth2PasswordRequestForm = Depends(), totp_code: Optional[str] = None):
    """Login with TOTP verification for accounts with 2FA enabled."""
    # Validate username/email format
    if not validate_email_or_username(form_data.username):
        raise HTTPException(status_code=400, detail="Invalid username or email format")
    
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    is_2fa_enabled = user.get("totp_enabled", False)
    
    if is_2fa_enabled:
        if not totp_code:
            raise HTTPException(status_code=403, detail="TOTP code required")
        
        secret = user.get("totp_secret")
        if not secret or not verify_totp(secret, totp_code):
            # Try backup code
            backup_codes = user.get("backup_codes", [])
            is_valid, updated_codes = use_backup_code(backup_codes, totp_code)
            
            if not is_valid:
                raise HTTPException(status_code=401, detail="Invalid TOTP code or backup code")
            
            # Update backup codes
            with store_lock:
                auth = load_auth_store()
                auth[form_data.username]["backup_codes"] = updated_codes
                save_auth_store(auth)
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/totp/setup", response_model=TOTPSetupResponse)
def setup_totp(username: str = Depends(get_current_username)):
    """Generate TOTP secret and QR code for 2FA setup."""
    secret = generate_totp_secret(username)
    qr_code_base64 = generate_qr_code(username, secret)
    backup_codes = generate_backup_codes(10)
    
    return TOTPSetupResponse(
        secret=secret,
        qr_code=qr_code_base64,
        backup_codes=backup_codes,
    )


@app.post("/auth/totp/enable")
def enable_totp(req: TOTPEnableRequest, username: str = Depends(get_current_username)):
    """Enable 2FA by verifying the TOTP code with the secret."""
    if req.username != username:
        raise HTTPException(status_code=403, detail="Cannot modify other users")
    
    # The client should send the secret they received from /setup
    # For security, we need them to provide the secret and verify it
    # In production, you'd store the temporary secret and verify against it
    
    with store_lock:
        auth = load_auth_store()
        if username not in auth:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = auth[username]
        if not verify_totp(user.get("totp_secret_temp", ""), req.totp_code):
            raise HTTPException(status_code=401, detail="Invalid TOTP code")
        
        # Move temp secret to active secret and enable 2FA
        user["totp_secret"] = user.get("totp_secret_temp", "")
        user["totp_enabled"] = True
        user["backup_codes"] = user.get("backup_codes_temp", [])
        user.pop("totp_secret_temp", None)
        user.pop("backup_codes_temp", None)
        
        save_auth_store(auth)
    
    return {"status": "2FA enabled", "username": username}


@app.post("/auth/totp/prepare")
def prepare_totp(username: str = Depends(get_current_username)):
    """Store temporary TOTP secret for later confirmation."""
    secret = generate_totp_secret(username)
    qr_code_base64 = generate_qr_code(username, secret)
    backup_codes = generate_backup_codes(10)
    
    # Temporarily store the secret (in production, use Redis or similar)
    with store_lock:
        auth = load_auth_store()
        if username not in auth:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = auth[username]
        user["totp_secret_temp"] = secret
        user["backup_codes_temp"] = backup_codes
        save_auth_store(auth)
    
    return TOTPSetupResponse(
        secret=secret,
        qr_code=qr_code_base64,
        backup_codes=backup_codes,
    )


@app.post("/auth/totp/disable")
def disable_totp(totp_code: str, username: str = Depends(get_current_username)):
    """Disable 2FA (requires current TOTP verification)."""
    with store_lock:
        auth = load_auth_store()
        if username not in auth:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = auth[username]
        if not user.get("totp_enabled", False):
            raise HTTPException(status_code=400, detail="2FA is not enabled")
        
        secret = user.get("totp_secret")
        if not secret or not verify_totp(secret, totp_code):
            raise HTTPException(status_code=401, detail="Invalid TOTP code")
        
        user["totp_enabled"] = False
        user.pop("totp_secret", None)
        user.pop("backup_codes", None)
        
        save_auth_store(auth)
    
    return {"status": "2FA disabled", "username": username}


@app.get("/auth/totp/status", response_model=TOTPStatusResponse)
def get_totp_status(username: str = Depends(get_current_username)):
    """Check if 2FA is enabled and get backup code count."""
    with store_lock:
        auth = load_auth_store()
        if username not in auth:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = auth[username]
        is_enabled = user.get("totp_enabled", False)
        backup_count = len(user.get("backup_codes", []))
    
    return TOTPStatusResponse(
        is_enabled=is_enabled,
        backup_codes_count=backup_count,
    )


@app.get("/auth/me")
def read_users_me(username: str = Depends(get_current_username)):
    return {"username": username}


@app.post("/auth/refresh", response_model=TokenPair)
def refresh_access_token(req: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    try:
        payload = jwt.decode(req.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        username = payload.get("sub")
        
        # Create new token pair
        access_token = create_access_token({"sub": username})
        refresh_token = create_access_token(
            {"sub": username, "type": "refresh"}, 
            expires_delta=timedelta(days=7)
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@app.post("/auth/forgot-password")
def forgot_password(req: PasswordResetRequest):
    """Send password reset email."""
    auth = load_auth_store()
    username = req.email  # Assuming email is used as username for simplicity
    
    if username not in auth:
        # Don't reveal if user exists or not for security
        return {"message": "If the account exists, a reset email has been sent"}
    
    reset_token = create_access_token(
        {"sub": username, "type": "reset"}, 
        timedelta(hours=1)
    )
    
    # Store reset token
    with store_lock:
        auth[username]["reset_token"] = reset_token
        save_auth_store(auth)
    
    # In a real app, send email here
    reset_link = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={reset_token}"
    logging.info(f"Password reset link for {username}: {reset_link}")
    
    return {"message": "If the account exists, a reset email has been sent"}


@app.post("/auth/reset-password")
def reset_password(req: PasswordResetConfirm):
    """Reset password using reset token."""
    try:
        payload = jwt.decode(req.token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "reset":
            raise HTTPException(status_code=401, detail="Invalid token")
        username = payload.get("sub")
        
        # Validate new password
        password_error = validate_password_strength(req.new_password)
        if password_error:
            raise HTTPException(status_code=400, detail=password_error)
        
        with store_lock:
            auth = load_auth_store()
            if username not in auth or auth[username].get("reset_token") != req.token:
                raise HTTPException(status_code=401, detail="Invalid reset token")
            
            auth[username]["password_hash"] = get_password_hash(req.new_password)
            auth[username].pop("reset_token", None)
            auth[username]["failed_attempts"] = 0  # Reset failed attempts
            save_auth_store(auth)
        
        return {"message": "Password reset successfully"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/embed")
def embed_image(req: VerifyRequest):
    """Return embedding for an input image (base64)."""
    img = decode_base64_image(req.image)
    emb = get_embedding(img)
    return {"embedding": emb.tolist(), "model": MODEL_NAME}


@app.post("/verify", response_model=VerifyResponse)
def verify_face(req: VerifyRequest):
    logging.info(f"VERIFY user_id={req.user_id}")
    with store_lock:
        store = load_store()
        users = get_users_map(store)
        if req.user_id not in users:
            raise HTTPException(status_code=404, detail="User not registered")
        entry = users[req.user_id]

    # Get gallery embedding(s)
    gallery_samples: List[List[float]] = []
    if "samples" in entry and entry["samples"]:
        gallery_samples = entry["samples"]
    elif "embedding" in entry:
        gallery_samples = [entry["embedding"]]
    else:
        raise HTTPException(status_code=400, detail="User has no samples")

    probe_img = decode_base64_image(req.image)
    probe_emb = get_embedding(probe_img)

    # Compare against mean of gallery samples
    gallery_mean = compute_mean_embedding(gallery_samples)
    score = float(np.dot(gallery_mean, probe_emb))  # cosine similarity (normalized)
    verified = score >= COSINE_THRESHOLD
    logging.info(f"VERIFY score={score:.4f} threshold={COSINE_THRESHOLD} samples={len(gallery_samples)} verified={verified}")

    return VerifyResponse(
        verified=verified,
        score=round(score, 4),
        threshold=COSINE_THRESHOLD,
        user_id=req.user_id,
        model=MODEL_NAME,
        samples_used=len(gallery_samples),
    )


# --------- Optimized Real-time Endpoints ---------

@app.post("/verify-realtime", response_model=VerifyRealtimeResponse)
def verify_face_realtime(req: VerifyRequest):
    """
    Optimized real-time face verification with:
    - Image quality checking
    - Embedding caching
    - Confidence scoring
    """
    import time
    
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Real-time optimization not available")
    
    start_time = time.time()
    
    logging.info(f"VERIFY-REALTIME user_id={req.user_id}")
    
    try:
        # Get user's gallery embeddings
        with store_lock:
            store = load_store()
            users = get_users_map(store)
            if req.user_id not in users:
                raise HTTPException(status_code=404, detail="User not registered")
            entry = users[req.user_id]
        
        gallery_samples: List[List[float]] = []
        if "samples" in entry and entry["samples"]:
            gallery_samples = entry["samples"]
        elif "embedding" in entry:
            gallery_samples = [entry["embedding"]]
        else:
            raise HTTPException(status_code=400, detail="User has no samples")
        
        gallery_mean = compute_mean_embedding(gallery_samples)
        
        # Decode and preprocess image
        probe_img = decode_base64_image(req.image)
        preprocessor = ImagePreprocessor()
        processed_img, quality_metrics = preprocessor.preprocess(probe_img, quality_check=True)
        
        # Get embedding
        probe_emb = get_embedding(processed_img)
        
        # Match with confidence
        match_result = face_matcher.match_with_confidence(probe_emb, gallery_mean)
        
        processing_time = time.time() - start_time
        
        logging.info(
            f"VERIFY-REALTIME user_id={req.user_id} similarity={match_result['similarity']} "
            f"confidence={match_result['confidence']} time={processing_time:.3f}s"
        )
        
        return VerifyRealtimeResponse(
            verified=match_result["verified"],
            similarity=match_result["similarity"],
            confidence=match_result["confidence"],
            threshold=COSINE_THRESHOLD,
            user_id=req.user_id,
            quality_metrics=quality_metrics,
            processing_time_ms=round(processing_time * 1000, 2),
            cached=False,
            margin=match_result["margin"]
        )
    
    except Exception as e:
        logging.error(f"VERIFY-REALTIME error: {str(e)}")
        raise


@app.post("/verify-batch", response_model=BatchVerifyResponse)
def verify_batch(req: BatchVerifyRequest):
    """
    Batch verification against multiple users
    Useful for identifying unknown faces
    """
    import time
    
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Real-time optimization not available")
    
    start_time = time.time()
    
    logging.info(f"VERIFY-BATCH against {len(req.user_ids)} users")
    
    try:
        # Get probe embedding
        probe_img = decode_base64_image(req.image)
        probe_emb = get_embedding(probe_img)
        
        # Get all candidate embeddings
        with store_lock:
            store = load_store()
            users = get_users_map(store)
        
        candidates = {}
        for user_id in req.user_ids:
            if user_id in users:
                entry = users[user_id]
                gallery_samples: List[List[float]] = []
                if "samples" in entry and entry["samples"]:
                    gallery_samples = entry["samples"]
                elif "embedding" in entry:
                    gallery_samples = [entry["embedding"]]
                
                if gallery_samples:
                    gallery_mean = compute_mean_embedding(gallery_samples)
                    candidates[user_id] = gallery_mean
        
        if not candidates:
            raise HTTPException(status_code=400, detail="No valid candidates")
        
        # Batch match
        results = BatchVerifier.verify_batch(
            probe_img,
            probe_emb,
            candidates,
            face_matcher,
            threshold=COSINE_THRESHOLD
        )
        
        processing_time = time.time() - start_time
        
        logging.info(f"VERIFY-BATCH completed in {processing_time:.3f}s")
        
        return BatchVerifyResponse(
            results=results,
            processing_time_ms=round(processing_time * 1000, 2)
        )
    
    except Exception as e:
        logging.error(f"VERIFY-BATCH error: {str(e)}")
        raise


@app.get("/performance-stats", response_model=PerformanceStatsResponse)
def get_performance_stats():
    """
    Get real-time performance statistics
    Includes cache hit rate, average processing time, and verification success rate
    """
    if not OPTIMIZATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Real-time optimization not available")
    
    # Get cache stats
    cache_stats = None
    if embedding_cache:
        cache_stats = embedding_cache.get_stats()
    
    # Calculate stats from matcher history
    history = face_matcher.get_match_history(100)
    
    if history:
        success_count = sum(1 for m in history if m["verified"])
        success_rate = success_count / len(history)
        avg_score = np.mean([m["similarity"] for m in history])
    else:
        success_rate = 0.0
        avg_score = 0.0
    
    return PerformanceStatsResponse(
        total_verifications=len(face_matcher.get_match_history(10000)),
        success_rate=round(success_rate, 4),
        average_processing_time_ms=0.0,  # Would track actual times
        average_match_score=round(avg_score, 4),
        cache_stats=cache_stats
    )


@app.get("/cache-info", response_model=CacheInfoResponse)
def get_cache_info():
    """Get current embedding cache information"""
    if not OPTIMIZATION_AVAILABLE or not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    stats = embedding_cache.get_stats()
    return CacheInfoResponse(**stats)


@app.delete("/cache/{user_id}")
def clear_user_cache(user_id: str):
    """Clear cache for specific user (call when updating embeddings)"""
    if not OPTIMIZATION_AVAILABLE or not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    embedding_cache.clear(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.delete("/cache")
def clear_all_cache():
    """Clear entire embedding cache"""
    if not OPTIMIZATION_AVAILABLE or not embedding_cache:
        raise HTTPException(status_code=503, detail="Cache not available")
    
    embedding_cache.clear_all()
    return {"status": "cleared", "message": "All embeddings cleared from cache"}


@app.get("/optimization-status")
def optimization_status():
    """Check if real-time optimization is enabled"""
    return {
        "optimization_available": OPTIMIZATION_AVAILABLE,
        "features": {
            "caching": embedding_cache is not None,
            "batch_verification": True,
            "performance_monitoring": True,
            "image_quality_check": True
        }
    }


@app.get("/system/stats")
def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get verification stats from performance monitor
        verification_stats = {
            "total": 0,
            "success_rate": 0.0,
            "average_processing_time_ms": 0.0
        }

        if OPTIMIZATION_AVAILABLE and hasattr(verification_engine, 'performance_stats'):
            stats = verification_engine.performance_stats
            verification_stats.update({
                "total": stats.get("total_verifications", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "average_processing_time_ms": stats.get("avg_processing_time", 0.0)
            })

        # Check available models and capabilities
        models = {
            "deepface_available": DeepFace is not None,
            "facenet_available": torch is not None and InceptionResnetV1 is not None,
            "mtcnn_available": torch is not None and MTCNN is not None,
            "gpu_available": torch is not None and torch.cuda.is_available(),
            "faiss_available": ADVANCED_AVAILABLE and hasattr(verification_engine, 'datastore')
        }

        # Get active sessions
        active_sessions = []
        if ADVANCED_AVAILABLE:
            active_sessions = manager.get_active_sessions()

        return {
            "verifications": verification_stats,
            "models": models,
            "active_sessions_count": len(active_sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logging.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


@app.get("/datastore/stats")
def get_datastore_stats():
    """Get datastore statistics"""
    try:
        if not ADVANCED_AVAILABLE or not hasattr(verification_engine, 'datastore'):
            return {
                "total_users": 0,
                "total_samples": 0,
                "indexing_enabled": False,
                "embedding_dim": 0,
                "avg_samples_per_user": 0.0
            }

        datastore = verification_engine.datastore
        stats = datastore.get_stats()

        return {
            "total_users": stats.get("total_users", 0),
            "total_samples": stats.get("total_samples", 0),
            "indexing_enabled": stats.get("indexing_enabled", False),
            "embedding_dim": stats.get("embedding_dim", 0),
            "avg_samples_per_user": stats.get("avg_samples_per_user", 0.0)
        }
    except Exception as e:
        logging.error(f"Error getting datastore stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get datastore statistics")


@app.get("/sessions/active")
def get_active_sessions():
    """Get information about active verification sessions"""
    try:
        if not ADVANCED_AVAILABLE:
            return {"active_sessions": []}

        sessions = manager.get_active_sessions()
        return {"active_sessions": sessions}
    except Exception as e:
        logging.error(f"Error getting active sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active sessions")


# --------- Advanced Real-time Endpoints ---------

class StreamVerifyRequest(BaseModel):
    session_id: str
    image: str = Field(..., description="Base64 image data")
    user_id: Optional[str] = Field(None, description="User to verify against (optional)")


class IdentifyRequest(BaseModel):
    image: str = Field(..., description="Base64 image data")
    max_candidates: int = Field(5, description="Maximum candidates to return")


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str]
    created: str
    frames_processed: int


@app.post("/stream/start-session")
def start_verification_session(user_id: Optional[str] = None) -> SessionResponse:
    """Start a new real-time verification session"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    session_id = verification_engine.create_session(user_id)
    session = verification_engine.active_sessions[session_id]

    return SessionResponse(
        session_id=session_id,
        user_id=session['user_id'],
        created=session['created'].isoformat(),
        frames_processed=0
    )


@app.delete("/stream/end-session/{session_id}")
def end_verification_session(session_id: str):
    """End a verification session"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    verification_engine.end_session(session_id)
    return {"status": "ended", "session_id": session_id}


@app.post("/stream/verify-frame")
async def verify_frame_stream(req: StreamVerifyRequest):
    """Verify a single frame in a streaming session"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    if req.session_id not in verification_engine.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await verification_engine.verify_frame(req.session_id, req.image)
    return result


@app.post("/identify")
async def identify_unknown_face(req: IdentifyRequest):
    """Identify an unknown face against all users in datastore"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    result = await verification_engine.identify_unknown(req.image, req.max_candidates)
    return result


@app.get("/datastore/stats")
def get_datastore_stats():
    """Get face datastore statistics"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    return verification_engine.datastore.get_stats()


@app.get("/system/stats")
def get_system_stats():
    """Get comprehensive system statistics"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    return verification_engine.get_stats()


@app.get("/sessions/active")
def get_active_sessions():
    """Get information about active verification sessions"""
    if not ADVANCED_AVAILABLE:
        raise HTTPException(status_code=503, detail="Advanced real-time features not available")

    sessions = []
    for session_id, session_data in verification_engine.active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "user_id": session_data.get('user_id'),
            "created": session_data['created'].isoformat(),
            "frames_processed": session_data.get('frames_processed', 0),
            "last_activity": session_data['last_activity'].isoformat(),
            "verifications_count": len(session_data.get('verifications', []))
        })

    return {"active_sessions": sessions, "count": len(sessions)}


# --------- WebSocket Endpoints for Real-time Streaming ---------

@app.websocket("/ws/verify/{session_id}")
async def websocket_verify(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time face verification streaming"""
    if not ADVANCED_AVAILABLE:
        await websocket.close(code=1008, reason="Advanced features not available")
        return

    await manager.connect(websocket, session_id)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "session_started",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        while True:
            # Receive frame data
            data = await websocket.receive_json()

            if data.get("type") == "frame":
                # Process frame
                result = await verification_engine.verify_frame(session_id, data["image"])

                # Send result back
                await websocket.send_json({
                    "type": "verification_result",
                    **result
                })

            elif data.get("type") == "heartbeat":
                # Respond to heartbeat
                await websocket.send_json({
                    "type": "heartbeat_response",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif data.get("type") == "end_session":
                # End session
                verification_engine.end_session(session_id)
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id
                })
                break

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        verification_engine.end_session(session_id)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        manager.disconnect(session_id)
        verification_engine.end_session(session_id)


@app.websocket("/ws/identify")
async def websocket_identify(websocket: WebSocket):
    """WebSocket endpoint for real-time face identification"""
    if not ADVANCED_AVAILABLE:
        await websocket.close(code=1008, reason="Advanced features not available")
        return

    session_id = f"identify_{id(websocket)}"  # Temporary session ID
    await manager.connect(websocket, session_id)

    try:
        await websocket.send_json({
            "type": "identification_ready",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        while True:
            data = await websocket.receive_json()

            if data.get("type") == "identify":
                result = await verification_engine.identify_unknown(data["image"])

                await websocket.send_json({
                    "type": "identification_result",
                    **result
                })

            elif data.get("type") == "heartbeat":
                await websocket.send_json({
                    "type": "heartbeat_response",
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logging.error(f"Identification WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        manager.disconnect(session_id)


if __name__ == "__main__":
    # For local development: python app/main.py
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
