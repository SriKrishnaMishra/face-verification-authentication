import os
import json
import cv2
import numpy as np

try:
    from deepface import DeepFace  # type: ignore
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), '..', 'storage', 'embeddings.json')


def _ensure_storage():
    dirpath = os.path.dirname(EMBEDDINGS_FILE)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def extract_embedding(image_path: str, model_name: str = 'VGG-Face'):
    """Extract embedding from an image.

    Uses DeepFace when available; otherwise falls back to a simple OpenCV-based
    histogram embedding for demo purposes (not production-accurate).
    """
    if DEEPFACE_AVAILABLE:
        reps = DeepFace.represent(img_path=image_path, model_name=model_name)
        if reps and len(reps) > 0:
            return np.array(reps[0]['embedding'], dtype=float)

    # Fallback: compute a simple color histogram as a lightweight embedding
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize for consistency
    img = cv2.resize(img, (160, 160))
    # 8-bin histogram per channel -> 24-dim vector
    hists = []
    for c in range(3):
        hist = cv2.calcHist([img], [c], None, [8], [0, 256]).flatten()
        hists.append(hist)
    emb = np.concatenate(hists).astype(float)
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def save_embedding(user_id: str, embedding):
    """Save a user's embedding to the embeddings.json file."""
    _ensure_storage()
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    else:
        data = {}

    # store as list for JSON compatibility
    data[user_id] = embedding if isinstance(embedding, list) else list(embedding)
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(data, f)


def load_embeddings():
    """Load all embeddings from the embeddings.json file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def verify_face(image_path: str, user_id: str, model_name: str = 'VGG-Face', threshold: float = 0.6):
    """Compare a new image to a stored embedding for verification.

    Returns (bool, similarity_score)
    """
    stored = load_embeddings().get(user_id)
    if not stored:
        return False, None

    new_embedding = extract_embedding(image_path, model_name)
    a = np.array(stored, dtype=float)
    b = np.array(new_embedding, dtype=float)
    if a.size == 0 or b.size == 0:
        return False, None
    similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    return similarity >= threshold, similarity


# Example usage:
# emb = extract_embedding('path/to/image.jpg')
# save_embedding('user123', emb)
# ok, score = verify_face('path/to/another.jpg', 'user123')
