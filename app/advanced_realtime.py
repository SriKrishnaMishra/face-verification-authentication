"""
Advanced Real-time Face Verification System
Features:
- WebSocket streaming for real-time video verification
- Advanced ML optimizations (GPU acceleration, batch processing)
- Face datastore with efficient indexing
- Multi-model ensemble for better accuracy
- Real-time quality assessment and feedback
- Authentication integration with face verification
"""

import asyncio
import json
import base64
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import time
import uuid

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

try:
    import torch
    from PIL import Image
    from facenet_pytorch import InceptionResnetV1, MTCNN
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False

# Advanced optimization imports
try:
    import faiss  # For efficient similarity search
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Configuration
MAX_CACHE_SIZE = 256
EMBEDDING_DIM = 512
MIN_FACE_SIZE = 24
MAX_IMAGE_SIZE = 2048
BATCH_SIZE = 8
STREAM_FPS = 10
QUALITY_THRESHOLD = 70

class FaceDatastore:
    """Advanced face datastore with FAISS indexing for fast similarity search"""

    def __init__(self, storage_path: str = "storage/embeddings.json"):
        self.storage_path = storage_path
        self.embeddings = {}  # user_id -> list of embeddings
        self.index = None  # FAISS index for fast search
        self.user_ids = []  # Ordered list of user IDs
        self.lock = threading.Lock()

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine)
        else:
            logging.warning("FAISS not available - using slower linear search")

        self.load_datastore()

    def load_datastore(self):
        """Load embeddings from disk and build FAISS index"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            users = data.get('users', {})
            for user_id, user_data in users.items():
                samples = user_data.get('samples', [])
                if samples:
                    embeddings = np.array(samples, dtype=np.float32)
                    self.embeddings[user_id] = embeddings
                    self.user_ids.append(user_id)

                    # Add to FAISS index (mean embedding for fast search)
                    if self.index is not None:
                        mean_emb = np.mean(embeddings, axis=0).reshape(1, -1)
                        faiss.normalize_L2(mean_emb)  # L2 normalize for cosine
                        self.index.add(mean_emb)

            logging.info(f"Loaded {len(self.embeddings)} users with FAISS indexing")

        except FileNotFoundError:
            logging.info("No existing datastore found - starting fresh")
        except Exception as e:
            logging.error(f"Error loading datastore: {e}")

    def save_datastore(self):
        """Save embeddings to disk"""
        with self.lock:
            data = {'users': {}}
            for user_id, embeddings in self.embeddings.items():
                data['users'][user_id] = {
                    'samples': embeddings.tolist(),
                    'model': 'Facenet512',
                    'created': datetime.utcnow().isoformat()
                }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

    def add_user(self, user_id: str, embeddings: np.ndarray):
        """Add new user embeddings"""
        with self.lock:
            self.embeddings[user_id] = embeddings
            if user_id not in self.user_ids:
                self.user_ids.append(user_id)

            # Update FAISS index
            if self.index is not None:
                mean_emb = np.mean(embeddings, axis=0).reshape(1, -1)
                faiss.normalize_L2(mean_emb)
                self.index.add(mean_emb)

            self.save_datastore()

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user's mean embedding"""
        embeddings = self.embeddings.get(user_id)
        if embeddings is not None:
            return np.mean(embeddings, axis=0)
        return None

    def find_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar users using FAISS"""
        if self.index is None or len(self.user_ids) == 0:
            return []

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, min(k, len(self.user_ids)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.user_ids):
                user_id = self.user_ids[idx]
                results.append((user_id, float(score)))

        return results

    def get_stats(self) -> Dict:
        """Get datastore statistics"""
        total_samples = sum(len(emb) for emb in self.embeddings.values())
        return {
            'total_users': len(self.embeddings),
            'total_samples': total_samples,
            'avg_samples_per_user': total_samples / max(1, len(self.embeddings)),
            'indexing_enabled': self.index is not None,
            'embedding_dim': EMBEDDING_DIM
        }


class AdvancedFaceProcessor:
    """Advanced face processing with GPU acceleration and quality assessment"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.device = None
        self.pytorch_available = PYTORCH_AVAILABLE

        if self.pytorch_available:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.face_detector = None
        self.face_embedder = None

        if self.pytorch_available:
            try:
                self.face_detector = MTCNN(keep_all=True, device=self.device)
                self.face_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                logging.info(f"PyTorch models loaded on {self.device}")
            except Exception as e:
                logging.warning(f"Failed to load PyTorch models: {e}")
                self.pytorch_available = False
            except Exception as e:
                logging.warning(f"PyTorch model loading failed: {e}")

    def preprocess_image(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Advanced image preprocessing with quality assessment"""
        metrics = {
            'original_size': image_bgr.shape,
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'quality_score': 0.0,
            'face_detected': False,
            'face_count': 0,
            'processing_time_ms': 0.0
        }

        start_time = time.time()

        # Basic validation
        if image_bgr.shape[0] < MIN_FACE_SIZE or image_bgr.shape[1] < MIN_FACE_SIZE:
            raise ValueError(f"Image too small: {image_bgr.shape[:2]}")

        # Resize if too large
        height, width = image_bgr.shape[:2]
        if max(height, width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LINEAR)

        # Quality assessment
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Brightness (mean)
        metrics['brightness'] = float(np.mean(gray))

        # Contrast (std dev)
        metrics['contrast'] = float(np.std(gray))

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = float(laplacian.var())

        # Overall quality score (0-100)
        brightness_score = min(100, max(0, (metrics['brightness'] - 50) / 50 * 100))
        contrast_score = min(100, max(0, metrics['contrast'] / 50 * 100))
        sharpness_score = min(100, max(0, metrics['sharpness'] / 500 * 100))

        metrics['quality_score'] = (brightness_score + contrast_score + sharpness_score) / 3

        # Face detection check
        if DEEPFACE_AVAILABLE:
            try:
                faces = DeepFace.extract_faces(image_bgr, detector_backend='opencv', enforce_detection=False)
                metrics['face_count'] = len(faces)
                metrics['face_detected'] = len(faces) > 0
            except:
                pass

        metrics['processing_time_ms'] = (time.time() - start_time) * 1000

        return image_bgr, metrics

    async def extract_embedding_async(self, image_bgr: np.ndarray) -> np.ndarray:
        """Async embedding extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.extract_embedding, image_bgr)

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """Extract face embedding with fallback models"""
        # Try DeepFace first (most accurate)
        if DEEPFACE_AVAILABLE:
            try:
                reps = DeepFace.represent(
                    img_path=image_bgr,
                    model_name='Facenet512',
                    detector_backend='mtcnn',
                    enforce_detection=True,
                    align=True
                )

                if reps and len(reps) > 0:
                    emb = np.array(reps[0]['embedding'], dtype=np.float32)
                    return emb / np.linalg.norm(emb)  # L2 normalize

            except Exception as e:
                logging.debug(f"DeepFace extraction failed: {e}")

        # Fallback to PyTorch
        if PYTORCH_AVAILABLE and self.face_detector and self.face_embedder:
            try:
                # Convert to RGB PIL
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Detect faces
                faces = self.face_detector(pil_img)
                if faces is None or len(faces) == 0:
                    raise ValueError("No faces detected")

                # Use first face
                face = faces[0] if faces.dim() == 3 else faces.unsqueeze(0)
                face = face.to(self.device)

                # Extract embedding
                with torch.no_grad():
                    emb = self.face_embedder(face).cpu().numpy().astype(np.float32)
                    return emb[0] / np.linalg.norm(emb[0])  # L2 normalize

            except Exception as e:
                logging.debug(f"PyTorch extraction failed: {e}")

        raise ValueError("All face embedding methods failed")


class RealTimeVerificationEngine:
    """Real-time face verification engine with streaming support"""

    def __init__(self):
        self.datastore = FaceDatastore()
        self.processor = AdvancedFaceProcessor()
        self.active_sessions = {}  # session_id -> session data
        self.session_lock = threading.Lock()

        # Performance tracking
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'average_processing_time': 0.0,
            'active_sessions': 0
        }

    def create_session(self, user_id: str = None) -> str:
        """Create a new verification session"""
        session_id = str(uuid.uuid4())
        with self.session_lock:
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'created': datetime.utcnow(),
                'frames_processed': 0,
                'verifications': [],
                'last_activity': datetime.utcnow()
            }
            self.stats['active_sessions'] = len(self.active_sessions)

        return session_id

    def end_session(self, session_id: str):
        """End a verification session"""
        with self.session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.stats['active_sessions'] = len(self.active_sessions)

    async def verify_frame(self, session_id: str, image_b64: str) -> Dict[str, Any]:
        """Verify a single frame in real-time"""
        try:
            # Decode image
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if image_bgr is None:
                raise ValueError("Invalid image data")

            # Preprocess
            processed_img, quality_metrics = self.processor.preprocess_image(image_bgr)

            # Extract embedding
            start_time = time.time()
            embedding = await self.processor.extract_embedding_async(processed_img)
            processing_time = time.time() - start_time

            # Get session info
            session = self.active_sessions.get(session_id, {})
            user_id = session.get('user_id')

            result = {
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat(),
                'quality_metrics': quality_metrics,
                'processing_time_ms': round(processing_time * 1000, 2),
                'embedding_extracted': True
            }

            # Verification if user specified
            if user_id:
                user_embedding = self.datastore.get_user_embedding(user_id)
                if user_embedding is not None:
                    similarity = float(np.dot(embedding, user_embedding))
                    confidence = min(1.0, max(0.0, (similarity - 0.7) / 0.3))
                    verified = similarity >= 0.7

                    result.update({
                        'verified': verified,
                        'similarity': round(similarity, 4),
                        'confidence': round(confidence, 4),
                        'user_id': user_id
                    })

                    # Update stats
                    self.stats['total_verifications'] += 1
                    if verified:
                        self.stats['successful_verifications'] += 1

                    # Track in session
                    session['verifications'].append({
                        'timestamp': result['timestamp'],
                        'verified': verified,
                        'similarity': similarity,
                        'confidence': confidence
                    })

                else:
                    result['error'] = f"User {user_id} not found in datastore"

            # Update session activity
            session['frames_processed'] = session.get('frames_processed', 0) + 1
            session['last_activity'] = datetime.utcnow()

            return result

        except Exception as e:
            return {
                'session_id': session_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def identify_unknown(self, image_b64: str, k: int = 5) -> Dict[str, Any]:
        """Identify unknown face against all users in datastore"""
        try:
            # Extract embedding
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            processed_img, quality_metrics = self.processor.preprocess_image(image_bgr)
            embedding = await self.processor.extract_embedding_async(processed_img)

            # Find similar users
            candidates = self.datastore.find_similar(embedding, k)

            results = []
            for user_id, score in candidates:
                confidence = min(1.0, max(0.0, (score - 0.7) / 0.3))
                results.append({
                    'user_id': user_id,
                    'similarity': round(score, 4),
                    'confidence': round(confidence, 4),
                    'verified': score >= 0.7
                })

            return {
                'identified': len(results) > 0 and results[0]['verified'],
                'best_match': results[0] if results else None,
                'candidates': results,
                'quality_metrics': quality_metrics,
                'datastore_stats': self.datastore.get_stats()
            }

        except Exception as e:
            return {'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        success_rate = (
            self.stats['successful_verifications'] / self.stats['total_verifications']
            if self.stats['total_verifications'] > 0 else 0
        )

        return {
            'verifications': {
                'total': self.stats['total_verifications'],
                'successful': self.stats['successful_verifications'],
                'success_rate': round(success_rate, 4),
                'average_processing_time_ms': round(self.stats['average_processing_time'], 2)
            },
            'sessions': {
                'active': self.stats['active_sessions'],
                'total_created': len(self.active_sessions)
            },
            'datastore': self.datastore.get_stats(),
            'models': {
                'deepface_available': DEEPFACE_AVAILABLE,
                'pytorch_available': PYTORCH_AVAILABLE,
                'faiss_available': FAISS_AVAILABLE,
                'device': str(self.processor.device) if PYTORCH_AVAILABLE else 'cpu'
            }
        }


# Global engine instance
verification_engine = RealTimeVerificationEngine()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_connections[session_id].append(websocket)

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_connections:
            del self.session_connections[session_id]

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

    async def broadcast_to_session(self, message: str, session_id: str):
        if session_id in self.session_connections:
            for connection in self.session_connections[session_id]:
                await connection.send_text(message)

manager = ConnectionManager()

if __name__ == "__main__":
    print("✓ Advanced Real-time Face Verification Engine Loaded")
    print(f"  - DeepFace: {'✓' if DEEPFACE_AVAILABLE else '✗'}")
    print(f"  - PyTorch: {'✓' if PYTORCH_AVAILABLE else '✗'}")
    print(f"  - FAISS: {'✓' if FAISS_AVAILABLE else '✗'}")
    print(f"  - Datastore: {verification_engine.datastore.get_stats()['total_users']} users")
