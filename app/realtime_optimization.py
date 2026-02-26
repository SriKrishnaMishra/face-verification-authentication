"""
Optimized Real-time Face Verification Module
Adds caching, batch processing, and streaming capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import json
import os
from datetime import datetime
import cv2

# Configuration for optimization
MAX_CACHE_SIZE = 128  # LRU cache size
EMBEDDING_DIM = 512  # Standard embedding dimension
MIN_FACE_SIZE = 16   # Minimum face size in pixels
MAX_IMAGE_SIZE = 2048  # Maximum image dimension


class EmbeddingCache:
    """LRU Cache for face embeddings to speed up real-time verification"""
    
    def __init__(self, max_size: int = 128):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, user_id: str) -> Optional[np.ndarray]:
        """Get cached embedding (returns mean embedding)"""
        if user_id in self.cache:
            self.access_count[user_id] = self.access_count.get(user_id, 0) + 1
            self.hit_count += 1
            return self.cache[user_id]
        self.miss_count += 1
        return None
    
    def put(self, user_id: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        if len(self.cache) >= self.max_size and user_id not in self.cache:
            # Remove least accessed item
            lru_user = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_user]
            del self.access_count[lru_user]
        
        self.cache[user_id] = embedding
        self.access_count[user_id] = 0
    
    def clear(self, user_id: str) -> None:
        """Clear cache for specific user"""
        if user_id in self.cache:
            del self.cache[user_id]
            if user_id in self.access_count:
                del self.access_count[user_id]
    
    def clear_all(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "users_cached": list(self.cache.keys())
        }


class ImagePreprocessor:
    """Optimize images for face detection and embedding"""
    
    @staticmethod
    def preprocess(image_bgr: np.ndarray, 
                   target_size: int = 640,
                   quality_check: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for better face detection
        Returns: (processed_image, quality_metrics)
        """
        metrics = {
            "original_size": image_bgr.shape,
            "brightness": None,
            "contrast": None,
            "is_blurry": False,
            "quality_score": 0.0
        }
        
        # Check image dimensions
        if image_bgr.shape[0] < MIN_FACE_SIZE or image_bgr.shape[1] < MIN_FACE_SIZE:
            raise ValueError(f"Image too small: {image_bgr.shape[0]}x{image_bgr.shape[1]}")
        
        # Resize if too large (preserve aspect ratio)
        height, width = image_bgr.shape[:2]
        if max(height, width) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(height, width)
            new_h, new_w = int(height * scale), int(width * scale)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Check brightness (Laplacian variance - blurriness metric)
        if quality_check:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
            metrics["is_blurry"] = blur_score < 100  # Threshold for blurriness
            metrics["brightness"] = float(np.mean(gray))
            metrics["contrast"] = float(np.std(gray))
            
            # Calculate quality score (0-100)
            quality_score = min(100, max(0, blur_score / 10))
            metrics["quality_score"] = round(quality_score, 2)
            
            # Warn if low quality
            if quality_score < 50:
                raise ValueError(f"Low image quality: {quality_score}%. Image may be too blurry.")
        
        # Normalize color channels
        image_normalized = cv2.normalize(image_bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        metrics["processed_size"] = image_normalized.shape
        return image_normalized, metrics


class FaceMatcher:
    """Optimized face matching with confidence scoring"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.match_history = []
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Ensure embeddings are normalized
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        return float(np.dot(e1, e2))
    
    def match_with_confidence(self, probe_embedding: np.ndarray,
                             gallery_embedding: np.ndarray) -> Dict:
        """
        Match embeddings with confidence metrics
        Returns confidence score and verification result
        """
        similarity = self.compute_similarity(probe_embedding, gallery_embedding)
        
        # Calculate confidence (distance from threshold)
        if similarity >= self.threshold:
            confidence = min(1.0, (similarity - self.threshold) / (1.0 - self.threshold))
            verified = True
        else:
            confidence = max(0.0, similarity / self.threshold)
            verified = False
        
        result = {
            "verified": verified,
            "similarity": round(similarity, 4),
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
            "margin": round(similarity - self.threshold, 4)
        }
        
        self.match_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            **result
        })
        
        return result
    
    def batch_match(self, probe_embedding: np.ndarray,
                    gallery_embeddings: List[np.ndarray]) -> Dict:
        """Match against multiple gallery embeddings"""
        if not gallery_embeddings:
            return {"error": "No gallery embeddings"}
        
        similarities = [
            self.compute_similarity(probe_embedding, emb) 
            for emb in gallery_embeddings
        ]
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return {
            "best_match_idx": int(best_idx),
            "best_score": round(float(best_score), 4),
            "verified": best_score >= self.threshold,
            "all_scores": [round(float(s), 4) for s in similarities],
            "mean_score": round(float(np.mean(similarities)), 4),
            "std_score": round(float(np.std(similarities)), 4)
        }
    
    def get_match_history(self, limit: int = 100) -> List[Dict]:
        """Get recent match history"""
        return self.match_history[-limit:]


class RealTimeVerificationPipeline:
    """Complete real-time face verification pipeline"""
    
    def __init__(self, embedding_getter, matcher: FaceMatcher, cache_enabled: bool = True):
        self.embedding_getter = embedding_getter
        self.matcher = matcher
        self.preprocessor = ImagePreprocessor()
        self.cache = EmbeddingCache(MAX_CACHE_SIZE) if cache_enabled else None
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "average_match_score": 0.0,
            "processing_times": []
        }
    
    def verify_realtime(self, probe_image: np.ndarray, 
                       user_id: str,
                       gallery_embedding: np.ndarray) -> Dict:
        """
        Real-time face verification with preprocessing and caching
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Preprocess image
            processed_image, quality_metrics = self.preprocessor.preprocess(
                probe_image,
                quality_check=True
            )
            
            # Step 2: Get embedding for probe image
            probe_embedding = self.embedding_getter(processed_image)
            
            # Step 3: Check cache for gallery embedding
            if self.cache:
                cached_gallery = self.cache.get(user_id)
                if cached_gallery is not None:
                    gallery_embedding = cached_gallery
            
            # Step 4: Match embeddings
            match_result = self.matcher.match_with_confidence(
                probe_embedding, 
                gallery_embedding
            )
            
            # Step 5: Update stats
            processing_time = time.time() - start_time
            self.stats["total_verifications"] += 1
            if match_result["verified"]:
                self.stats["successful_verifications"] += 1
            else:
                self.stats["failed_verifications"] += 1
            self.stats["processing_times"].append(processing_time)
            
            # Update average match score
            scores = [m["similarity"] for m in self.matcher.get_match_history(100)]
            self.stats["average_match_score"] = round(np.mean(scores), 4) if scores else 0.0
            
            return {
                **match_result,
                "quality_metrics": quality_metrics,
                "processing_time_ms": round(processing_time * 1000, 2),
                "cached": self.cache is not None and user_id in self.cache.cache if self.cache else False
            }
        
        except Exception as e:
            self.stats["total_verifications"] += 1
            self.stats["failed_verifications"] += 1
            raise e
    
    def get_performance_stats(self) -> Dict:
        """Get pipeline performance statistics"""
        success_rate = (
            self.stats["successful_verifications"] / self.stats["total_verifications"]
            if self.stats["total_verifications"] > 0 else 0
        )
        
        avg_time = (
            np.mean(self.stats["processing_times"])
            if self.stats["processing_times"] else 0
        )
        
        return {
            "total_verifications": self.stats["total_verifications"],
            "success_rate": round(success_rate, 4),
            "average_processing_time_ms": round(avg_time * 1000, 2),
            "average_match_score": self.stats["average_match_score"],
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def clear_cache_for_user(self, user_id: str) -> None:
        """Clear cache when user updates embeddings"""
        if self.cache:
            self.cache.clear(user_id)


# Batch processing for multiple users
class BatchVerifier:
    """Batch verification of multiple users"""
    
    @staticmethod
    def verify_batch(probe_image: np.ndarray,
                     probe_embedding: np.ndarray,
                     candidates: Dict[str, np.ndarray],
                     matcher: FaceMatcher,
                     threshold: float = 0.7) -> List[Dict]:
        """
        Verify against multiple candidate users
        Returns sorted list of matches
        """
        results = []
        
        for user_id, gallery_embedding in candidates.items():
            similarity = matcher.compute_similarity(probe_embedding, gallery_embedding)
            results.append({
                "user_id": user_id,
                "similarity": round(similarity, 4),
                "verified": similarity >= threshold,
                "confidence": round(min(1.0, max(0.0, (similarity - threshold) / 0.3)), 4)
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


if __name__ == "__main__":
    print("✓ Optimized Face Verification Module Loaded")
    print(f"  - Cache Size: {MAX_CACHE_SIZE}")
    print(f"  - Max Image Size: {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")
    print(f"  - Min Face Size: {MIN_FACE_SIZE}x{MIN_FACE_SIZE}")
