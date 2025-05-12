#1. This code is created such that the api can be used for accessing S3 and  That API would provide the dataset as a list of (name, bytes) tuples
#2. This code will take input image from frontend and image dataset for classification as a dataset and output will be person's ID
#3. Output API should take the Person's unique ID and return the data. The unique ID is stored in "result"

#code

import cv2
import numpy as np
import insightface
import faiss
import logging
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.6  # Threshold for positive identification (0-1 range)
DETECTION_SIZE: Tuple[int, int] = (640, 640)  # Smaller size for faster processing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CandidateVerifier:
    """
    Verifies if a candidate image matches any image in a provided dataset.
    Works completely in-memory without file system access.
    """
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD,
                 detection_size: Tuple[int, int] = DETECTION_SIZE):
        """
        Initialize the verifier with face detection model.
        
        Args:
            similarity_threshold: Threshold for positive identification (0-1 range)
            detection_size: Size for face detection
        """
        self.similarity_threshold = similarity_threshold
        self.detection_size = detection_size
        self.face_model = None
        self.faiss_index = None
        self.dataset_embeddings = None
        self.image_names = None
        
        self._initialize_face_model()

    def _initialize_face_model(self):
        """Initialize InsightFace detection model."""
        self.face_model = insightface.app.FaceAnalysis()
        try:
            self.face_model.prepare(ctx_id=0, det_size=self.detection_size)
            logger.info("Face detection model initialized")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def load_dataset_from_bytes(self, dataset_images: List[Tuple[str, bytes]]):
        """
        Load dataset from in-memory byte arrays.
        
        Args:
            dataset_images: List of tuples containing (image_name, image_bytes)
        """
        if not dataset_images:
            raise ValueError("Empty dataset provided")
        
        embeddings = []
        names = []
        
        for image_name, image_bytes in dataset_images:
            try:
                # Load image from bytes
                img = self._bytes_to_image(image_bytes)
                if img is None:
                    logger.warning(f"Could not decode image: {image_name}")
                    continue
                    
                # Get face embeddings
                processed_img = self._preprocess_image(img)
                faces = self.face_model.get(processed_img)
                
                if len(faces) == 0:
                    logger.warning(f"No faces detected in {image_name}")
                    continue
                
                # Use the first face found in the image
                face = faces[0]
                embedding = face.embedding.astype(np.float32)
                embeddings.append(embedding)
                names.append(image_name)  # Store the provided image name
                
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("No valid faces found in the dataset")
        
        self.dataset_embeddings = np.array(embeddings)
        self.image_names = names
        self._build_faiss_index()
        logger.info(f"Loaded {len(self.image_names)} dataset images")
    
    def _build_faiss_index(self):
        """Build FAISS index with normalized embeddings."""
        if self.dataset_embeddings is None or len(self.dataset_embeddings) == 0:
            raise ValueError("No embeddings available for index creation")
        
        try:
            embedding_dim = self.dataset_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            faiss.normalize_L2(self.dataset_embeddings)
            self.faiss_index.add(self.dataset_embeddings)
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} embeddings")
        except Exception as e:
            logger.error(f"Index construction failed: {e}")
            raise
    
    def verify_candidate(self, candidate_image_bytes: bytes) -> Dict[str, Optional[str]]:
        """
        Verify if the candidate image matches any image in the dataset.
        
        Args:
            candidate_image_bytes: Image bytes to verify against the dataset
            
        Returns:
            Dictionary with:
            - 'match': dataset image name if match found, None otherwise
            - 'similarity': highest similarity score (0-1)
            - 'status': 'present' or 'not present'
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            raise RuntimeError("Dataset not loaded")
        
        # Load candidate image from bytes
        candidate_image = self._bytes_to_image(candidate_image_bytes)
        if candidate_image is None:
            raise ValueError("Could not decode candidate image")
        
        # Process candidate image
        processed_img = self._preprocess_image(candidate_image)
        faces = self.face_model.get(processed_img)
        
        if len(faces) == 0:
            logger.warning("No faces detected in candidate image")
            return {'match': None, 'similarity': 0, 'status': 'not present'}
        
        # Use the first face found in the candidate image
        candidate_face = faces[0]
        candidate_embedding = candidate_face.embedding.astype(np.float32).reshape(1, -1)
        
        # Search in the dataset
        distances, indices = self.faiss_index.search(candidate_embedding, 1)
        similarity = distances[0][0]
        best_match_idx = indices[0][0]
        
        if similarity >= self.similarity_threshold and best_match_idx < len(self.image_names):
            match_name = self.image_names[best_match_idx]
            return {
                'match': match_name,
                'similarity': float(similarity),
                'status': 'present'
            }
        else:
            return {
                'match': None,
                'similarity': float(similarity),
                'status': 'not present'
            }
    
    def _bytes_to_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert bytes to OpenCV image."""
        try:
            img_array = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error converting bytes to image: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Basic image preprocessing for better face detection."""
        # Convert to RGB (InsightFace expects RGB)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Simple histogram equalization for better contrast
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        processed_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize verifier
        verifier = CandidateVerifier()
        
        # The API would provide the dataset as a list of (name, bytes) tuples
        # This is just example data - in reality the API would get these from S3
        dataset_images = [
            ("person1", open("person1.jpg", "rb").read()),
            ("person2", open("person2.jpg", "rb").read()),
            # ... more images
        ]
        
        # Load dataset from bytes
        print("\nLoading dataset...")
        verifier.load_dataset_from_bytes(dataset_images)
        
        # In a real scenario, the frontend would send the image bytes
        candidate_bytes = open("candidate.jpg", "rb").read()
        
        # Verify candidate
        result = verifier.verify_candidate(candidate_bytes)
        
        
        
    except Exception as e:
        print(f"\nError: {str(e)}")