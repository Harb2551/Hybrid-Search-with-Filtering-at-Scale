"""
Product Embedder for E-commerce Hybrid Search
Optimized for embedding Amazon product data with rich context
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EmbedderConfig
import logging


class ProductEmbedder:
    """
    Specialized embedder for e-commerce products
    Creates rich embeddings from multiple product fields
    """
    
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load embedding model
        self.model = SentenceTransformer(
            config.embedding_model,
            device=self.device
        )
        
        # Verify embedding dimensions
        test_embedding = self.model.encode("test")
        actual_dim = len(test_embedding)
        if actual_dim != config.embedding_dimension:
            self.logger.warning(
                f"Model dimension {actual_dim} doesn't match config {config.embedding_dimension}"
            )
            self.config.embedding_dimension = actual_dim
            
        self.logger.info(f"Loaded {config.embedding_model} with dimension {actual_dim}")
    
    def create_product_text(self, product: Dict[str, Any]) -> str:
        """
        Create rich text representation from product data
        Combines multiple fields for better semantic understanding
        """
        parts = []
        
        # Title (most important)
        if product.get('title'):
            parts.append(f"Product: {product['title']}")
        
        # Category and brand for context
        if product.get('category'):
            parts.append(f"Category: {product['category']}")
        
        if product.get('brand'):
            parts.append(f"Brand: {product['brand']}")
        
        # Description for detailed context
        if product.get('description'):
            desc = product['description'][:500]  # Limit description length
            parts.append(f"Description: {desc}")
        
        # Features for specific attributes
        if product.get('features'):
            features = product['features']
            if isinstance(features, list):
                # Join first 3 features to avoid token limit
                feature_text = " | ".join(features[:3])
                parts.append(f"Features: {feature_text}")
            elif isinstance(features, str):
                parts.append(f"Features: {features[:200]}")
        
        # Price for value context (optional)
        if product.get('price'):
            parts.append(f"Price: ${product['price']}")
        
        return " | ".join(parts)
    
    def embed_products(self, products: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Create embeddings for a batch of products
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of embeddings as lists of floats
        """
        # Create rich text representations
        texts = [self.create_product_text(product) for product in products]
        
        # Generate embeddings in batches
        embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=len(batch)
                )
                
                # Convert to numpy and normalize
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Normalize embeddings for cosine similarity
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)
                
                embeddings.extend(batch_embeddings.tolist())
                
            except Exception as e:
                self.logger.error(f"Error embedding batch {i//batch_size}: {e}")
                # Return zero embeddings for failed batch
                zero_embedding = [0.0] * self.config.embedding_dimension
                embeddings.extend([zero_embedding] * len(batch))
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Create embedding for a search query
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding as list of floats
        """
        try:
            embedding = self.model.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error embedding query '{query}': {e}")
            return [0.0] * self.config.embedding_dimension
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.config.embedding_dimension
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.embedding_model,
            "dimension": self.config.embedding_dimension,
            "device": self.device,
            "max_sequence_length": self.config.max_sequence_length,
            "batch_size": self.config.batch_size
        }