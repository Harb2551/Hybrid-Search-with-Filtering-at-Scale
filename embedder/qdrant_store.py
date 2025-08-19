"""
Qdrant Vector Store Manager for E-commerce Hybrid Search
Handles all Qdrant operations optimized for product data at scale
"""

from typing import List, Dict, Any, Optional, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo,
    PointStruct, Filter, FieldCondition, 
    Range, MatchValue, HnswConfigDiff,
    OptimizersConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType
)
from .config import EmbedderConfig
import logging
import time
import json


class QdrantVectorStore:
    """
    Qdrant vector store optimized for e-commerce product search
    Handles 1M+ products with efficient filtering and search
    """
    
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Qdrant client - use minimal config that matches working curl
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=60
        )
        
        self.collection_name = config.collection_name
        self.logger.info(f"Connected to Qdrant at {config.qdrant_url}")
    
    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        Create optimized collection for e-commerce products
        
        Args:
            force_recreate: Whether to delete existing collection first
            
        Returns:
            True if collection was created, False if already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    self.logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    self.logger.info(f"Collection {self.collection_name} already exists")
                    return False
            
            # Create collection with optimized settings for 1M+ products
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=self.config.hnsw_m,
                        ef_construct=self.config.hnsw_ef_construct,
                        full_scan_threshold=self.config.full_scan_threshold
                    )
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,  # Optimize for 1M+ scale
                    indexing_threshold=20000,
                    flush_interval_sec=5
                ),
                # Enable scalar quantization for memory efficiency
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
            
            self.logger.info(f"Created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return None
    
    def upsert_products(self, products: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """
        Insert or update products in the vector store
        
        Args:
            products: List of product dictionaries
            embeddings: Corresponding embeddings
            
        Returns:
            True if successful
        """
        if len(products) != len(embeddings):
            raise ValueError("Products and embeddings must have same length")
        
        try:
            points = []
            for product, embedding in zip(products, embeddings):
                # Create point with product data as payload
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "product_id": product.get("id", ""),
                        "title": product.get("title", ""),
                        "description": product.get("description", "")[:500],  # Limit length
                        "category": product.get("category", ""),
                        "brand": product.get("brand", ""),
                        "price": self._parse_price(product.get("price", 0)),
                        "features": product.get("features", [])[:5] if isinstance(product.get("features"), list) else [],
                        "image_url": product.get("image_url", ""),
                        "source": product.get("source", ""),
                        # Add searchable fields for filtering
                        "has_image": bool(product.get("image_url")),
                        "has_brand": bool(product.get("brand")),
                        "price_range": self._get_price_range(product.get("price", 0))
                    }
                )
                points.append(point)
            
            # Upsert in batches
            batch_size = min(100, self.config.batch_size)
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                retry_count = 0
                while retry_count < self.config.max_retries:
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch
                        )
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= self.config.max_retries:
                            self.logger.error(f"Failed to upsert batch after {self.config.max_retries} retries: {e}")
                            raise
                        
                        self.logger.warning(f"Retry {retry_count} for batch upsert: {e}")
                        time.sleep(self.config.retry_delay * retry_count)
            
            self.logger.info(f"Successfully upserted {len(products)} products")
            return True
            
        except Exception as e:
            self.logger.error(f"Error upserting products: {e}")
            raise
    
    def search_products(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for products using vector similarity with optional filtering
        
        Args:
            query_embedding: Query vector
            filters: Optional filters (category, brand, price_range, etc.)
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with scores
        """
        try:
            # Build filter conditions
            filter_conditions = self._build_filter_conditions(filters) if filters else None
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "score": float(result.score),
                    "product": result.payload
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching products: {e}")
            raise
    
    def count_products(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count products matching filters
        
        Args:
            filters: Optional filters
            
        Returns:
            Number of matching products
        """
        try:
            filter_conditions = self._build_filter_conditions(filters) if filters else None
            
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_conditions
            )
            
            return result.count
            
        except Exception as e:
            self.logger.error(f"Error counting products: {e}")
            return 0
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter conditions from filter dict"""
        conditions = []
        
        # Category filter
        if filters.get("category"):
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=filters["category"])
                )
            )
        
        # Brand filter
        if filters.get("brand"):
            conditions.append(
                FieldCondition(
                    key="brand",
                    match=MatchValue(value=filters["brand"])
                )
            )
        
        # Price range filter
        if filters.get("min_price") is not None or filters.get("max_price") is not None:
            price_range = Range(
                gte=filters.get("min_price"),
                lte=filters.get("max_price")
            )
            conditions.append(
                FieldCondition(
                    key="price",
                    range=price_range
                )
            )
        
        # Price range category filter
        if filters.get("price_range"):
            conditions.append(
                FieldCondition(
                    key="price_range",
                    match=MatchValue(value=filters["price_range"])
                )
            )
        
        # Has image filter
        if filters.get("has_image") is not None:
            conditions.append(
                FieldCondition(
                    key="has_image",
                    match=MatchValue(value=filters["has_image"])
                )
            )
        
        # Has brand filter
        if filters.get("has_brand") is not None:
            conditions.append(
                FieldCondition(
                    key="has_brand",
                    match=MatchValue(value=filters["has_brand"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _parse_price(self, price) -> float:
        """Parse price from various formats to float"""
        if not price:
            return 0.0
            
        if isinstance(price, (int, float)):
            return float(price)
            
        if isinstance(price, str):
            # Remove currency symbols and whitespace
            price_str = price.strip().replace('$', '').replace(',', '')
            try:
                return float(price_str)
            except (ValueError, TypeError):
                return 0.0
        
        return 0.0
    
    def _get_price_range(self, price) -> str:
        """Categorize price into ranges for filtering"""
        price_float = self._parse_price(price)
        
        if price_float == 0:
            return "unknown"
        elif price_float < 25:
            return "under_25"
        elif price_float < 50:
            return "25_to_50"
        elif price_float < 100:
            return "50_to_100"
        elif price_float < 200:
            return "100_to_200"
        elif price_float < 500:
            return "200_to_500"
        else:
            return "over_500"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.get_collection_info()
            if not info:
                return {"error": "Collection not found"}
            
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "config": {
                    "dimension": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance,
                    "hnsw_m": info.config.params.vectors.hnsw_config.m,
                    "ef_construct": info.config.params.vectors.hnsw_config.ef_construct
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False