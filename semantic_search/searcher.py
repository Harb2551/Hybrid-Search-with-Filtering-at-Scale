"""
Semantic Search implementation using Qdrant vector store
Performs semantic search queries against product embeddings
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from embedder.product_embedder import ProductEmbedder
from embedder.qdrant_store import QdrantVectorStore


class SemanticSearcher:
    """
    Semantic search engine for e-commerce products
    Queries the Qdrant vector store with natural language queries
    """
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize semantic searcher
        
        Args:
            config: EmbedderConfig with Qdrant and model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedder = ProductEmbedder(config)
        self.vector_store = QdrantVectorStore(config)
        
        self.logger.info(f"Semantic searcher initialized for collection: {config.collection_name}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the product catalog
        
        Args:
            query: Natural language search query
            top_k: Number of results to return (max 100)
            filters: Optional filters (category, brand, price_range, etc.)
            score_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of search results with scores and product data
        """
        try:
            # Validate inputs
            top_k = min(max(1, top_k), 100)  # Clamp between 1 and 100
            score_threshold = max(0.0, min(1.0, score_threshold))  # Clamp between 0 and 1
            
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            self.logger.info(f"Searching for: '{query}' (top_k={top_k}, filters={filters})")
            
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search_products(
                query_embedding=query_embedding,
                filters=filters,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results for better usability
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = {
                    "rank": i,
                    "score": result["score"],
                    "product": result["product"],
                    "relevance": self._score_to_relevance(result["score"])
                }
                formatted_results.append(formatted_result)
            
            self.logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def search_similar_products(
        self,
        product_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find products similar to a given product
        
        Args:
            product_id: ID of the reference product
            top_k: Number of similar products to return
            filters: Optional filters to apply
            
        Returns:
            List of similar products
        """
        try:
            # TODO: Implement product-to-product similarity
            # This would require getting the embedding of the reference product
            # and searching for similar embeddings
            raise NotImplementedError("Product similarity search not yet implemented")
            
        except Exception as e:
            self.logger.error(f"Similar products search failed: {e}")
            raise
    
    def multi_search(
        self,
        queries: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform multiple searches efficiently
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Optional filters to apply to all queries
            
        Returns:
            Dictionary mapping queries to their results
        """
        try:
            results = {}
            for query in queries:
                if query.strip():
                    results[query] = self.search(query, top_k, filters)
                else:
                    results[query] = []
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-search failed: {e}")
            raise
    
    def list_all_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.vector_store.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []
    
    def create_payload_indexes(self) -> bool:
        """Create indexes for filtering fields"""
        try:
            from qdrant_client.models import PayloadSchemaType
            
            # Create indexes for common filter fields
            indexes_to_create = [
                ("category", PayloadSchemaType.KEYWORD),
                ("brand", PayloadSchemaType.KEYWORD),
                ("price_range", PayloadSchemaType.KEYWORD),
                ("has_image", PayloadSchemaType.BOOL),
                ("has_brand", PayloadSchemaType.BOOL),
                ("price", PayloadSchemaType.FLOAT)
            ]
            
            self.logger.info("Creating payload indexes for filtering...")
            
            for field_name, field_type in indexes_to_create:
                try:
                    self.vector_store.client.create_payload_index(
                        collection_name=self.config.collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                    self.logger.info(f"SUCCESS: Created index for field: {field_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        self.logger.info(f"SUCCESS: Index for {field_name} already exists")
                    else:
                        self.logger.warning(f"WARNING: Failed to create index for {field_name}: {e}")
            
            self.logger.info("Payload index creation completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create payload indexes: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant"""
        try:
            stats = self.get_collection_stats()
            return "error" not in stats
        except Exception as e:
            return False
    
    def _score_to_relevance(self, score: float) -> str:
        """Convert similarity score to human-readable relevance"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results for display
        
        Args:
            results: Search results from search() method
            
        Returns:
            Formatted string for display
        """
        if not results:
            return "No results found."
        
        output = []
        output.append(f"\nFound {len(results)} results:\n")
        output.append("=" * 80)
        
        for result in results:
            product = result["product"]
            rank = result["rank"]
            score = result["score"]
            relevance = result["relevance"]
            
            output.append(f"\n#{rank} - {relevance} Match (Score: {score:.3f})")
            output.append("-" * 40)
            output.append(f"Product: {product.get('title', 'No title')}")
            output.append(f"Category: {product.get('category', 'Unknown')}")
            output.append(f"Brand: {product.get('brand', 'Unknown')}")
            price = product.get('price', 0)
            if price == 0:
                output.append(f"Price: Price not available")
            else:
                output.append(f"Price: ${price:.2f}")
            
            # Show description if available
            description = product.get('description', '')
            if description:
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                output.append(f"Description: {desc_preview}")
            
            # Show features if available
            features = product.get('features', [])
            if features:
                if isinstance(features, list) and features:
                    feature_preview = ", ".join(features[:2])
                    if len(features) > 2:
                        feature_preview += f" (+{len(features)-2} more)"
                    output.append(f"Features: {feature_preview}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)


# Convenience function for quick searches
def quick_search(
    query: str,
    top_k: int = 5,
    collection_name: str = "amazon_products",
    qdrant_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Quick search function for simple queries
    
    Args:
        query: Search query
        top_k: Number of results
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL
        api_key: Qdrant API key
        
    Returns:
        Search results
    """
    config = EmbedderConfig(
        collection_name=collection_name
    )
    
    # Override with provided values if given
    if qdrant_url:
        config.qdrant_url = qdrant_url
    if api_key:
        config.qdrant_api_key = api_key
    
    searcher = SemanticSearcher(config)
    return searcher.search(query, top_k)