"""
BM25 Searcher for E-commerce Product Search
Performs keyword-based search using pre-built BM25 index
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from .bm25_indexer import BM25Indexer
from embedder.config import EmbedderConfig


class BM25Searcher:
    """
    BM25 search engine for keyword-based product search
    Uses pre-built BM25 index for fast keyword matching
    """
    
    def __init__(self, config: EmbedderConfig, index_path: Optional[str] = None):
        """
        Initialize BM25 searcher
        
        Args:
            config: EmbedderConfig for consistent settings
            index_path: Optional path to pre-built index
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize indexer
        self.indexer = BM25Indexer(config)
        
        # Load index if provided
        if index_path:
            self.load_index(index_path)
        
        self.logger.info("BM25 searcher initialized")
    
    def load_index(self, index_path: str) -> bool:
        """
        Load BM25 index from disk
        
        Args:
            index_path: Path to the saved index
            
        Returns:
            True if successful, False otherwise
        """
        return self.indexer.load_index(index_path)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters (category, brand, price_range, etc.)
            score_threshold: Minimum BM25 score threshold
            
        Returns:
            List of search results with BM25 scores
        """
        try:
            if not self.indexer.bm25:
                raise ValueError("BM25 index not loaded. Please load an index first.")
            
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Validate parameters
            top_k = min(max(1, top_k), 1000)  # Clamp between 1 and 1000
            score_threshold = max(0.0, score_threshold)
            
            self.logger.info(f"BM25 search for: '{query}' (top_k={top_k}, filters={filters})")
            
            # Preprocess query
            tokenized_query = self.indexer.preprocess_text(query)
            
            if not tokenized_query:
                self.logger.warning(f"No valid tokens in query: '{query}'")
                return []
            
            # FILTER FIRST: Get filtered document indices if filters provided
            if filters:
                filtered_indices = []
                for idx, doc_id in enumerate(self.indexer.doc_ids):
                    product = self.indexer.documents[doc_id]
                    if self._matches_filters(product, filters):
                        filtered_indices.append(idx)
                
                if not filtered_indices:
                    self.logger.info("No documents match the filters")
                    return []
                
                self.logger.info(f"Filtered to {len(filtered_indices)} documents matching filters")
                
                # Get BM25 scores only for filtered documents
                all_scores = self.indexer.bm25.get_scores(tokenized_query)
                filtered_scores = [(idx, all_scores[idx]) for idx in filtered_indices]
                
                # Sort filtered results by score
                filtered_scores.sort(key=lambda x: x[1], reverse=True)
                top_filtered = filtered_scores[:top_k]
                
            else:
                # No filters: get BM25 scores for all documents
                scores = self.indexer.bm25.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][:top_k]
                top_filtered = [(idx, scores[idx]) for idx in top_indices]
            
            # Format results
            results = []
            for idx, score in top_filtered:
                score = float(score)
                
                # Apply score threshold
                if score < score_threshold:
                    continue
                
                doc_id = self.indexer.doc_ids[idx]
                product = self.indexer.documents[doc_id].copy()
                
                result = {
                    "rank": len(results) + 1,
                    "score": score,
                    "product": product,
                    "relevance": self._score_to_relevance(score),
                    "doc_id": doc_id
                }
                
                results.append(result)
            
            self.logger.info(f"Found {len(results)} BM25 results")
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            raise
    
    def multi_search(
        self,
        queries: List[str],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform multiple BM25 searches efficiently
        
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
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Product data or None if not found
        """
        return self.indexer.documents.get(doc_id)
    
    def _matches_filters(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if product matches the given filters
        
        Args:
            product: Product data
            filters: Filter conditions
            
        Returns:
            True if product matches all filters
        """
        try:
            # Category filter
            if filters.get("category"):
                product_category = product.get("category", "")
                if isinstance(product_category, list):
                    if not any(filters["category"] in cat for cat in product_category):
                        return False
                else:
                    if filters["category"] not in str(product_category):
                        return False
            
            # Brand filter
            if filters.get("brand"):
                product_brand = str(product.get("brand", "")).lower()
                if filters["brand"].lower() not in product_brand:
                    return False
            
            # Price range filter
            if filters.get("price_range"):
                price = product.get("price", 0)
                if isinstance(price, (int, float)) and price > 0:
                    price_range = filters["price_range"]
                    if price_range == "under_25" and price >= 25:
                        return False
                    elif price_range == "25_to_50" and not (25 <= price < 50):
                        return False
                    elif price_range == "50_to_100" and not (50 <= price < 100):
                        return False
                    elif price_range == "100_to_200" and not (100 <= price < 200):
                        return False
                    elif price_range == "200_to_500" and not (200 <= price < 500):
                        return False
                    elif price_range == "over_500" and price < 500:
                        return False
                elif price == 0 and price_range != "unknown":
                    return False
            
            # Has image filter
            if filters.get("has_image") is not None:
                has_image = bool(product.get("image_url"))
                if has_image != filters["has_image"]:
                    return False
            
            # Has brand filter
            if filters.get("has_brand") is not None:
                has_brand = bool(product.get("brand"))
                if has_brand != filters["has_brand"]:
                    return False
            
            # Price filters
            if filters.get("min_price") is not None:
                price = product.get("price", 0)
                if isinstance(price, (int, float)) and price < filters["min_price"]:
                    return False
            
            if filters.get("max_price") is not None:
                price = product.get("price", 0)
                if isinstance(price, (int, float)) and price > filters["max_price"]:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error applying filters: {e}")
            return True  # Include item if filter check fails
    
    def _score_to_relevance(self, score: float) -> str:
        """Convert BM25 score to human-readable relevance"""
        if score >= 15.0:
            return "Excellent"
        elif score >= 10.0:
            return "Very Good"
        elif score >= 5.0:
            return "Good"
        elif score >= 2.0:
            return "Fair"
        elif score >= 1.0:
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
        output.append(f"\nFound {len(results)} BM25 results:\n")
        output.append("=" * 80)
        
        for result in results:
            product = result["product"]
            rank = result["rank"]
            score = result["score"]
            relevance = result["relevance"]
            
            output.append(f"\n#{rank} - {relevance} Match (BM25 Score: {score:.3f})")
            output.append("-" * 40)
            output.append(f"Product: {product.get('title', 'No title')}")
            output.append(f"Category: {product.get('category', 'Unknown')}")
            output.append(f"Brand: {product.get('brand', 'Unknown')}")
            
            # Handle price display
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
                    feature_preview = ", ".join(str(f) for f in features[:2])
                    if len(features) > 2:
                        feature_preview += f" (+{len(features)-2} more)"
                    output.append(f"Features: {feature_preview}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        return self.indexer.get_index_stats()
    
    def test_connection(self) -> bool:
        """Test if BM25 index is loaded and ready"""
        return self.indexer.bm25 is not None


# Convenience function for quick BM25 searches
def quick_bm25_search(
    query: str,
    index_path: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Quick BM25 search function
    
    Args:
        query: Search query
        index_path: Path to BM25 index
        top_k: Number of results
        
    Returns:
        Search results
    """
    config = EmbedderConfig()
    searcher = BM25Searcher(config, index_path)
    return searcher.search(query, top_k)