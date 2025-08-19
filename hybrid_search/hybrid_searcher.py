"""
Hybrid Search Engine for E-commerce Products
Combines Semantic Search (dense vectors) and BM25 Search (sparse keywords)
using Reciprocal Rank Fusion (RRF) for optimal search results
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from semantic_search.searcher import SemanticSearcher
from bm25.bm25_searcher import BM25Searcher
from .fusion_strategies import RRFusion


class HybridSearcher:
    """
    Hybrid search engine that combines semantic and BM25 search
    Uses RRF (Reciprocal Rank Fusion) to merge results from both engines
    """
    
    def __init__(
        self,
        config: EmbedderConfig,
        bm25_index_path: str,
        rrf_k: int = 60,
        parallel_search: bool = True
    ):
        """
        Initialize hybrid searcher
        
        Args:
            config: EmbedderConfig for Qdrant and embedding settings
            bm25_index_path: Path to the BM25 index file
            rrf_k: RRF smoothing parameter (default: 60)
            parallel_search: Whether to run searches in parallel (default: True)
        """
        self.config = config
        self.bm25_index_path = bm25_index_path
        self.parallel_search = parallel_search
        self.logger = logging.getLogger(__name__)
        
        # Initialize search engines
        self.semantic_searcher = SemanticSearcher(config)
        self.bm25_searcher = BM25Searcher(config, bm25_index_path)
        
        # Initialize fusion strategy
        self.fusion = RRFusion(k=rrf_k)
        
        self.logger.info(f"Hybrid searcher initialized with RRF k={rrf_k}")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        semantic_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and BM25 results
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            filters: Optional filters (category, brand, price_range, etc.)
            semantic_top_k: Number of semantic results to retrieve (default: top_k * 2)
            bm25_top_k: Number of BM25 results to retrieve (default: top_k * 2)
            score_threshold: Minimum score threshold for individual engines
            
        Returns:
            List of hybrid search results with RRF scores
        """
        try:
            # Validate inputs
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            top_k = min(max(1, top_k), 100)
            
            # Set retrieval limits (get more results for better fusion)
            semantic_limit = semantic_top_k or min(top_k * 2, 50)
            bm25_limit = bm25_top_k or min(top_k * 2, 50)
            
            self.logger.info(f"Hybrid search: '{query}' (top_k={top_k}, filters={filters})")
            
            # Perform searches (parallel or sequential)
            start_time = time.time()
            
            if self.parallel_search:
                semantic_results, bm25_results = self._parallel_search(
                    query, semantic_limit, bm25_limit, filters, score_threshold
                )
            else:
                semantic_results, bm25_results = self._sequential_search(
                    query, semantic_limit, bm25_limit, filters, score_threshold
                )
            
            search_time = time.time() - start_time
            
            # Fuse results using RRF
            start_fusion = time.time()
            fused_results = self.fusion.fuse_results(
                semantic_results, bm25_results, max_results=top_k
            )
            fusion_time = time.time() - start_fusion
            
            # Add timing information
            for result in fused_results:
                result['timing'] = {
                    'search_time': search_time,
                    'fusion_time': fusion_time,
                    'total_time': search_time + fusion_time
                }
            
            self.logger.info(
                f"Hybrid search completed: {len(fused_results)} results "
                f"(search: {search_time:.3f}s, fusion: {fusion_time:.3f}s)"
            )
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise
    
    def _parallel_search(
        self,
        query: str,
        semantic_limit: int,
        bm25_limit: int,
        filters: Optional[Dict[str, Any]],
        score_threshold: float
    ) -> tuple:
        """Perform semantic and BM25 searches in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches
            semantic_future = executor.submit(
                self.semantic_searcher.search,
                query, semantic_limit, filters, score_threshold
            )
            bm25_future = executor.submit(
                self.bm25_searcher.search,
                query, bm25_limit, filters, score_threshold
            )
            
            # Get results
            semantic_results = semantic_future.result()
            bm25_results = bm25_future.result()
            
            return semantic_results, bm25_results
    
    def _sequential_search(
        self,
        query: str,
        semantic_limit: int,
        bm25_limit: int,
        filters: Optional[Dict[str, Any]],
        score_threshold: float
    ) -> tuple:
        """Perform semantic and BM25 searches sequentially"""
        semantic_results = self.semantic_searcher.search(
            query, semantic_limit, filters, score_threshold
        )
        bm25_results = self.bm25_searcher.search(
            query, bm25_limit, filters, score_threshold
        )
        
        return semantic_results, bm25_results
    
    def multi_search(
        self,
        queries: List[str],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform multiple hybrid searches efficiently
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Optional filters to apply to all queries
            
        Returns:
            Dictionary mapping queries to their hybrid results
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
    
    def get_search_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed statistics about the hybrid search results
        
        Args:
            results: Results from search() method
            
        Returns:
            Dictionary with comprehensive statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Get fusion statistics
        fusion_stats = self.fusion.get_fusion_stats(results)
        
        # Calculate timing statistics
        if results and 'timing' in results[0]:
            timing_stats = {
                'search_time': results[0]['timing']['search_time'],
                'fusion_time': results[0]['timing']['fusion_time'],
                'total_time': results[0]['timing']['total_time']
            }
        else:
            timing_stats = {}
        
        # Analyze result sources
        semantic_ranks = []
        bm25_ranks = []
        
        for result in results:
            fusion_details = result.get('fusion_details', {})
            if fusion_details.get('semantic_rank'):
                semantic_ranks.append(fusion_details['semantic_rank'])
            if fusion_details.get('bm25_rank'):
                bm25_ranks.append(fusion_details['bm25_rank'])
        
        return {
            **fusion_stats,
            **timing_stats,
            'semantic_contribution': {
                'items_contributed': len(semantic_ranks),
                'avg_rank': sum(semantic_ranks) / len(semantic_ranks) if semantic_ranks else 0,
                'best_rank': min(semantic_ranks) if semantic_ranks else None
            },
            'bm25_contribution': {
                'items_contributed': len(bm25_ranks),
                'avg_rank': sum(bm25_ranks) / len(bm25_ranks) if bm25_ranks else 0,
                'best_rank': min(bm25_ranks) if bm25_ranks else None
            },
            'rrf_k_parameter': self.fusion.k,
            'parallel_search': self.parallel_search
        }
    
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
        """
        Format hybrid search results for display
        
        Args:
            results: Hybrid search results
            
        Returns:
            Formatted string for display
        """
        if not results:
            return "No results found."
        
        output = []
        output.append(f"\nFound {len(results)} hybrid search results:\n")
        output.append("=" * 80)
        
        for result in results:
            product = result["product"]
            rank = result["rank"]
            rrf_score = result["rrf_score"]
            relevance = result["relevance"]
            fusion_details = result.get("fusion_details", {})
            
            # Header with RRF score and sources
            sources = fusion_details.get('found_in', [])
            consensus = "[CONSENSUS]" if fusion_details.get('consensus', False) else f"[{'/'.join(sources).upper()}]"
            
            output.append(f"\n#{rank} - {relevance} Match (RRF: {rrf_score:.4f}) {consensus}")
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
            
            # Show fusion details
            semantic_rank = fusion_details.get('semantic_rank')
            bm25_rank = fusion_details.get('bm25_rank')
            if semantic_rank and bm25_rank:
                output.append(f"Rankings: Semantic #{semantic_rank}, BM25 #{bm25_rank}")
            elif semantic_rank:
                output.append(f"Rankings: Semantic #{semantic_rank}, BM25 not found")
            elif bm25_rank:
                output.append(f"Rankings: BM25 #{bm25_rank}, Semantic not found")
            
            # Show description if available
            description = product.get('description', '')
            if description:
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                output.append(f"Description: {desc_preview}")
        
        output.append("\n" + "=" * 80)
        return "\n".join(output)
    
    def test_connection(self) -> bool:
        """Test connections to both search engines"""
        try:
            semantic_ok = self.semantic_searcher.test_connection()
            bm25_ok = self.bm25_searcher.test_connection()
            
            return semantic_ok and bm25_ok
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False