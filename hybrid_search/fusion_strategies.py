"""
Fusion Strategies for Hybrid Search
Implements RRF (Reciprocal Rank Fusion) for combining search results
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict


class RRFusion:
    """
    Reciprocal Rank Fusion (RRF) implementation
    Combines multiple ranked lists using the formula: score = 1/(rank + k)
    
    Used by Google, Microsoft, and other major search engines for hybrid search
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion
        
        Args:
            k: Smoothing parameter (typically 60), prevents division by small numbers
        """
        self.k = k
        self.logger = logging.getLogger(__name__)
    
    def fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and BM25 results using Score-Aware RRF
        
        This implementation incorporates actual relevance scores via min-max normalization
        to reduce artificial alternating patterns in hybrid search results.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            max_results: Maximum number of results to return
            
        Returns:
            List of fused results sorted by score-aware RRF score
        """
        try:
            # Normalize scores within each result set
            semantic_normalized = self._normalize_scores(semantic_results)
            bm25_normalized = self._normalize_scores(bm25_results)
            
            # Create mapping of product_id to combined scores
            rrf_scores = defaultdict(lambda: {
                'rrf_score': 0.0,
                'semantic_rank': None,
                'bm25_rank': None,
                'semantic_score': 0.0,
                'bm25_score': 0.0,
                'product': None,
                'found_in': set()
            })
            
            # Process semantic results with score-aware RRF
            for rank, result in enumerate(semantic_normalized, 1):
                product = result["product"]
                product_id = self._get_product_id(product)
                
                # Score-aware RRF: boost RRF score by normalized relevance
                base_rrf = 1.0 / (rank + self.k)
                normalized_score = result.get("normalized_score", 0.5)
                score_aware_rrf = base_rrf * (1.0 + normalized_score)
                
                rrf_scores[product_id]['rrf_score'] += score_aware_rrf
                rrf_scores[product_id]['semantic_rank'] = rank
                rrf_scores[product_id]['semantic_score'] = result.get("score", 0.0)
                rrf_scores[product_id]['product'] = product
                rrf_scores[product_id]['found_in'].add('semantic')
            
            # Process BM25 results with score-aware RRF
            for rank, result in enumerate(bm25_normalized, 1):
                product = result["product"]
                product_id = self._get_product_id(product)
                
                # Score-aware RRF: boost RRF score by normalized relevance
                base_rrf = 1.0 / (rank + self.k)
                normalized_score = result.get("normalized_score", 0.5)
                score_aware_rrf = base_rrf * (1.0 + normalized_score)
                
                rrf_scores[product_id]['rrf_score'] += score_aware_rrf
                rrf_scores[product_id]['bm25_rank'] = rank
                rrf_scores[product_id]['bm25_score'] = result.get("score", 0.0)
                
                # If not found in semantic, store product data
                if product_id not in [self._get_product_id(r["product"]) for r in semantic_results]:
                    rrf_scores[product_id]['product'] = product
                
                rrf_scores[product_id]['found_in'].add('bm25')
            
            # Convert to list and sort by RRF score
            fused_results = []
            for product_id, data in rrf_scores.items():
                if data['product'] is None:
                    continue
                
                result = {
                    'rank': 0,  # Will be set after sorting
                    'rrf_score': data['rrf_score'],
                    'product': data['product'],
                    'fusion_details': {
                        'semantic_rank': data['semantic_rank'],
                        'bm25_rank': data['bm25_rank'],
                        'semantic_score': data['semantic_score'],
                        'bm25_score': data['bm25_score'],
                        'found_in': list(data['found_in']),
                        'consensus': len(data['found_in']) > 1
                    },
                    'relevance': self._rrf_score_to_relevance(data['rrf_score'])
                }
                fused_results.append(result)
            
            # Sort by RRF score (descending)
            fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
            
            # Add final ranks
            for i, result in enumerate(fused_results[:max_results], 1):
                result['rank'] = i
            
            self.logger.info(f"RRF fusion completed: {len(fused_results)} unique products")
            return fused_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"RRF fusion failed: {e}")
            raise
    
    def _get_product_id(self, product: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for a product
        Uses multiple fields to create a robust ID
        """
        # Try different ID strategies
        if product.get('id'):
            return str(product['id'])
        elif product.get('product_id'):
            return str(product['product_id'])
        else:
            # Create ID from title + brand + category
            title = str(product.get('title', ''))[:50]
            brand = str(product.get('brand', ''))
            category = str(product.get('category', ''))
            if isinstance(category, list):
                category = str(category[0]) if category else ''
            
            # Create a deterministic hash-like ID
            id_components = [title, brand, category]
            product_id = '_'.join(c.replace(' ', '_') for c in id_components if c)
            return product_id[:100]  # Limit length
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores within a result set to 0-1 range using min-max normalization
        
        Args:
            results: Search results with scores
            
        Returns:
            Results with added normalized_score field (0.0 to 1.0)
        """
        if not results:
            return results
        
        # Extract scores
        scores = [r.get("score", 0.0) for r in results]
        if not scores:
            return results
        
        # Calculate min-max normalization parameters
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        # Handle edge case: all scores are identical
        if score_range == 0:
            # All scores identical, assign neutral normalized score
            normalized_results = []
            for result in results:
                result_copy = result.copy()
                result_copy['normalized_score'] = 0.5  # Neutral score
                normalized_results.append(result_copy)
            return normalized_results
        
        # Apply min-max normalization: (score - min) / (max - min)
        normalized_results = []
        for result in results:
            result_copy = result.copy()
            raw_score = result.get("score", 0.0)
            
            # Min-max normalization formula
            normalized_score = (raw_score - min_score) / score_range
            
            # Ensure score is in valid range [0.0, 1.0]
            normalized_score = max(0.0, min(1.0, normalized_score))
            
            result_copy['normalized_score'] = normalized_score
            normalized_results.append(result_copy)
        
        return normalized_results
    
    def _rrf_score_to_relevance(self, rrf_score: float) -> str:
        """Convert RRF score to human-readable relevance"""
        if rrf_score >= 0.025:
            return "Excellent"
        elif rrf_score >= 0.020:
            return "Very Good"
        elif rrf_score >= 0.015:
            return "Good"
        elif rrf_score >= 0.010:
            return "Fair"
        elif rrf_score >= 0.005:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_fusion_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the fusion results
        
        Args:
            results: Fused results from fuse_results()
            
        Returns:
            Dictionary with fusion statistics
        """
        if not results:
            return {}
        
        consensus_count = sum(1 for r in results if r['fusion_details']['consensus'])
        semantic_only = sum(1 for r in results if r['fusion_details']['found_in'] == ['semantic'])
        bm25_only = sum(1 for r in results if r['fusion_details']['found_in'] == ['bm25'])
        
        avg_rrf_score = sum(r['rrf_score'] for r in results) / len(results)
        
        return {
            'total_results': len(results),
            'consensus_items': consensus_count,
            'semantic_only': semantic_only,
            'bm25_only': bm25_only,
            'consensus_percentage': (consensus_count / len(results)) * 100,
            'average_rrf_score': avg_rrf_score,
            'rrf_k_parameter': self.k,
            'top_score': results[0]['rrf_score'] if results else 0,
            'score_range': {
                'max': max(r['rrf_score'] for r in results) if results else 0,
                'min': min(r['rrf_score'] for r in results) if results else 0
            }
        }