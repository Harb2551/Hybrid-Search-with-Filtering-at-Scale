"""
Main script for testing Hybrid Search functionality
Demonstrates RRF-based combination of Semantic and BM25 search
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from hybrid_search.hybrid_searcher import HybridSearcher


class HybridSearchDemo:
    """
    Demo class for hybrid search functionality
    Tests the combination of semantic and BM25 search using RRF
    """
    
    def __init__(self):
        """Initialize the hybrid search demo"""
        
        # Configuration
        self.collection_name = "amazon_products"
        self.bm25_index_path = "../bm25/bm25_index.pkl.gz"
        
        # Search parameters
        self.test_queries = [
            "wireless bluetooth headphones",
            "running shoes nike",
            "gaming laptop computer",
            "stainless steel coffee maker",
            "noise cancelling audio equipment"
        ]
        
        # Initialize components
        self.config = None
        self.hybrid_searcher = None
        self.logger = None
        
        self._setup_logging()
        self._initialize_searcher()
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_searcher(self):
        """Initialize the hybrid searcher"""
        try:
            # Check if BM25 index exists
            if not Path(self.bm25_index_path).exists():
                print(f"ERROR: BM25 index not found: {self.bm25_index_path}")
                print("Please build the BM25 index first by running: cd ../bm25 && python main.py")
                sys.exit(1)
            
            # Initialize configuration
            self.config = EmbedderConfig(
                collection_name=self.collection_name
            )
            
            # Initialize hybrid searcher
            self.hybrid_searcher = HybridSearcher(
                config=self.config,
                bm25_index_path=self.bm25_index_path,
                rrf_k=60,
                parallel_search=True
            )
            
            self.logger.info("Hybrid searcher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid searcher: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connections to both search engines"""
        print("Testing connections to search engines...")
        
        if not self.hybrid_searcher.test_connection():
            print("ERROR: Connection test failed!")
            return False
        
        print("SUCCESS: Both semantic and BM25 search engines connected successfully!")
        return True
    
    def basic_hybrid_search_test(self) -> bool:
        """Test basic hybrid search functionality"""
        try:
            print("\nBasic Hybrid Search Test")
            print("=" * 60)
            
            query = "wireless bluetooth headphones"
            print(f"Query: '{query}'")
            print(f"Using RRF fusion with k={self.hybrid_searcher.fusion.k}")
            
            # Perform hybrid search
            results = self.hybrid_searcher.search(
                query=query,
                top_k=10,
                filters={"category": "Electronics"}
            )
            
            if results:
                print(f"\nFound {len(results)} hybrid results:")
                
                for i, result in enumerate(results[:5], 1):
                    product = result["product"]
                    rrf_score = result["rrf_score"]
                    relevance = result["relevance"]
                    fusion_details = result["fusion_details"]
                    
                    title = product.get('title', 'No title')[:70]
                    brand = product.get('brand', 'Unknown')
                    price = product.get('price', 0)
                    
                    if price == 0:
                        price_display = "Price not available"
                    else:
                        try:
                            price_display = f"${float(price):.2f}"
                        except (ValueError, TypeError):
                            price_display = f"${price}"
                    
                    # Show which engines found this result
                    sources = fusion_details['found_in']
                    consensus = "[C] CONSENSUS" if fusion_details['consensus'] else f"[S] {'/'.join(sources).upper()}"
                    
                    print(f"\n  {i}. [{rrf_score:.4f}] {relevance} {consensus}")
                    print(f"     {title}")
                    print(f"     Brand: {brand} | Price: {price_display}")
                    
                    # Show individual rankings
                    semantic_rank = fusion_details.get('semantic_rank')
                    bm25_rank = fusion_details.get('bm25_rank')
                    if semantic_rank and bm25_rank:
                        print(f"     Rankings: Semantic #{semantic_rank}, BM25 #{bm25_rank}")
                    elif semantic_rank:
                        print(f"     Rankings: Semantic #{semantic_rank}, BM25 not found")
                    elif bm25_rank:
                        print(f"     Rankings: BM25 #{bm25_rank}, Semantic not found")
                
                # Show fusion statistics
                stats = self.hybrid_searcher.get_search_stats(results)
                print(f"\nFusion Statistics:")
                print(f"  - Consensus items: {stats['consensus_items']}/{stats['total_results']} ({stats['consensus_percentage']:.1f}%)")
                print(f"  - Semantic only: {stats['semantic_only']}")
                print(f"  - BM25 only: {stats['bm25_only']}")
                print(f"  - Search time: {stats.get('search_time', 0):.3f}s")
                print(f"  - Fusion time: {stats.get('fusion_time', 0):.3f}s")
                
            else:
                print("  No results found")
            
            print("\nSUCCESS: Basic hybrid search test completed!")
            return True
            
        except Exception as e:
            print(f"ERROR: Hybrid search test failed: {e}")
            return False
    
    def comparison_test(self) -> bool:
        """Compare hybrid vs individual search engines"""
        try:
            print("\nSearch Engine Comparison Test")
            print("=" * 60)
            
            query = "gaming laptop computer"
            top_k = 5
            filters = {"category": "Electronics"}
            
            print(f"Query: '{query}' (top_{top_k})")
            print(f"Filters: {filters}\n")
            
            # Get results from all three approaches
            print("Running searches...")
            
            # Semantic only
            semantic_results = self.hybrid_searcher.semantic_searcher.search(
                query, top_k, filters
            )
            
            # BM25 only  
            bm25_results = self.hybrid_searcher.bm25_searcher.search(
                query, top_k, filters
            )
            
            # Hybrid (RRF)
            hybrid_results = self.hybrid_searcher.search(
                query, top_k, filters
            )
            
            # Display comparison
            print(f"\nResults Comparison:")
            print(f"{'Rank':<4} {'Semantic':<30} {'BM25':<30} {'Hybrid (RRF)':<30}")
            print("-" * 100)
            
            for i in range(max(len(semantic_results), len(bm25_results), len(hybrid_results))):
                rank = i + 1
                
                # Semantic result
                if i < len(semantic_results):
                    sem_title = semantic_results[i]['product'].get('title', 'N/A')[:25] + "..."
                    sem_score = f"({semantic_results[i]['score']:.3f})"
                    semantic_display = f"{sem_title} {sem_score}"
                else:
                    semantic_display = "-"
                
                # BM25 result
                if i < len(bm25_results):
                    bm25_title = bm25_results[i]['product'].get('title', 'N/A')[:25] + "..."
                    bm25_score = f"({bm25_results[i]['score']:.2f})"
                    bm25_display = f"{bm25_title} {bm25_score}"
                else:
                    bm25_display = "-"
                
                # Hybrid result
                if i < len(hybrid_results):
                    hyb_title = hybrid_results[i]['product'].get('title', 'N/A')[:25] + "..."
                    hyb_score = f"({hybrid_results[i]['rrf_score']:.4f})"
                    consensus = "[C]" if hybrid_results[i]['fusion_details']['consensus'] else "[S]"
                    hybrid_display = f"{hyb_title} {hyb_score} {consensus}"
                else:
                    hybrid_display = "-"
                
                print(f"{rank:<4} {semantic_display:<30} {bm25_display:<30} {hybrid_display:<30}")
            
            print(f"\nLegend: [C] = Found by both engines, [S] = Found by one engine")
            print("\nSUCCESS: Comparison test completed!")
            return True
            
        except Exception as e:
            print(f"ERROR: Comparison test failed: {e}")
            return False
    
    def multi_query_test(self) -> bool:
        """Test multiple queries to show hybrid search effectiveness"""
        try:
            print("\nMulti-Query Hybrid Search Test")
            print("=" * 60)
            
            results = self.hybrid_searcher.multi_search(
                queries=self.test_queries[:3],
                top_k=3,
                filters={"category": "Electronics"}
            )
            
            for query, query_results in results.items():
                print(f"\n Query: '{query}'")
                print("-" * 40)
                
                if query_results:
                    for i, result in enumerate(query_results, 1):
                        product = result["product"]
                        rrf_score = result["rrf_score"]
                        fusion_details = result["fusion_details"]
                        
                        title = product.get('title', 'No title')[:60]
                        sources = fusion_details['found_in']
                        consensus = "[C]" if fusion_details['consensus'] else f"[S]{'/'.join(sources).upper()}"
                        
                        print(f"  {i}. [{rrf_score:.4f}] {consensus} {title}")
                else:
                    print("  No results found")
            
            print("\nSUCCESS: Multi-query test completed!")
            return True
            
        except Exception as e:
            print(f"ERROR: Multi-query test failed: {e}")
            return False
    
    def interactive_search(self):
        """Interactive hybrid search mode"""
        try:
            print("\nInteractive Hybrid Search")
            print("=" * 60)
            print("Enter search queries (or 'quit' to exit)")
            print("[C] = Consensus (found by both engines)")
            print("[S] = Single engine result")
            
            while True:
                try:
                    query = input("\n Hybrid Query: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    results = self.hybrid_searcher.search(query, top_k=5)
                    
                    if results:
                        print(f"\nFound Found {len(results)} hybrid results:")
                        for i, result in enumerate(results, 1):
                            product = result["product"]
                            rrf_score = result["rrf_score"]
                            relevance = result["relevance"]
                            fusion_details = result["fusion_details"]
                            
                            title = product.get('title', 'No title')[:60]
                            brand = product.get('brand', 'Unknown')
                            price = product.get('price', 0)
                            
                            if price == 0:
                                price_display = "Price not available"
                            else:
                                try:
                                    price_display = f"${float(price):.2f}"
                                except (ValueError, TypeError):
                                    price_display = f"${price}"
                            consensus = "[C]" if fusion_details['consensus'] else "[S]"
                            
                            print(f"  {i}. [{rrf_score:.4f}] {relevance} {consensus} {title}")
                            print(f"     Brand: {brand} | Price: {price_display}")
                    else:
                        print("No results found")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"ERROR: {e}")
            
            print("\nGoodbye!")
            
        except Exception as e:
            print(f"ERROR: Interactive search error: {e}")
    
    def run_demo(self):
        """Run the complete hybrid search demo"""
        try:
            print("Hybrid Search Demo")
            print("=" * 80)
            print("Combining Semantic Search + BM25 Search using RRF Fusion")
            print("=" * 80)
            
            # Test connections
            if not self.test_connection():
                return 1
            
            # Basic hybrid search test
            if not self.basic_hybrid_search_test():
                return 1
            
            # Comparison test
            if not self.comparison_test():
                return 1
            
            # Multi-query test
            if not self.multi_query_test():
                return 1
            
            print("\nSUCCESS: All hybrid search tests completed successfully!")
            
            # Optional interactive search
            interactive = input("\n Try interactive hybrid search? (y/n): ").lower().startswith('y')
            if interactive:
                self.interactive_search()
            
            return 0
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            return 1
        except Exception as e:
            print(f"ERROR: Demo failed: {e}")
            return 1


def main():
    """Main function"""
    try:
        demo = HybridSearchDemo()
        return demo.run_demo()
    except Exception as e:
        print(f"ERROR: Failed to initialize hybrid search demo: {e}")
        return 1


if __name__ == "__main__":
    exit(main())