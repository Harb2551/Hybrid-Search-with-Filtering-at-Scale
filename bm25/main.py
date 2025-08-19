"""
Main script for building and testing BM25 index
Creates BM25 index from Amazon product data
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from bm25.bm25_indexer import BM25Indexer
from bm25.bm25_searcher import BM25Searcher


class BM25Demo:
    """
    Demo class for BM25 indexing and searching
    """
    
    def __init__(self):
        """Initialize the BM25 demo"""
        self.data_file = "../data/amazon_products_1.1M.json.gz"
        self.index_file = "bm25_index.pkl.gz"
        
        # Initialize configuration
        self.config = EmbedderConfig()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.indexer = None
        self.searcher = None
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def build_index(self, max_products: Optional[int] = None) -> bool:
        """
        Build BM25 index from product data
        
        Args:
            max_products: Optional limit on number of products to index
            
        Returns:
            True if successful
        """
        try:
            print("Building BM25 Index")
            print("=" * 50)
            
            # Check if data file exists
            if not Path(self.data_file).exists():
                print(f"ERROR: Data file not found: {self.data_file}")
                print("Please ensure the Amazon product data is available at the specified path.")
                return False
            
            # Initialize indexer
            self.indexer = BM25Indexer(self.config)
            
            # Build index from file
            print(f"Data file: {self.data_file}")
            if max_products:
                print(f"Max products: {max_products:,}")
            
            success = self.indexer.build_index_from_file(
                file_path=self.data_file,
                max_products=max_products
            )
            
            if not success:
                print("ERROR: Failed to build BM25 index")
                return False
            
            # Save index
            print(f" Saving index to: {self.index_file}")
            if not self.indexer.save_index(self.index_file):
                print("ERROR: Failed to save BM25 index")
                return False
            
            # Show stats
            stats = self.indexer.get_index_stats()
            print("\n Index Statistics:")
            print(f"  - Total documents: {stats['total_documents']:,}")
            print(f"  - Vocabulary size: {stats['vocabulary_size']:,}")
            print(f"  - Average doc length: {stats['avg_doc_length']:.1f} tokens")
            print(f"  - Stop words: {stats['stop_words_count']:,}")
            
            print("\nSUCCESS: BM25 index built successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Error building index: {e}")
            return False
    
    def test_search(self) -> bool:
        """
        Test BM25 search functionality
        
        Returns:
            True if successful
        """
        try:
            print("\n Testing BM25 Search")
            print("=" * 50)
            
            # Check if index exists
            if not Path(self.index_file).exists():
                print(f"ERROR: Index file not found: {self.index_file}")
                print("Please build the index first using build_index()")
                return False
            
            # Initialize searcher
            self.searcher = BM25Searcher(self.config, self.index_file)
            
            # Test queries with different filters
            test_cases = [
                {
                    "query": "wireless bluetooth headphones",
                    "filters": {"category": "Electronics"},
                    "description": "Electronics only"
                },
                {
                    "query": "running shoes",
                    "filters": {"price_range": "25_to_50"},
                    "description": "$25-$50 price range"
                },
                {
                    "query": "laptop computer",
                    "filters": {"has_brand": True},
                    "description": "With brand info"
                },
                {
                    "query": "bluetooth speaker",
                    "filters": None,
                    "description": "No filters"
                }
            ]
            
            for test_case in test_cases:
                query = test_case["query"]
                filters = test_case["filters"]
                description = test_case["description"]
                
                print(f"\n Query: '{query}' ({description})")
                print(f" Filters: {filters}")
                print("-" * 40)
                
                results = self.searcher.search(query, top_k=3, filters=filters)
                
                if results:
                    for i, result in enumerate(results, 1):
                        product = result["product"]
                        score = result["score"]
                        relevance = result["relevance"]
                        
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
                        
                        print(f"  {i}. [{score:.2f}] {relevance} - {title}")
                        print(f"     Brand: {brand} | Price: {price_display}")
                else:
                    print("  No results found")
            
            print("\nSUCCESS: BM25 search test completed!")
            return True
            
        except Exception as e:
            print(f"ERROR: Error testing search: {e}")
            return False
    
    def test_filtering(self) -> bool:
        """
        Test filtering functionality specifically
        
        Returns:
            True if successful
        """
        try:
            print("\n Testing BM25 Filtering")
            print("=" * 50)
            
            if not self.searcher:
                if not Path(self.index_file).exists():
                    print(f"ERROR: Index file not found: {self.index_file}")
                    return False
                self.searcher = BM25Searcher(self.config, self.index_file)
            
            query = "headphones"
            
            # Test 1: No filters vs Category filter
            print(f"\n Filter Test 1: Category Filtering")
            print("-" * 40)
            
            # No filters
            results_no_filter = self.searcher.search(query, top_k=5)
            print(f"Without filters: Found {len(results_no_filter)} results")
            if results_no_filter:
                categories = [r["product"].get("category", "Unknown") for r in results_no_filter[:3]]
                print(f"  Categories: {categories}")
            
            # Electronics filter
            results_electronics = self.searcher.search(query, top_k=5, filters={"category": "Electronics"})
            print(f"Electronics only: Found {len(results_electronics)} results")
            if results_electronics:
                # Validate all results are Electronics
                all_electronics = True
                for result in results_electronics:
                    category = result["product"].get("category", "")
                    if "Electronics" not in str(category):
                        all_electronics = False
                        break
                print(f"  SUCCESS: All results are Electronics: {all_electronics}")
            
            # Test 2: Price range filtering
            print(f"\n Filter Test 2: Price Range Filtering")
            print("-" * 40)
            
            results_price = self.searcher.search(query, top_k=5, filters={"price_range": "25_to_50"})
            print(f"$25-$50 range: Found {len(results_price)} results")
            if results_price:
                # Validate price ranges
                prices_valid = True
                prices = []
                for result in results_price:
                    price = result["product"].get("price", 0)
                    prices.append(price)
                    if isinstance(price, (int, float)) and price > 0:
                        if not (25 <= price < 50):
                            prices_valid = False
                print(f"  Prices found: {prices[:3]}")
                print(f"  SUCCESS: All prices in $25-$50 range: {prices_valid}")
            
            # Test 3: Brand filtering
            print(f"\n Filter Test 3: Brand Filtering")
            print("-" * 40)
            
            results_brand = self.searcher.search(query, top_k=5, filters={"has_brand": True})
            print(f"With brands: Found {len(results_brand)} results")
            if results_brand:
                # Validate all have brands
                brands_valid = True
                brands = []
                for result in results_brand:
                    brand = result["product"].get("brand", "")
                    brands.append(brand)
                    if not brand:
                        brands_valid = False
                print(f"  Brands found: {brands[:3]}")
                print(f"  SUCCESS: All results have brands: {brands_valid}")
            
            # Test 4: Combined filters
            print(f"\n Filter Test 4: Combined Filters")
            print("-" * 40)
            
            combined_filters = {
                "category": "Electronics",
                "has_brand": True,
                "price_range": "25_to_50"
            }
            results_combined = self.searcher.search(query, top_k=5, filters=combined_filters)
            print(f"Combined filters: Found {len(results_combined)} results")
            print(f"  Filters: {combined_filters}")
            
            if results_combined:
                # Validate combined criteria
                for i, result in enumerate(results_combined[:2], 1):
                    product = result["product"]
                    category = str(product.get("category", ""))
                    brand = product.get("brand", "")
                    price = product.get("price", 0)
                    
                    print(f"  Result {i}:")
                    print(f"    Category: {category} ({'SUCCESS:' if 'Electronics' in category else 'ERROR:'})")
                    print(f"    Brand: {brand} ({'SUCCESS:' if brand else 'ERROR:'})")
                    print(f"    Price: ${price} ({'SUCCESS:' if isinstance(price, (int, float)) and 25 <= price < 50 else 'ERROR:'})")
            
            print("\nSUCCESS: BM25 filtering test completed!")
            return True
            
        except Exception as e:
            print(f"ERROR: Error testing filtering: {e}")
            return False
    
    def interactive_search(self):
        """Interactive BM25 search mode"""
        try:
            if not self.searcher:
                if not Path(self.index_file).exists():
                    print(f"ERROR: Index file not found: {self.index_file}")
                    return
                
                self.searcher = BM25Searcher(self.config, self.index_file)
            
            print("\n Interactive BM25 Search")
            print("=" * 50)
            print("Enter search queries (or 'quit' to exit)")
            
            while True:
                try:
                    query = input("\n BM25 Query: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    results = self.searcher.search(query, top_k=5)
                    
                    if results:
                        print(f"\nFound Found {len(results)} BM25 results:")
                        for i, result in enumerate(results, 1):
                            product = result["product"]
                            score = result["score"]
                            relevance = result["relevance"]
                            
                            title = product.get('title', 'No title')[:70]
                            brand = product.get('brand', 'Unknown')
                            category = product.get('category', 'Unknown')
                            price = product.get('price', 0)
                            
                            if price == 0:
                                price_display = "Price not available"
                            else:
                                try:
                                    price_display = f"${float(price):.2f}"
                                except (ValueError, TypeError):
                                    price_display = f"${price}"
                            
                            print(f"  {i}. [{score:.2f}] {relevance} - {title}")
                            print(f"     Category: {category} | Brand: {brand} | Price: {price_display}")
                    else:
                        print("No results found")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"ERROR: Error: {e}")
            
            print("\n Goodbye!")
            
        except Exception as e:
            print(f"ERROR: Interactive search error: {e}")
    
    def run_demo(self):
        """Run the complete BM25 demo"""
        try:
            print(" BM25 Search Demo")
            print("=" * 60)
            
            # Check if index exists
            if Path(self.index_file).exists():
                print(f"SUCCESS: Found existing BM25 index: {self.index_file}")
                
                # Ask if user wants to rebuild
                rebuild = input(" Rebuild index? (y/n): ").lower().startswith('y')
                if rebuild:
                    if not self.build_index():
                        return 1
            else:
                print(" No existing index found. Building new index...")
                if not self.build_index():
                    return 1
            
            # Test search
            if not self.test_search():
                return 1
            
            # Test filtering
            if not self.test_filtering():
                print("WARNING: Filtering test failed, but continuing...")
            
            # Optional interactive search
            interactive = input("\n Try interactive BM25 search? (y/n): ").lower().startswith('y')
            if interactive:
                self.interactive_search()
            
            return 0
            
        except KeyboardInterrupt:
            print("\n Demo interrupted by user")
            return 1
        except Exception as e:
            print(f"ERROR: Demo failed: {e}")
            return 1


def main():
    """Main function"""
    try:
        demo = BM25Demo()
        return demo.run_demo()
    except Exception as e:
        print(f"ERROR: Failed to initialize BM25 demo: {e}")
        return 1


if __name__ == "__main__":
    exit(main())