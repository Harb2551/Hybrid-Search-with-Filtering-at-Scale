"""
Main script for testing semantic search functionality
Demonstrates how to use the SemanticSearcher class with class-based structure
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from semantic_search.searcher import SemanticSearcher


class SemanticSearchDemo:
    """
    Demo class for semantic search functionality
    Provides various testing and demonstration methods
    """
    
    def __init__(self):
        """Initialize the demo with default configuration"""
        
        # Configuration variables
        self.collection_name = "amazon_products"  # or "amazon_products_sample" for testing
        
        # Search parameters
        self.default_query = "wireless bluetooth headphones"
        self.default_top_k = 10
        self.score_threshold = 0.0
        self.verbose = True
        
        # Default filters
        self.default_filters = {
            "category": "Electronics",
            # "brand": "Sony",
            # "price_range": "50_to_100",
            # "has_image": True
        }
        
        # Initialize components
        self.config = None
        self.searcher = None
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
        """Initialize the semantic searcher"""
        try:
            self.config = EmbedderConfig(
                collection_name=self.collection_name
            )
            
            self.searcher = SemanticSearcher(self.config)
            self.logger.info("Semantic searcher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize searcher: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant"""
        print("Testing connection...")
        
        if not self.searcher.test_connection():
            print("ERROR: Connection failed!")
            return False
        
        print("SUCCESS: Connection successful!")
        
        # Get and display collection stats
        stats = self.searcher.get_collection_stats()
        if "points_count" in stats:
            print(f"Collection has {stats['points_count']:,} products")
        
        return True
    
    def setup_indexes(self) -> bool:
        """Create payload indexes for filtering"""
        print("Setting up payload indexes for filtering...")
        
        if self.searcher.create_payload_indexes():
            print("SUCCESS: Payload indexes created successfully!")
            return True
        else:
            print("ERROR: Failed to create payload indexes!")
            return False
    
    def basic_search(
        self,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_filters: bool = True
    ) -> bool:
        """
        Test basic semantic search functionality
        
        Args:
            query: Search query (uses default if None)
            top_k: Number of results (uses default if None)
            filters: Custom filters (uses default if None and use_filters=True)
            use_filters: Whether to apply filters
            
        Returns:
            True if successful, False otherwise
        """
        print("Basic Semantic Search Test")
        print("=" * 50)
        
        # Use defaults if not provided
        query = query or self.default_query
        top_k = top_k or self.default_top_k
        filters = filters if filters is not None else (self.default_filters if use_filters else None)
        
        print(f" Searching for: '{query}'")
        print(f" Parameters: top_k={top_k}, filters={filters}")
        
        try:
            results = self.searcher.search(
                query=query,
                top_k=top_k,
                filters=filters,
                score_threshold=self.score_threshold
            )
            
            # Display results
            if self.verbose:
                formatted_output = self.searcher.format_results_for_display(results)
                print(formatted_output)
            else:
                self._display_results_summary(results)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Search failed: {e}")
            return False
    
    def multi_search(self, queries: Optional[List[str]] = None) -> bool:
        """
        Test multiple search queries
        
        Args:
            queries: List of queries (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        print("\n Multi-Search Test")
        print("=" * 50)
        
        # Default queries
        queries = queries or [
            "bluetooth speaker",
            "running shoes",
            "laptop computer",
            "coffee maker"
        ]
        
        print(f" Searching for multiple queries: {queries}")
        
        try:
            results = self.searcher.multi_search(
                queries=queries,
                top_k=3,
                filters={"category": "Electronics"}
            )
            
            # Display results
            for query, query_results in results.items():
                print(f"\nQuery: Query: '{query}' - {len(query_results)} results")
                for i, result in enumerate(query_results, 1):
                    product = result["product"]
                    score = result["score"]
                    title = product.get('title', 'No title')
                    truncated_title = title[:50] + "..." if len(title) > 50 else title
                    print(f"  {i}. [{score:.3f}] {truncated_title}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Multi-search failed: {e}")
            return False
    
    def filtered_search_test(self) -> bool:
        """Test search with various filter combinations"""
        print("\n Filtered Search Test")
        print("=" * 50)
        
        # Test different filter combinations
        filter_tests = [
            {"name": "No filters", "filters": None},
            {"name": "Electronics only", "filters": {"category": "Electronics"}},
            {"name": "Clothing only", "filters": {"category": "Clothing, Shoes & Jewelry"}},
            {"name": "Under $50", "filters": {"price_range": "25_to_50"}},
            {"name": "With brand info", "filters": {"has_brand": True}},
        ]
        
        query = "wireless headphones"
        
        try:
            for test in filter_tests:
                print(f"\n Test: {test['name']}")
                print(f"   Filters: {test['filters']}")
                
                results = self.searcher.search(
                    query=query,
                    top_k=5,
                    filters=test['filters']
                )
                
                print(f"   Results: {len(results)} products found")
                if results:
                    # Show first result
                    first_result = results[0]
                    product = first_result["product"]
                    score = first_result["score"]
                    title = product.get('title', 'No title')
                    truncated_title = title[:60] + "..." if len(title) > 60 else title
                    print(f"   Top result: [{score:.3f}] {truncated_title}")
                    price = product.get('price', 0)
                    price_display = "Price not available" if price == 0 else f"${price:.2f}"
                    print(f"   Category: {product.get('category', 'Unknown')} | Price: {price_display}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Filtered search test failed: {e}")
            return False
    
    def interactive_search(self):
        """Interactive search mode"""
        print("\n Interactive Search Mode")
        print("=" * 50)
        print("Enter search queries (or 'quit' to exit)")
        
        while True:
            try:
                query = input("\n Search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                
                results = self.searcher.search(query, top_k=5)
                self._display_results_summary(results)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"ERROR: Error: {e}")
        
        print("\n Goodbye!")
    
    def _display_results_summary(self, results: List[Dict[str, Any]]):
        """Display a summary of search results"""
        print(f"\nFound Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            product = result["product"]
            score = result["score"]
            relevance = result["relevance"]
            title = product.get('title', 'No title')
            truncated_title = title[:60] + "..." if len(title) > 60 else title
            print(f"  {i}. [{score:.3f}] {relevance} - {truncated_title}")
            price = product.get('price', 0)
            price_display = "Price not available" if price == 0 else f"${price:.2f}"
            print(f"     {product.get('category', 'Unknown')} | Price: {price_display}")
    
    def run_all_tests(self) -> bool:
        """Run all demonstration tests"""
        print(" Semantic Search Demo")
        print("=" * 60)
        
        try:
            # Test connection
            if not self.test_connection():
                return False
            
            # Setup indexes for filtering
            if not self.setup_indexes():
                print("WARNING: Continuing without filtering capabilities...")
                # Continue without filtering - basic search should still work
            
            # Basic search test
            if not self.basic_search():
                print("ERROR: Basic search test failed!")
                return False
            
            # Multi-search test
            if not self.multi_search():
                print("ERROR: Multi-search test failed!")
                return False
            
            # Filtered search test
            if not self.filtered_search_test():
                print("ERROR: Filtered search test failed!")
                return False
            
            print("\nSUCCESS: All tests completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            return False
    
    def run_demo(self):
        """Run the complete demo with optional interactive mode"""
        try:
            # Run all tests
            if not self.run_all_tests():
                return 1
            
            # Optional interactive mode
            response = input("\n Would you like to try interactive search? (y/n): ")
            if response.lower().startswith('y'):
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
        demo = SemanticSearchDemo()
        return demo.run_demo()
    except Exception as e:
        print(f"ERROR: Failed to initialize demo: {e}")
        return 1


if __name__ == "__main__":
    exit(main())