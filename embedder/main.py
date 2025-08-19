"""
Main script for running the e-commerce product embedding pipeline
Demonstrates how to use the hybrid search embedder system
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to import embedder modules
sys.path.append(str(Path(__file__).parent.parent))

from embedder.config import EmbedderConfig
from embedder.pipeline import EmbeddingPipeline
from embedder.product_embedder import ProductEmbedder
from embedder.qdrant_store import QdrantVectorStore

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# =============================================================================

# Mode selection: "full", "sample", "search", "stats"
MODE = "sample"

# Configuration options


COLLECTION_NAME = "amazon_products"
DATA_FILE_PATH = "../data/amazon_products_1.1M.json.gz"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
CHUNK_SIZE = 1000
DEVICE = "auto"  # auto, cpu, cuda, mps

# Mode-specific options
FORCE_RECREATE = False  # Force recreate collection for full mode
SAMPLE_SIZE = 1000      # Sample size for sample mode
SEARCH_QUERY = "wireless headphones"  # Search query for search mode

# Logging options
LOG_LEVEL = "INFO"
LOG_FILE = None  # Set to filename if you want file logging

# =============================================================================


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure logging for the pipeline"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def run_full_pipeline(config: EmbedderConfig, force_recreate: bool = False):
    """Run the complete embedding pipeline"""
    
    print(" Starting E-commerce Product Embedding Pipeline")
    print("=" * 60)
    print(f" Data file: {config.data_file_path}")
    print(f" Model: {config.embedding_model}")
    print(f" Collection: {config.collection_name}")
    print(f" Qdrant URL: {config.qdrant_url}")
    print(f" Batch size: {config.batch_size}")
    print(f" Chunk size: {config.chunk_size}")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(config)
    
    # Run pipeline
    stats = pipeline.run_full_pipeline(force_recreate_collection=force_recreate)
    
    # Print results
    print("\nPIPELINE RESULTS")
    print("=" * 40)
    
    if stats.get("failed"):
        print("ERROR: Pipeline failed!")
        for error in stats.get("errors", []):
            print(f"   Error: {error}")
        return False
    
    print(f"SUCCESS: Pipeline completed successfully!")
    print(f"Products loaded: {stats['products_loaded']:,}")
    print(f"Products processed: {stats['products_processed']:,}")
    print(f"Products skipped: {stats['products_skipped']:,}")
    print(f"Duration: {stats['duration_minutes']:.2f} minutes")
    
    if stats['duration_seconds'] > 0:
        rate = stats['products_processed'] / (stats['duration_seconds'] / 60)
        print(f"🚄 Processing rate: {rate:.0f} products/minute")
    
    # Final collection stats
    final_stats = stats.get('final_collection_stats', {})
    if 'points_count' in final_stats:
        print(f"Total points in collection: {final_stats['points_count']:,}")
    
    return True


def run_sample_pipeline(config: EmbedderConfig, sample_size: int = 1000):
    """Run a sample pipeline for testing"""
    
    print(f"🧪 Starting Sample Pipeline ({sample_size:,} products)")
    print("=" * 50)
    
    pipeline = EmbeddingPipeline(config)
    stats = pipeline.run_sample_pipeline(sample_size)
    
    print("\nSAMPLE RESULTS")
    print("=" * 30)
    print(f"Target sample size: {stats['sample_size']:,}")
    print(f"Products processed: {stats['products_processed']:,}")
    
    final_stats = stats.get('final_stats', {})
    if 'points_count' in final_stats:
        print(f"Points in sample collection: {final_stats['points_count']:,}")
    
    return True


def test_search(config: EmbedderConfig, query: str = "wireless headphones"):
    """Test search functionality"""
    
    print(f"Testing search with query: '{query}'")
    print("=" * 40)
    
    try:
        # Initialize components
        embedder = ProductEmbedder(config)
        vector_store = QdrantVectorStore(config)
        
        # Check collection exists
        collection_info = vector_store.get_collection_info()
        if not collection_info:
            print("ERROR: Collection not found. Run the pipeline first.")
            return False
        
        print(f" Collection has {collection_info.points_count:,} products")
        
        # Create query embedding
        query_embedding = embedder.embed_query(query)
        
        # Search without filters
        print(f"\n Searching for '{query}' (no filters)...")
        results = vector_store.search_products(
            query_embedding=query_embedding,
            limit=5
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            product = result['product']
            score = result['score']
            print(f"  {i}. [{score:.3f}] {product.get('title', 'No title')}")
            print(f"     Category: {product.get('category', 'Unknown')}")
            print(f"     Brand: {product.get('brand', 'Unknown')}")
            print(f"     Price: ${product.get('price', 0):.2f}")
            print()
        
        # Search with filters
        print(f" Searching for '{query}' (Electronics only)...")
        filtered_results = vector_store.search_products(
            query_embedding=query_embedding,
            filters={"category": "Electronics"},
            limit=5
        )
        
        print(f"Found {len(filtered_results)} filtered results:")
        for i, result in enumerate(filtered_results, 1):
            product = result['product']
            score = result['score']
            print(f"  {i}. [{score:.3f}] {product.get('title', 'No title')}")
            print(f"     Brand: {product.get('brand', 'Unknown')}")
            print(f"     Price: ${product.get('price', 0):.2f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Search test failed: {e}")
        return False


def get_collection_stats(config: EmbedderConfig):
    """Get and display collection statistics"""
    
    print(" Collection Statistics")
    print("=" * 30)
    
    try:
        vector_store = QdrantVectorStore(config)
        stats = vector_store.get_stats()
        
        if stats.get("error"):
            print(f"ERROR: Error: {stats['error']}")
            return False
        
        print(f" Collection: {stats['collection_name']}")
        print(f" Points count: {stats['points_count']:,}")
        print(f" Vectors count: {stats['vectors_count']:,}")
        print(f" Indexed vectors: {stats['indexed_vectors_count']:,}")
        print(f" Status: {stats['status']}")
        
        config_info = stats.get('config', {})
        if config_info:
            print(f" Dimension: {config_info.get('dimension')}")
            print(f" Distance: {config_info.get('distance')}")
            print(f" HNSW M: {config_info.get('hnsw_m')}")
            print(f" EF Construct: {config_info.get('ef_construct')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to get stats: {e}")
        return False


def main():
    """Main function using configuration variables"""
    
    print(" E-commerce Product Embedding Pipeline")
    print(f" Mode: {MODE}")
    print("=" * 50)
    
    # Setup logging
    setup_logging(LOG_LEVEL, LOG_FILE)
    
    # Create configuration
    config = EmbedderConfig(
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        device=DEVICE,
        data_file_path=DATA_FILE_PATH
    )
    
    # Validate data file exists for modes that need it
    if MODE in ["full", "sample"] and not os.path.exists(config.data_file_path):
        print(f"ERROR: Data file not found: {config.data_file_path}")
        print("Make sure to run the data preparation scripts first!")
        return 1
    
    # Run selected mode
    try:
        if MODE == "full":
            success = run_full_pipeline(config, FORCE_RECREATE)
        elif MODE == "sample":
            success = run_sample_pipeline(config, SAMPLE_SIZE)
        elif MODE == "search":
            success = test_search(config, SEARCH_QUERY)
        elif MODE == "stats":
            success = get_collection_stats(config)
        else:
            print(f"ERROR: Unknown mode: {MODE}")
            print("Available modes: full, sample, search, stats")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"ERROR: Pipeline failed with error: {e}")
        logging.exception("Pipeline error")
        return 1


if __name__ == "__main__":
    exit(main())