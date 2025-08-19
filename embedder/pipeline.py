"""
Embedding Pipeline for E-commerce Hybrid Search
Orchestrates the entire process of loading, embedding, and storing products
"""

import os
import json
import gzip
from typing import Dict, Any, List, Iterator
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .config import EmbedderConfig
from .product_embedder import ProductEmbedder
from .qdrant_store import QdrantVectorStore


class EmbeddingPipeline:
    """
    Complete pipeline for embedding e-commerce products at scale
    Handles checkpointing, batch processing, and error recovery
    """
    
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedder = ProductEmbedder(config)
        self.vector_store = QdrantVectorStore(config)
        
        # Checkpoint management
        self.checkpoint_file = "embedded_product_ids.json"
        self.embedded_ids = self._load_checkpoint()
        
        self.logger.info(f"Pipeline initialized with {len(self.embedded_ids)} already processed products")
    
    def _load_checkpoint(self) -> set:
        """Load list of already processed product IDs"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                self.logger.warning(f"Error loading checkpoint: {e}")
        return set()
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(list(self.embedded_ids), f)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def load_products_from_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Load products from JSONL file (supports .gz compression)
        
        Args:
            file_path: Path to the product data file
            
        Yields:
            Product dictionaries
        """
        try:
            if file_path.endswith('.gz'):
                file_handle = gzip.open(file_path, 'rt', encoding='utf-8')
            else:
                file_handle = open(file_path, 'r', encoding='utf-8')
            
            with file_handle as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        product = json.loads(line)
                        yield product
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error loading products from {file_path}: {e}")
            raise
    
    def filter_unprocessed_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already processed products"""
        unprocessed = []
        for product in products:
            product_id = product.get('id', '')
            if product_id and product_id not in self.embedded_ids:
                unprocessed.append(product)
        return unprocessed
    
    def process_batch(self, products: List[Dict[str, Any]]) -> bool:
        """
        Process a batch of products: embed and store
        
        Args:
            products: Batch of products to process
            
        Returns:
            True if successful, False otherwise
        """
        if not products:
            return True
        
        try:
            # Generate embeddings
            self.logger.debug(f"Embedding batch of {len(products)} products")
            embeddings = self.embedder.embed_products(products)
            
            # Store in Qdrant
            self.logger.debug(f"Storing batch in Qdrant")
            success = self.vector_store.upsert_products(products, embeddings)
            
            if success:
                # Update checkpoint
                for product in products:
                    if product.get('id'):
                        self.embedded_ids.add(product['id'])
                
                self._save_checkpoint()
                return True
            else:
                self.logger.error("Failed to store batch in Qdrant")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return False
    
    def run_full_pipeline(self, force_recreate_collection: bool = False) -> Dict[str, Any]:
        """
        Run the complete embedding pipeline
        
        Args:
            force_recreate_collection: Whether to recreate the Qdrant collection
            
        Returns:
            Pipeline execution statistics
        """
        start_time = time.time()
        stats = {
            "start_time": start_time,
            "products_loaded": 0,
            "products_processed": 0,
            "products_skipped": 0,
            "batches_processed": 0,
            "batches_failed": 0,
            "errors": []
        }
        
        try:
            # Initialize Qdrant collection
            self.logger.info("Setting up Qdrant collection...")
            collection_created = self.vector_store.create_collection(force_recreate_collection)
            if collection_created:
                # Reset checkpoint if collection was recreated
                if force_recreate_collection:
                    self.embedded_ids = set()
                    self._save_checkpoint()
            
            # Log collection info
            collection_info = self.vector_store.get_collection_info()
            if collection_info:
                self.logger.info(f"Collection status: {collection_info.status}")
                self.logger.info(f"Current points: {collection_info.points_count}")
            
            # Load and process products
            self.logger.info(f"Loading products from {self.config.data_file_path}")
            
            batch = []
            batch_count = 0
            
            for product in self.load_products_from_file(self.config.data_file_path):
                stats["products_loaded"] += 1
                
                # Skip if already processed
                if product.get('id') in self.embedded_ids:
                    stats["products_skipped"] += 1
                    continue
                
                batch.append(product)
                
                # Process batch when it reaches chunk_size
                if len(batch) >= self.config.chunk_size:
                    batch_count += 1
                    
                    if self.process_batch(batch):
                        stats["products_processed"] += len(batch)
                        stats["batches_processed"] += 1
                        
                        if batch_count % 10 == 0:  # Log every 10 batches
                            self.logger.info(
                                f"Processed {stats['products_processed']} products "
                                f"({batch_count} batches)"
                            )
                    else:
                        stats["batches_failed"] += 1
                        stats["errors"].append(f"Failed to process batch {batch_count}")
                    
                    batch = []
                    
                    # Report progress
                    if stats["products_loaded"] % self.config.progress_report_interval == 0:
                        self.logger.info(
                            f"Progress: {stats['products_loaded']} loaded, "
                            f"{stats['products_processed']} processed, "
                            f"{stats['products_skipped']} skipped"
                        )
            
            # Process remaining products in final batch
            if batch:
                batch_count += 1
                if self.process_batch(batch):
                    stats["products_processed"] += len(batch)
                    stats["batches_processed"] += 1
                else:
                    stats["batches_failed"] += 1
                    stats["errors"].append(f"Failed to process final batch {batch_count}")
            
            # Final statistics
            end_time = time.time()
            stats["end_time"] = end_time
            stats["duration_seconds"] = end_time - start_time
            stats["duration_minutes"] = stats["duration_seconds"] / 60
            
            # Get final collection stats
            final_stats = self.vector_store.get_stats()
            stats["final_collection_stats"] = final_stats
            
            self.logger.info("=== PIPELINE COMPLETE ===")
            self.logger.info(f"Products loaded: {stats['products_loaded']}")
            self.logger.info(f"Products processed: {stats['products_processed']}")
            self.logger.info(f"Products skipped: {stats['products_skipped']}")
            self.logger.info(f"Duration: {stats['duration_minutes']:.2f} minutes")
            self.logger.info(f"Processing rate: {stats['products_processed']/(stats['duration_seconds']/60):.0f} products/minute")
            
            if final_stats and "points_count" in final_stats:
                self.logger.info(f"Total points in collection: {final_stats['points_count']}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            stats["errors"].append(str(e))
            stats["failed"] = True
            return stats
    
    def run_sample_pipeline(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Run pipeline on a sample of products for testing
        
        Args:
            sample_size: Number of products to process
            
        Returns:
            Pipeline execution statistics
        """
        self.logger.info(f"Running sample pipeline with {sample_size} products")
        
        # Temporarily modify chunk size for sample
        original_chunk_size = self.config.chunk_size
        self.config.chunk_size = min(100, sample_size // 10)
        
        try:
            stats = {"sample_size": sample_size, "products_processed": 0}
            
            # Create test collection
            test_collection_name = f"{self.config.collection_name}_sample"
            original_collection_name = self.config.collection_name
            self.config.collection_name = test_collection_name
            self.vector_store.collection_name = test_collection_name
            
            # Create collection
            self.vector_store.create_collection(force_recreate=True)
            
            # Process sample
            batch = []
            count = 0
            
            for product in self.load_products_from_file(self.config.data_file_path):
                if count >= sample_size:
                    break
                
                batch.append(product)
                count += 1
                
                if len(batch) >= self.config.chunk_size:
                    if self.process_batch(batch):
                        stats["products_processed"] += len(batch)
                    batch = []
            
            # Process final batch
            if batch:
                if self.process_batch(batch):
                    stats["products_processed"] += len(batch)
            
            # Get final stats
            stats["final_stats"] = self.vector_store.get_stats()
            
            self.logger.info(f"Sample pipeline complete: {stats['products_processed']} products processed")
            
            # Restore original collection name
            self.config.collection_name = original_collection_name
            self.vector_store.collection_name = original_collection_name
            
            return stats
            
        finally:
            # Restore original chunk size
            self.config.chunk_size = original_chunk_size
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "embedded_products_count": len(self.embedded_ids),
            "config": self.config.to_dict(),
            "embedder_info": self.embedder.get_model_info(),
            "collection_stats": self.vector_store.get_stats()
        }