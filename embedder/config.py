"""
Configuration settings for the hybrid search embedder
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class EmbedderConfig:
    """Configuration for the embedder system"""
    
    # Embedding Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_sequence_length: int = 512
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Qdrant Settings
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_name: str = "amazon_products"
    
    # Collection Configuration
    distance_metric: str = "Cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    full_scan_threshold: int = 10000
    
    # Data Processing Settings
    data_file_path: str = "../data/amazon_products_1.1M.json.gz"
    chunk_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance Settings
    parallel_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "embedder.log"
    progress_report_interval: int = 1000
    
    @classmethod
    def from_env(cls) -> 'EmbedderConfig':
        """Create config from environment variables"""
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION", "amazon_products"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            device=os.getenv("DEVICE", "auto"),
            data_file_path=os.getenv("DATA_FILE_PATH", "../data/amazon_products_1.1M.json.gz"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            parallel_workers=int(os.getenv("PARALLEL_WORKERS", "4")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }