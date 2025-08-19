# BM25 Keyword Search Module

This module provides BM25-based keyword search functionality for the hybrid search system. It uses the `rank-bm25` library for efficient indexing and search.

## Overview

BM25 (Best Matching 25) is a probabilistic ranking function used for keyword-based search. This module complements the semantic search by providing:

- **Exact keyword matching**
- **Traditional relevance scoring**
- **Fast term-based retrieval**
- **Query expansion capabilities**

## Components

### 1. BM25Indexer (`bm25_indexer.py`)
- **Purpose**: Builds BM25 indexes from product data
- **Features**:
  - Uses `rank-bm25` library for efficiency
  - NLTK-based text preprocessing
  - Stemming and stop word removal
  - Weighted field processing (title gets 3x weight)
  - Compressed index storage

### 2. BM25Searcher (`bm25_searcher.py`)
- **Purpose**: Performs BM25 keyword searches
- **Features**:
  - Fast keyword-based retrieval
  - Filtering support
  - Multi-query search
  - Relevance scoring
  - Result formatting

### 3. Demo Script (`main.py`)
- **Purpose**: Build and test BM25 functionality
- **Features**:
  - Index building from 1.1M products
  - Search testing
  - Interactive search mode
  - Performance statistics

## Installation

Install required dependencies:

```bash
pip install -r ../requirements.txt
```

Required packages:
- `rank-bm25==0.2.2` - BM25 implementation
- `nltk==3.8.1` - Text processing
- `scikit-learn==1.3.2` - Additional utilities

## Usage

### Building BM25 Index

```python
from bm25.bm25_indexer import BM25Indexer
from embedder.config import EmbedderConfig

# Initialize
config = EmbedderConfig()
indexer = BM25Indexer(config)

# Build index from data file
indexer.build_index_from_file("data/amazon_products_1.1M.json.gz")

# Save index
indexer.save_index("bm25/bm25_index.pkl.gz")
```

### Performing BM25 Search

```python
from bm25.bm25_searcher import BM25Searcher

# Initialize searcher
searcher = BM25Searcher(config, "bm25/bm25_index.pkl.gz")

# Search for products
results = searcher.search(
    query="wireless bluetooth headphones",
    top_k=10,
    filters={"category": "Electronics"}
)

# Display results
for result in results:
    product = result["product"]
    score = result["score"]
    print(f"[{score:.2f}] {product['title']}")
```

### Running Demo

```bash
cd hybrid-search/bm25
python main.py
```

The demo will:
1. **Build BM25 index** from 1.1M products (if not exists)
2. **Test search queries** with sample queries
3. **Offer interactive search** for real-time testing

## Text Processing Pipeline

### 1. Field Weighting
- **Title**: 3x weight (most important for keyword matching)
- **Brand**: 2x weight (important for exact matches)
- **Category**: 1x weight (contextual relevance)
- **Description**: 1x weight (detailed keywords)
- **Features**: 1x weight (specific attributes)

### 2. Tokenization
- Lowercase conversion
- Alphanumeric token extraction
- Stop word removal (English + product-specific)
- Porter stemming
- Token length filtering (2-20 characters)

### 3. Stop Words
Common words removed for better relevance:
- **English**: "the", "and", "is", "of", etc.
- **Product-specific**: "product", "brand", "new", "quality", etc.

## BM25 Scoring

### Score Interpretation
- **≥15.0**: Excellent match (exact keyword matches)
- **≥10.0**: Very Good match (strong keyword relevance)
- **≥5.0**: Good match (moderate keyword overlap)
- **≥2.0**: Fair match (some keyword matches)
- **≥1.0**: Poor match (weak keyword relevance)
- **<1.0**: Very Poor match (minimal relevance)

### BM25 Parameters
- **k1=1.5**: Term frequency saturation (default)
- **b=0.75**: Length normalization (default)

## Index Statistics

Typical index for 1.1M products:
- **Documents**: 1,100,000 products
- **Vocabulary**: ~200,000 unique terms
- **Avg doc length**: ~50 tokens
- **Index size**: ~150-200 MB compressed

## Integration with Hybrid Search

This BM25 module is designed to work with:
1. **Semantic Search** (dense vectors) - for conceptual matching
2. **Hybrid Fusion** - combining BM25 + semantic scores
3. **Filtering System** - consistent filter interface

## Performance

### Indexing Performance
- **1.1M products**: ~5-10 minutes on modern hardware
- **Memory usage**: ~2-4 GB during indexing
- **Storage**: ~150-200 MB compressed index

### Search Performance
- **Query time**: ~10-50ms for top-100 results
- **Memory usage**: ~500MB-1GB loaded index
- **Throughput**: ~100-500 queries/second

## Files Generated

- `bm25_index.pkl.gz` - Compressed BM25 index (~150-200 MB)
- NLTK downloads in `~/nltk_data/` (first run only)

## Troubleshooting

### Common Issues

1. **Missing NLTK data**:
   ```
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

2. **Memory errors during indexing**:
   - Use `max_products` parameter to limit dataset size
   - Increase system memory or use smaller batches

3. **Slow search performance**:
   - Ensure index is loaded from disk
   - Check available system memory
   - Consider index optimization

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

This BM25 module provides the keyword search foundation for:
1. **Hybrid Search System** - combining with semantic search
2. **Query Expansion** - using BM25 for related terms
3. **Faceted Search** - BM25 + metadata filtering
4. **Performance Benchmarking** - comparing search methods