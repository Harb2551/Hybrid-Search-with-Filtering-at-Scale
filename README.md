# Hybrid Search System

This module implements a **Hybrid Search Engine** that combines **Semantic Search** (dense vectors) and **BM25 Search** (sparse keywords) using **Reciprocal Rank Fusion (RRF)** for optimal search results.

## Overview

Hybrid search provides the best of both worlds:
- **Semantic Search**: Understands concepts and meaning ("noise cancelling audio gear" → headphones)
- **BM25 Search**: Excels at exact keyword matching ("Sony WH-1000XM4" → exact model)
- **RRF Fusion**: Combines results intelligently, boosting items found by both engines

## Architecture

```
Query: "wireless bluetooth headphones"
         ↓
    ┌────────────┐  ┌──────────────┐
    │ Semantic   │  │ BM25         │
    │ Search     │  │ Search       │
    │ (Qdrant)   │  │ (rank-bm25)  │
    └─────┬──────┘  └──────┬───────┘
          │                │
          ↓                ↓
      Top 50 results   Top 50 results
          │                │
          └────────┬───────┘
                   ↓
          ┌─────────────────┐
          │ RRF Fusion      │
          │ score = 1/(r+k) │
          └─────────────────┘
                   ↓
            Final Top 20 Results
            (Ranked by RRF Score)
```

## Components

### 1. HybridSearcher (`hybrid_searcher.py`)
- **Main search engine** that orchestrates the entire process
- **Parallel execution** of both search engines for speed
- **Filter-first architecture** for optimal performance
- **Comprehensive statistics** and timing information

### 2. RRFusion (`fusion_strategies.py`)
- **RRF implementation** using the formula: `score = 1/(rank + k)`
- **Product deduplication** using robust ID generation
- **Consensus detection** (items found by both engines)
- **Detailed fusion statistics**

### 3. Demo Script (`main.py`)
- **Comprehensive testing** of hybrid search functionality
- **Engine comparison** (Semantic vs BM25 vs Hybrid)
- **Interactive search mode** for real-time testing
- **Performance benchmarking**

## Usage

### Basic Hybrid Search

```python
from hybrid_search import HybridSearcher
from embedder.config import EmbedderConfig

# Initialize
config = EmbedderConfig(
    qdrant_url="your-qdrant-url",
    qdrant_api_key="your-api-key",
    collection_name="amazon_products"
)

searcher = HybridSearcher(
    config=config,
    bm25_index_path="path/to/bm25_index.pkl.gz",
    rrf_k=60
)

# Search
results = searcher.search(
    query="wireless bluetooth headphones",
    top_k=10,
    filters={"category": "Electronics"}
)

# Results include RRF scores and fusion details
for result in results:
    print(f"Product: {result['product']['title']}")
    print(f"RRF Score: {result['rrf_score']:.4f}")
    print(f"Found in: {result['fusion_details']['found_in']}")
    print(f"Consensus: {result['fusion_details']['consensus']}")
```

### Running the Demo

```bash
cd hybrid-search/hybrid_search
python main.py
```

**Demo includes:**
1. **🔗 Connection Test** - Verify both engines are ready
2. **🔍 Basic Hybrid Search** - Test RRF fusion
3. **⚖️ Engine Comparison** - Compare Semantic vs BM25 vs Hybrid
4. **🔍 Multi-Query Test** - Test various query types
5. **🔍 Interactive Mode** - Real-time search testing

## RRF (Reciprocal Rank Fusion) Explained

### Formula: `score = 1/(rank + k)`

- **rank**: Position in search results (1st, 2nd, 3rd, etc.)
- **k**: Smoothing parameter (default: 60)

### Example:

**Query**: "wireless headphones"

**Semantic Results:**
1. Sony WH-1000XM4 → RRF = 1/(1+60) = 0.0164
2. Bose QuietComfort → RRF = 1/(2+60) = 0.0161
3. Apple AirPods → RRF = 1/(3+60) = 0.0159

**BM25 Results:**
1. Apple AirPods → RRF = 1/(1+60) = 0.0164
2. Sony WH-1000XM4 → RRF = 1/(2+60) = 0.0161
3. JBL Headphones → RRF = 1/(3+60) = 0.0159

**Final Combined Scores:**
- **Sony WH-1000XM4**: 0.0164 + 0.0161 = **0.0325** 🎯 (Consensus)
- **Apple AirPods**: 0.0159 + 0.0164 = **0.0323** 🎯 (Consensus)
- **Bose QuietComfort**: 0.0161 + 0 = **0.0161** 📍 (Semantic only)
- **JBL Headphones**: 0 + 0.0159 = **0.0159** 📍 (BM25 only)

**Final Ranking**: Sony → Apple → Bose → JBL

## Key Features

### 🚀 Performance Optimizations
- **Parallel search execution** (both engines run simultaneously)
- **Filter-first architecture** (filter before search for speed)
- **Configurable retrieval limits** (get more candidates for better fusion)
- **Efficient product deduplication**

### 🎯 Search Quality
- **Consensus boost** (items found by both engines rank higher)
- **No score normalization needed** (rank-based fusion)
- **Handles different query types** (conceptual vs exact matches)
- **Comprehensive relevance scoring**

### 📊 Analytics & Insights
- **Detailed fusion statistics** (consensus percentage, source breakdown)
- **Performance timing** (search time, fusion time)
- **Engine contribution analysis** (which engine contributed what)
- **Search result provenance** (track where each result came from)

## Result Format

Each result includes:

```python
{
    "rank": 1,
    "rrf_score": 0.0325,
    "product": {
        "title": "Sony WH-1000XM4 Wireless Headphones",
        "brand": "Sony",
        "category": "Electronics",
        "price": 299.99,
        # ... other product fields
    },
    "fusion_details": {
        "semantic_rank": 1,        # Rank in semantic results
        "bm25_rank": 2,           # Rank in BM25 results  
        "semantic_score": 0.856,   # Original semantic score
        "bm25_score": 15.7,       # Original BM25 score
        "found_in": ["semantic", "bm25"],  # Which engines found it
        "consensus": True          # Found by both engines
    },
    "relevance": "Excellent",     # Human-readable relevance
    "timing": {
        "search_time": 2.145,     # Time for both searches
        "fusion_time": 0.023,     # Time for RRF fusion
        "total_time": 2.168       # Total time
    }
}
```

## Configuration Options

```python
HybridSearcher(
    config=config,                    # Qdrant and embedding config
    bm25_index_path="path/to/index",  # BM25 index location
    rrf_k=60,                        # RRF smoothing parameter
    parallel_search=True             # Run searches in parallel
)
```

### Search Parameters

```python
searcher.search(
    query="search query",            # Search query string
    top_k=20,                       # Final number of results
    filters={"category": "Electronics"},  # Optional filters
    semantic_top_k=50,              # Candidates from semantic search
    bm25_top_k=50,                  # Candidates from BM25 search
    score_threshold=0.0             # Minimum individual engine scores
)
```

## Performance

### Typical Performance (1.1M products):
- **Search Time**: ~2-3 seconds (parallel execution)
- **Fusion Time**: ~20-50ms (very fast)
- **Memory Usage**: ~2-3GB (both indexes loaded)
- **Consensus Rate**: ~30-50% (varies by query)

### Scalability:
- **Filter-first**: Dramatically improves performance on filtered queries
- **Parallel execution**: ~2x faster than sequential
- **Efficient deduplication**: Handles overlapping results well

## Integration

This hybrid search system is designed to work seamlessly with:
- **Qdrant Cloud** (1.1M product embeddings)
- **BM25 Index** (953MB compressed index)
- **Consistent filtering** (same filters across both engines)
- **Production APIs** (ready for web service integration)

## Next Steps

The hybrid search system provides the foundation for:
1. **Web API Service** - REST API for production use
2. **A/B Testing** - Compare hybrid vs individual engines
3. **Query Analysis** - Understand which queries benefit most from hybrid
4. **Performance Tuning** - Optimize RRF parameters for your data
5. **Advanced Features** - Query expansion, personalization, etc.