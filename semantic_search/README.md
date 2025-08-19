# Semantic Search Module

This module provides semantic search capabilities for the hybrid search system, allowing natural language queries against the Qdrant vector store containing embedded product data.

## 🏗️ Architecture

```
semantic_search/
├── __init__.py          # Package initialization
├── searcher.py          # SemanticSearcher class
├── main.py             # Demo and testing (class-based)
└── README.md           # This file
```

## 🚀 Quick Start

### Basic Search
```python
from hybrid_search.semantic_search import SemanticSearcher
from hybrid_search.embedder.config import EmbedderConfig

# Configure connection
config = EmbedderConfig(
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-api-key",
    collection_name="amazon_products"
)

# Initialize searcher
searcher = SemanticSearcher(config)

# Perform search
results = searcher.search(
    query="wireless bluetooth headphones",
    top_k=10,
    filters={"category": "Electronics"}
)

# Display results
formatted_output = searcher.format_results_for_display(results)
print(formatted_output)
```

### Run Demo
```bash
cd hybrid-search/semantic_search
python main.py
```

## 🔍 Search Features

### 1. **Basic Semantic Search**
```python
results = searcher.search(
    query="wireless headphones",
    top_k=10,
    score_threshold=0.7
)
```

### 2. **Filtered Search**
```python
results = searcher.search(
    query="running shoes",
    top_k=5,
    filters={
        "category": "Clothing, Shoes & Jewelry",
        "price_range": "50_to_100",
        "has_brand": True
    }
)
```

### 3. **Multi-Query Search**
```python
results = searcher.multi_search(
    queries=["laptop", "headphones", "mouse"],
    top_k=5,
    filters={"category": "Electronics"}
)
```

### 4. **Quick Search Function**
```python
from hybrid_search.semantic_search.searcher import quick_search

results = quick_search(
    query="bluetooth speaker",
    top_k=5
)
```

## 🎛️ Search Parameters

### **Core Parameters**
- `query`: Natural language search query
- `top_k`: Number of results to return (1-100)
- `score_threshold`: Minimum similarity score (0.0-1.0)
- `filters`: Optional filters for refined search

### **Available Filters**
- `category`: Product category (e.g., "Electronics")
- `brand`: Brand name (e.g., "Sony")
- `price_range`: Predefined ranges ("under_25", "25_to_50", etc.)
- `min_price` / `max_price`: Custom price range
- `has_image`: Products with images (boolean)
- `has_brand`: Products with brand info (boolean)

## 📊 Search Results Format

```python
[
    {
        "rank": 1,
        "score": 0.95,
        "relevance": "Excellent",
        "product": {
            "product_id": "B001",
            "title": "Sony WH-1000XM4 Headphones",
            "category": "Electronics",
            "brand": "Sony",
            "price": 249.99,
            "description": "Industry-leading noise canceling...",
            "features": ["30-hour battery", "Touch controls"],
            "image_url": "https://...",
            "has_image": True,
            "has_brand": True,
            "price_range": "200_to_500"
        }
    }
]
```

## 🎯 Relevance Scoring

The system provides human-readable relevance labels:

| Score Range | Relevance |
|-------------|-----------|
| 0.9 - 1.0   | Excellent |
| 0.8 - 0.9   | Very Good |
| 0.7 - 0.8   | Good      |
| 0.6 - 0.7   | Fair      |
| 0.5 - 0.6   | Poor      |
| 0.0 - 0.5   | Very Poor |

## 🧪 Testing & Demo

### **SemanticSearchDemo Class**
The [`main.py`](main.py) file contains a comprehensive demo class:

```python
demo = SemanticSearchDemo()

# Run all tests
demo.run_all_tests()

# Test specific functionality
demo.basic_search("laptop computer", top_k=5)
demo.multi_search(["headphones", "speakers"])
demo.filtered_search_test()

# Interactive mode
demo.interactive_search()
```

### **Available Tests**
- **Connection Test**: Verify Qdrant connectivity
- **Basic Search**: Single query with/without filters
- **Multi-Search**: Multiple queries simultaneously
- **Filtered Search**: Various filter combinations
- **Interactive Mode**: Real-time query testing

## 🔧 Configuration

The module reuses [`EmbedderConfig`](../embedder/config.py) for consistency:

```python
config = EmbedderConfig(
    # Qdrant settings
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-api-key",
    collection_name="amazon_products",
    
    # Model settings
    embedding_model="all-MiniLM-L6-v2",
    device="auto",  # auto, cpu, cuda, mps
    
    # Performance
    batch_size=32,
    timeout=30
)
```

## 📈 Performance

### **Expected Performance**
- **Search Latency**: <100ms for filtered queries
- **Memory Usage**: ~500MB for model + client
- **Throughput**: ~50-100 queries/second
- **Accuracy**: 90%+ relevance with proper filtering

### **Optimization Tips**
1. **Use Filters**: Reduce search space for better quality
2. **Reasonable top_k**: Don't request more results than needed
3. **Score Thresholds**: Filter out low-quality results
4. **Batch Queries**: Use `multi_search()` for multiple queries

## 🚨 Error Handling

The module includes comprehensive error handling:

```python
try:
    results = searcher.search("invalid query")
except ValueError as e:
    print(f"Invalid input: {e}")
except ConnectionError as e:
    print(f"Qdrant connection failed: {e}")
except Exception as e:
    print(f"Search failed: {e}")
```

## 🔗 Integration

### **With Embedder Module**
```python
# Shared configuration
from hybrid_search.embedder.config import EmbedderConfig

config = EmbedderConfig.from_env()  # Load from environment
searcher = SemanticSearcher(config)
```

### **With Future Hybrid Components**
The semantic search module is designed to integrate with:
- **BM25 Search**: For keyword-based search
- **Fusion Engine**: To combine semantic + keyword results
- **API Layer**: FastAPI wrapper for web service
- **Frontend**: Streamlit/React interfaces

This semantic search module provides the foundation for the dense vector search component of your hybrid search system!