# Hybrid Search Demo - Streamlit App

Interactive web interface for demonstrating semantic, BM25, and hybrid search on 1.1M Amazon products.

## Quick Start

### 1. Install Dependencies
```bash
cd hybrid-search
pip install -r requirements.txt
```

### 2. Set Up Environment
Ensure your `.env` file exists with Qdrant credentials:
```bash
QDRANT_URL="your-qdrant-url"
QDRANT_API_KEY="your-api-key"
```

### 3. Ensure BM25 Index Exists
The app requires a pre-built BM25 index:
```bash
# If index doesn't exist, build it first
cd bm25
python main.py
cd ..
```

### 4. Launch the Demo
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### Search Modes
- **Semantic Search**: AI-powered meaning-based search using embeddings
- **BM25 Search**: Traditional keyword-based search with TF-IDF scoring  
- **Hybrid Search**: RRF fusion combining both approaches

### Filtering Options
- **Category**: Electronics, Clothing, or All
- **Price Range**: Under $25, $25-$50, $50-$100, $100-$200, $200-$500, Over $500
- **Brand**: Free text brand filtering

### Result Features
- **Hybrid Results**: Shows consensus indicators ([C] = both engines, [S] = single engine)
- **Performance Metrics**: Search time, result count, consensus rate
- **Detailed Product Info**: Title, brand, category, price, description, features
- **Relevance Scoring**: Different scoring systems for each search type

## Demo Queries

### Excellent for Semantic Search:
- "comfortable running gear"
- "something for music while jogging"
- "home office setup equipment"
- "stylish winter clothing"

### Excellent for BM25 Search:
- "Sony WH-1000XM4"
- "Nike Air Force 1"
- "iPhone 13 Pro Max"
- "Samsung Galaxy S21"

### Great for Hybrid Search:
- "wireless bluetooth headphones"
- "nike running shoes"
- "stainless steel coffee maker"
- "gaming laptop computer"

## Architecture Overview

```
User Query → Streamlit Interface
     ↓
┌────────────────────────────────────┐
│ Three Search Options:              │
│ 1. Semantic (Qdrant Vector DB)    │
│ 2. BM25 (rank-bm25 library)      │  
│ 3. Hybrid (RRF Fusion)           │
└────────────────────────────────────┘
     ↓
Results Display with Filtering
```

## Technical Details

### Dataset
- **Size**: 1.1M Amazon products
- **Categories**: Electronics + Clothing
- **Fields**: title, brand, category, price, description, features

### Search Engines
- **Semantic**: all-MiniLM-L6-v2 embeddings (384D) in Qdrant Cloud
- **BM25**: rank-bm25 with NLTK preprocessing and stemming
- **Hybrid**: RRF fusion with k=60 parameter

### Performance
- **Search Time**: 1-3 seconds typical
- **Memory Usage**: ~2GB for vector embeddings + 954MB for BM25 index
- **Concurrent Users**: Streamlit handles multiple sessions

## Troubleshooting

### Common Issues

**"Failed to initialize search engines"**
- Check that `.env` file exists with correct Qdrant credentials
- Verify BM25 index exists at `bm25/bm25_index.pkl.gz`
- Ensure all dependencies are installed

**"BM25 index not found"**
```bash
cd bm25
python main.py  # This builds the index
```

**"Connection to Qdrant failed"**
- Verify QDRANT_URL and QDRANT_API_KEY in .env file
- Check internet connection for Qdrant Cloud access
- Ensure Qdrant collection exists with embeddings

**"Slow initial load"**
- First load initializes all search engines (cached afterward)
- BM25 index loading takes 5-10 seconds
- Semantic search connection setup takes 2-3 seconds

### Performance Tips
- Use specific queries for better BM25 results
- Use conceptual queries for better semantic results
- Try hybrid search for balanced results
- Apply filters to reduce search space

## Demo Script for Presentations

### 1. Introduction (30 seconds)
"This is our hybrid search system running on 1.1M Amazon products. You can choose between three search approaches."

### 2. Semantic Demo (60 seconds)
Query: "comfortable running gear"
- Show semantic understanding
- Highlight non-keyword matches

### 3. BM25 Demo (60 seconds)  
Query: "Sony WH-1000XM4"
- Show exact keyword matching
- Demonstrate precision for product names

### 4. Hybrid Demo (90 seconds)
Query: "wireless bluetooth headphones"
- Show consensus indicators
- Explain RRF fusion benefits
- Compare with individual engines

### 5. Filtering Demo (30 seconds)
- Apply category and price filters
- Show real-time result updates

## File Structure
```
hybrid-search/
├── streamlit_app.py          # Main Streamlit application
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── bm25/bm25_index.pkl.gz   # Pre-built BM25 index
├── semantic_search/          # Semantic search module
├── bm25/                     # BM25 search module  
├── hybrid_search/            # Hybrid fusion module
└── embedder/                 # Configuration and utilities
```

## Next Steps
- Add more advanced filtering options
- Implement query suggestions
- Add search analytics and logging
- Create A/B testing framework
- Add export functionality for results