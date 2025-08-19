"""
Simplified Streamlit Demo App for Hybrid Search System
Fast and direct - no caching, just create objects and search
"""

import streamlit as st
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Fix PyTorch-Streamlit compatibility issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our search modules
try:
    from semantic_search.searcher import SemanticSearcher
    from bm25.bm25_searcher import BM25Searcher
    from hybrid_search.hybrid_searcher import HybridSearcher
    from embedder.config import EmbedderConfig
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Hybrid Search Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def format_search_results(results: List[Dict[str, Any]], search_type: str) -> None:
    """Format and display search results"""
    if not results:
        st.warning("No results found for your query.")
        return
    
    st.success(f"Found {len(results)} results")
    
    # Display results
    for i, result in enumerate(results, 1):
        with st.expander(f"#{i} - {result.get('product', {}).get('title', 'No title')}", expanded=i <= 3):
            product = result.get('product', {})
            
            # Create columns for better layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Product details
                st.markdown(f"**Title:** {product.get('title', 'N/A')}")
                st.markdown(f"**Brand:** {product.get('brand', 'N/A')}")
                st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                
                # Price handling
                price_raw = product.get('price', 0)
                try:
                    if isinstance(price_raw, str):
                        price_clean = ''.join(c for c in price_raw if c.isdigit() or c == '.')
                        price = float(price_clean) if price_clean else 0
                    else:
                        price = float(price_raw) if price_raw else 0
                    
                    if price and price > 0:
                        st.markdown(f"**Price:** ${price:.2f}")
                    else:
                        st.markdown(f"**Price:** Not available")
                except (ValueError, TypeError):
                    st.markdown(f"**Price:** Not available")
                
                # Description
                description = product.get('description', '')
                if description:
                    desc_preview = description[:200] + "..." if len(description) > 200 else description
                    st.markdown(f"**Description:** {desc_preview}")
            
            with col2:
                # Search-specific information
                if search_type == "Hybrid":
                    fusion_details = result.get('fusion_details', {})
                    consensus = fusion_details.get('consensus', False)
                    found_in = fusion_details.get('found_in', [])
                    
                    if consensus:
                        st.success("CONSENSUS")
                        st.caption("Found by both engines")
                    else:
                        engine_text = " + ".join(found_in)
                        st.info(f"{engine_text.upper()}")
                        st.caption("Found by single engine")
                    
                    rrf_score = result.get('rrf_score', 0)
                    st.metric("RRF Score", f"{rrf_score:.4f}")
                
                else:
                    # Single engine results
                    score = result.get('score', 0)
                    relevance = result.get('relevance', 'Unknown')
                    
                    if search_type == "Semantic":
                        st.metric("Similarity Score", f"{score:.4f}")
                    else:  # BM25
                        st.metric("BM25 Score", f"{score:.3f}")
                    
                    st.caption(f"Relevance: {relevance}")

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("Hybrid Search Demo")
    st.markdown("**Search 1.1M Amazon Products using Semantic, Keyword, or Hybrid Search**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Search Configuration")
        
        # Number of results
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=10)
        
        # Filters
        st.subheader("Filters")
        
        # Category filter
        category_options = ["All", "Electronics", "Clothing"]
        selected_category = st.selectbox("Category", category_options)
        
        # Price range filter
        price_options = ["All", "Under $25", "$25-$50", "$50-$100", "$100-$200", "$200-$500", "Over $500"]
        selected_price = st.selectbox("Price Range", price_options)
        
        # Brand filter
        brand_filter = st.text_input("Brand (optional)", placeholder="e.g., Sony, Nike")
        
        # Build filters dictionary
        filters = {}
        if selected_category != "All":
            filters["category"] = selected_category
        
        if selected_price != "All":
            price_map = {
                "Under $25": "under_25",
                "$25-$50": "25_to_50", 
                "$50-$100": "50_to_100",
                "$100-$200": "100_to_200",
                "$200-$500": "200_to_500",
                "Over $500": "over_500"
            }
            filters["price_range"] = price_map[selected_price]
        
        if brand_filter.strip():
            filters["brand"] = brand_filter.strip()
        
        # Display active filters
        if filters:
            st.subheader("Active Filters")
            for key, value in filters.items():
                st.caption(f"**{key.title()}:** {value}")
    
    # Search interface
    st.header("Search Interface")
    
    # Query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., wireless bluetooth headphones, nike running shoes, stainless steel coffee maker",
        help="Try different types of queries to see how each search engine performs"
    )
    
    # Search type selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        semantic_btn = st.button("Semantic Search", use_container_width=True, type="primary")
        st.caption("Uses AI embeddings for meaning-based search")
    
    with col2:
        bm25_btn = st.button("BM25 Search", use_container_width=True, type="primary")
        st.caption("Traditional keyword-based search")
    
    with col3:
        hybrid_btn = st.button("Hybrid Search", use_container_width=True, type="primary")
        st.caption("Combines both approaches with RRF fusion")
    
    # Process search
    if query and (semantic_btn or bm25_btn or hybrid_btn):
        
        if semantic_btn:
            search_type = "Semantic"
            with st.spinner("Searching with semantic embeddings..."):
                start_time = time.time()
                try:
                    # Create semantic searcher directly - no caching
                    config = EmbedderConfig()
                    searcher = SemanticSearcher(config)
                    
                    results = searcher.search(
                        query=query,
                        top_k=top_k,
                        filters=filters if filters else None
                    )
                    search_time = time.time() - start_time
                    
                    # Display results immediately
                    st.header(f"Search Results - {search_type}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Search Time", f"{search_time:.2f}s")
                    with col2:
                        st.metric("Results Found", len(results))
                    
                    format_search_results(results, search_type)
                    
                except Exception as e:
                    st.error(f"Semantic search failed: {e}")
        
        elif bm25_btn:
            search_type = "BM25"
            with st.spinner("Searching with BM25 keyword matching..."):
                start_time = time.time()
                try:
                    # Create BM25 searcher directly - no caching
                    config = EmbedderConfig()
                    bm25_index_path = "bm25/bm25_index.pkl.gz"
                    searcher = BM25Searcher(config, bm25_index_path)
                    
                    results = searcher.search(
                        query=query,
                        top_k=top_k,
                        filters=filters if filters else None
                    )
                    search_time = time.time() - start_time
                    
                    # Display results immediately
                    st.header(f"Search Results - {search_type}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Search Time", f"{search_time:.2f}s")
                    with col2:
                        st.metric("Results Found", len(results))
                    
                    format_search_results(results, search_type)
                    
                except Exception as e:
                    st.error(f"BM25 search failed: {e}")
        
        elif hybrid_btn:
            search_type = "Hybrid"
            with st.spinner("Searching with hybrid fusion (Semantic + BM25)..."):
                start_time = time.time()
                try:
                    # Create hybrid searcher directly - no caching
                    config = EmbedderConfig()
                    bm25_index_path = "bm25/bm25_index.pkl.gz"
                    searcher = HybridSearcher(
                        config=config,
                        bm25_index_path=bm25_index_path,
                        rrf_k=60,
                        parallel_search=True
                    )
                    
                    results = searcher.search(
                        query=query,
                        top_k=top_k,
                        filters=filters if filters else None
                    )
                    search_time = time.time() - start_time
                    
                    # Display results immediately
                    st.header(f"Search Results - {search_type}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Search Time", f"{search_time:.2f}s")
                    with col2:
                        st.metric("Results Found", len(results))
                    with col3:
                        if results:
                            consensus_count = sum(1 for r in results 
                                                if r.get('fusion_details', {}).get('consensus', False))
                            consensus_pct = (consensus_count / len(results)) * 100
                            st.metric("Consensus Rate", f"{consensus_pct:.1f}%")
                    
                    format_search_results(results, search_type)
                    
                except Exception as e:
                    st.error(f"Hybrid search failed: {e}")

if __name__ == "__main__":
    main()