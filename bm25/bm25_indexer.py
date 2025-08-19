"""
BM25 Indexer for E-commerce Product Search
Uses rank-bm25 library for efficient BM25 indexing and search
"""

import json
import gzip
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
except ImportError as e:
    print("Missing required packages. Install with:")
    print("pip install rank-bm25 nltk scikit-learn")
    raise e

from embedder.config import EmbedderConfig


class BM25Indexer:
    """
    BM25 indexer using rank-bm25 library
    Efficient indexing and search for product data
    """
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize BM25 indexer
        
        Args:
            config: EmbedderConfig for consistent settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self._setup_nltk()
        
        # BM25 components
        self.bm25 = None
        self.documents = {}  # doc_id -> product data
        self.tokenized_corpus = []  # tokenized documents for BM25
        self.doc_ids = []  # ordered list of document IDs
        
        # Text processing
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add product-specific stop words
        self.stop_words.update({
            'product', 'item', 'brand', 'new', 'free', 'shipping',
            'best', 'quality', 'premium', 'perfect', 'great', 'good',
            'available', 'order', 'buy', 'purchase', 'sale', 'price'
        })
        
        self.logger.info("BM25 indexer initialized with rank-bm25 library")
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip if token is stop word, too short, or too long
            if (token not in self.stop_words and 
                2 < len(token) < 20 and 
                token.isalnum()):
                
                # Apply stemming
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
        
        return processed_tokens
    
    def create_searchable_text(self, product: Dict[str, Any]) -> str:
        """
        Create searchable text from product data
        
        Args:
            product: Product dictionary
            
        Returns:
            Combined searchable text
        """
        text_parts = []
        
        # Title (highest weight)
        if product.get('title'):
            # Repeat title 3 times for higher weight
            title = str(product['title'])
            text_parts.extend([title] * 3)
        
        # Brand (high weight)
        if product.get('brand'):
            brand = str(product['brand'])
            text_parts.extend([brand] * 2)
        
        # Category
        if product.get('category'):
            category = product['category']
            if isinstance(category, list):
                text_parts.extend([str(cat) for cat in category])
            else:
                text_parts.append(str(category))
        
        # Description (limit length)
        if product.get('description'):
            desc = str(product['description'])[:1000]
            text_parts.append(desc)
        
        # Features (limit to first 5)
        if product.get('features'):
            features = product['features']
            if isinstance(features, list):
                text_parts.extend([str(f) for f in features[:5]])
            elif isinstance(features, str):
                text_parts.append(features[:500])
        
        return ' '.join(text_parts)
    
    def add_document(self, doc_id: str, product: Dict[str, Any]):
        """
        Add a document to the index
        
        Args:
            doc_id: Unique document identifier
            product: Product data dictionary
        """
        # Create searchable text
        searchable_text = self.create_searchable_text(product)
        
        # Preprocess text
        tokens = self.preprocess_text(searchable_text)
        
        if not tokens:
            self.logger.warning(f"No valid tokens for document {doc_id}")
            return
        
        # Store document
        self.documents[doc_id] = product.copy()
        self.tokenized_corpus.append(tokens)
        self.doc_ids.append(doc_id)
    
    def build_index(self):
        """
        Build the BM25 index from added documents
        """
        if not self.tokenized_corpus:
            self.logger.error("No documents to index")
            return False
        
        try:
            self.logger.info(f"Building BM25 index for {len(self.tokenized_corpus):,} documents...")
            
            # Create BM25 index
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            self.logger.info("BM25 index built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build BM25 index: {e}")
            return False
    
    def build_index_from_file(self, file_path: str, max_products: Optional[int] = None) -> bool:
        """
        Build BM25 index from compressed JSON file
        
        Args:
            file_path: Path to the compressed product data file
            max_products: Optional limit on number of products to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"Data file not found: {file_path}")
                return False
            
            self.logger.info(f"Building BM25 index from: {file_path}")
            
            products_processed = 0
            batch_size = 10000
            
            # Clear existing data
            self.documents.clear()
            self.tokenized_corpus.clear()
            self.doc_ids.clear()
            
            # Process file
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        product = json.loads(line.strip())
                        
                        # Create document ID
                        doc_id = f"prod_{line_num}"
                        
                        # Add to index
                        self.add_document(doc_id, product)
                        products_processed += 1
                        
                        # Progress logging
                        if products_processed % batch_size == 0:
                            self.logger.info(f"Processed {products_processed:,} products...")
                        
                        # Respect max limit
                        if max_products and products_processed >= max_products:
                            self.logger.info(f"Reached maximum product limit: {max_products:,}")
                            break
                            
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")
                        continue
            
            # Build the BM25 index
            if not self.build_index():
                return False
            
            self.logger.info(f"BM25 index built successfully with {products_processed:,} products")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build index from file: {e}")
            return False
    
    def save_index(self, index_path: str) -> bool:
        """
        Save the BM25 index to disk
        
        Args:
            index_path: Path where to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.bm25:
                self.logger.error("No BM25 index to save")
                return False
            
            index_data = {
                'bm25': self.bm25,
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'tokenized_corpus': self.tokenized_corpus,
                'stop_words': self.stop_words
            }
            
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(index_path, 'wb') as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = Path(index_path).stat().st_size / (1024 * 1024)
            self.logger.info(f"BM25 index saved to: {index_path} ({file_size:.1f} MB)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, index_path: str) -> bool:
        """
        Load BM25 index from disk
        
        Args:
            index_path: Path to the saved index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(index_path).exists():
                self.logger.error(f"Index file not found: {index_path}")
                return False
            
            with gzip.open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25 = index_data['bm25']
            self.documents = index_data['documents']
            self.doc_ids = index_data['doc_ids']
            self.tokenized_corpus = index_data['tokenized_corpus']
            self.stop_words = index_data.get('stop_words', self.stop_words)
            
            file_size = Path(index_path).stat().st_size / (1024 * 1024)
            self.logger.info(f"BM25 index loaded from: {index_path} ({file_size:.1f} MB)")
            self.logger.info(f"  - Documents: {len(self.documents):,}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_documents': len(self.documents),
            'has_index': self.bm25 is not None,
            'avg_doc_length': sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0,
            'vocabulary_size': len(set(token for doc in self.tokenized_corpus for token in doc)) if self.tokenized_corpus else 0,
            'stop_words_count': len(self.stop_words)
        }