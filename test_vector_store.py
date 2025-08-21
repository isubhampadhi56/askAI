#!/usr/bin/env python3
"""
Test script to verify VectorStore functionality
"""

import os
import tempfile
import sys
sys.path.insert(0, 'doc-parser')
from vector_store import VectorStore

def test_vector_store():
    # Create a temporary SQLite database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Use SQLite for testing (in production, use PostgreSQL with pgvector)
        db_url = f"sqlite:///{db_path}"
        
        # Initialize vector store
        vector_store = VectorStore(db_url)
        
        # Test adding texts
        texts = [
            "This is a test document about artificial intelligence",
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks for complex tasks"
        ]
        sources = ["test1.txt", "test2.txt", "test3.txt"]
        
        vector_store.add_texts(texts, sources)
        print("✓ Texts added successfully")
        
        # Test search functionality
        results = vector_store.search("neural networks", k=2)
        print("✓ Search completed successfully")
        print(f"Search results: {results}")
        
        # Test document parsing methods
        # Create a test text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
            f.write("This is a test text file content")
            txt_path = f.name
        
        try:
            content = vector_store._read_txt(txt_path)
            print(f"✓ Text file reading: {content[:50]}...")
        finally:
            os.unlink(txt_path)
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    test_vector_store()
