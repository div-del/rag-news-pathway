
import asyncio
import logging
from rag.rag_engine import RAGEngine

# Mock Config to avoid environment issues
class MockConfig:
    OPENROUTER_API_KEY = "mock_key"
    LLM_BASE_URL = "https://mock.url"
    LLM_MODEL = "mock_model"
    RAG_TOP_K = 5 
    LLM_TEMPERATURE = 0
    LLM_MAX_TOKENS = 100

import config
config.Config = MockConfig

async def debug_search():
    engine = RAGEngine()
    
    # Add dummy documents similar to what might be in the system
    docs = [
        {
            "article_id": "1",
            "title": "Tech News: AI is taking over",
            "content": "Artificial Intelligence is growing rapidly.",
            "category": "Technology",
            "topics": ["AI", "Tech"]
        },
         {
            "article_id": "2",
            "title": "Finance Update: Stocks are up",
            "content": "The stock market hit a record high today.",
            "category": "Finance",
            "topics": ["Stocks", "Market"]
        }
    ]
    
    for doc in docs:
        engine.add_document(doc)
        
    print(f"Indexed {len(engine._documents)} documents.")
    
    queries = [
        "whats the news",
        "AI",
        "stocks",
        "technology"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = engine._simple_search(q, top_k=5)
        print(f"Results: {len(results)}")
        for r in results:
            print(f" - {r['title']} (Score details hidden)")

if __name__ == "__main__":
    asyncio.run(debug_search())
