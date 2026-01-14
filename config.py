import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Pathway
    PATHWAY_LICENSE_KEY = os.getenv("PATHWAY_LICENSE_KEY")
    
    # Database
    POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://user:password@localhost:5432/news_ai")
    
    # App Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # News Categories (Prioritizing Finance as requested)
    NEWS_CATEGORIES = [
        "Finance",
        "Technology",
        "Science",
        "Business",
        "Health",
        "Entertainment",
        "Sports",
        "Politics",
        "World"
    ]
    
    # Serper Settings
    SERPER_URL = "https://google.serper.dev/search"
    NEWS_FETCH_INTERVAL_SECONDS = 300  # 5 minutes
    
    # LLM Settings (OpenRouter)
    LLM_MODEL = "openai/gpt-4o"  # Defaulting to 4o via OpenRouter, can be changed
    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1000))
    
    # Embedding Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", 30))
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", 100))
    
    # Database Pool Settings
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 10))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", 20))
    
    # Pathway Settings
    PATHWAY_ENABLE_MONITORING = os.getenv("PATHWAY_ENABLE_MONITORING", "true").lower() == "true"
    PATHWAY_CACHE_DIR = os.getenv("PATHWAY_CACHE_DIR", "./cache")
    
    # RAG Settings
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))  # Number of documents to retrieve
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))  # Characters per chunk
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))  # Overlap between chunks
    
    # User Preference Settings
    PREFERENCE_SCORE_INCREMENT = float(os.getenv("PREFERENCE_SCORE_INCREMENT", 0.1))
    PREFERENCE_DECAY_FACTOR = float(os.getenv("PREFERENCE_DECAY_FACTOR", 0.95))
