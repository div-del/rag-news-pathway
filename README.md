# Live AI News Platform

A real-time, personalized news analysis platform built with **Pathway** for the DataQuest 2026 hackathon at IIT Kharagpur.

## 🚀 Features

- **📰 Live News Ingestion**: Continuously fetches news from multiple sources via Serper API
- **🤖 Adaptive RAG**: Real-time Retrieval-Augmented Generation using Pathway's streaming framework
- **👤 Personalization**: Learns from user interactions to personalize news feeds
- **💬 Article Chat**: Chat with AI about specific articles with dynamic context expansion
- **🔄 Smart Comparisons**: Compare articles (Tesla vs BMW style) with AI-powered analysis
- **⚡ No Restarts**: Updates knowledge instantly without manual re-indexing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Live AI News Platform                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Serper    │───▶│   News      │───▶│   Pathway Document Store    │ │
│  │   API       │    │   Scraper   │    │   (Vector + Hybrid Index)   │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
│                                                    │                    │
│                                                    ▼                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      RAG Query Engine                            │   │
│  │  • Global queries (all articles)                                 │   │
│  │  • Article-specific chat (with context expansion)                │   │
│  │  • Multi-article comparison                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                            │                                            │
│  ┌───────────────┐         ▼          ┌────────────────────────────┐   │
│  │    User       │    ┌─────────┐     │  Recommendation Engine     │   │
│  │  Preferences  │◀──▶│   API   │◀───▶│  • Personalized feed       │   │
│  │    Engine     │    │  Server │     │  • Smart suggestions       │   │
│  └───────────────┘    └─────────┘     └────────────────────────────┘   │
│                            ▲                                            │
│                            │                                            │
│                    ┌───────────────┐                                    │
│                    │   Frontend    │                                    │
│                    │   (Web UI)    │                                    │
│                    └───────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
News AI/
├── api/
│   ├── main.py           # FastAPI application
│   ├── db_models.py      # SQLAlchemy database models
│   ├── db_utils.py       # Database utilities
│   └── init_db.py        # Database initialization
├── connectors/
│   ├── news_connector.py # Serper API integration
│   └── article_scraper.py # news-please scraper
├── pipeline/
│   └── document_pipeline.py # Pathway document processing
├── rag/
│   └── rag_engine.py     # RAG with multi-context support
├── user/
│   ├── user_profile.py   # User preference management
│   └── recommendation_engine.py # Personalization logic
├── frontend/
│   ├── index.html        # Web UI
│   ├── styles.css        # Styling
│   └── app.js            # Frontend logic
├── app.py                # Main entry point
├── config.py             # Configuration
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── requirements.txt      # Python dependencies
```

## 🛠️ Setup

### Prerequisites

- Docker and Docker Compose
- Serper API key (already configured)
- OpenRouter API key

### Quick Start with Docker

```bash
# Clone and navigate to the project
cd "News AI"

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Access the application
open http://localhost:8000
```

### Development Mode

```bash
# Start postgres only
docker-compose up -d postgres

# Run the app locally with hot reload
docker-compose --profile dev up app-dev
```

### Manual Setup

```bash
# Create virtual environment (Python 3.11+)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🔧 Configuration

Environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | Serper API key for news search | Required |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM | Required |
| `POSTGRES_CONNECTION_STRING` | PostgreSQL connection | `postgresql://postgres:postgres@localhost:5432/news_ai` |
| `HOST` | API server host | `0.0.0.0` |
| `PORT` | API server port | `8000` |
| `NEWS_FETCH_INTERVAL_SECONDS` | News fetch interval | `300` |

## 📡 API Endpoints

### News Feed
- `GET /api/news/feed` - Get personalized news feed
- `GET /api/news/article/{id}` - Get article details
- `GET /api/news/search` - Search articles
- `POST /api/news/fetch` - Trigger news fetching

### Chat (RAG)
- `POST /api/chat/query` - Query all articles
- `POST /api/chat/article/{id}` - Chat about specific article
- `POST /api/chat/compare` - Compare multiple articles

### User
- `POST /api/user/interaction` - Track interaction
- `GET /api/user/{id}/preferences` - Get preferences
- `GET /api/user/{id}/recommendations` - Get recommendations

### WebSocket
- `WS /ws/feed` - Real-time feed updates
- `WS /ws/chat/{session}` - Streaming chat

## 🎯 Demonstrating Live AI

The key hackathon requirement is showing **dynamic RAG behavior**:

1. **Start the app** and fetch initial news
2. **Ask a question** (e.g., "What's happening with Tesla?")
3. **Wait for new articles** to be ingested
4. **Ask the same question** - response will include new information!
5. **No restart needed** - Pathway handles incremental updates

## 🧪 Testing

```bash
# Fetch news
curl -X POST http://localhost:8000/api/news/fetch

# Query the RAG
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the latest in technology?"}'

# Compare articles
curl -X POST http://localhost:8000/api/chat/compare \
  -H "Content-Type: application/json" \
  -d '{"article_ids": ["id1", "id2"], "query": "Compare these"}'
```

## 📹 Demo Video

[Link to 3-minute demo video showing live RAG behavior]

## 🏆 Hackathon Criteria

| Criteria | Weight | How We Address It |
|----------|--------|-------------------|
| Real-Time Capability | 35% | Pathway streaming, instant updates, no restarts |
| Technical Implementation | 30% | Clean architecture, idiomatic Pathway usage |
| Innovation & UX | 20% | Personalization, smart comparisons, context expansion |
| Impact & Feasibility | 15% | Solves info overload, scalable design |

## 👥 Team

DataQuest 2026 - IIT Kharagpur

---

**Built with Pathway - The Live AI Framework**
