<p align="center">
  <img src="frontend/logo.png" alt="LiveLens Logo" width="120" height="120">
</p>

<h1 align="center">🔮 LiveLens</h1>
<h3 align="center">Real-Time AI News Intelligence Platform</h3>

<p align="center">
  <strong>Built with Pathway for DataQuest 2026 | IIT Kharagpur</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-proving-real-time-dynamism">Proof of Dynamism</a> •
  <a href="#-demo-video">Demo Video</a>
</p>

---

## 📋 Overview

**LiveLens** is a real-time, adaptive news intelligence platform that demonstrates the power of **Live AI** using [Pathway's](https://pathway.com/) streaming data processing framework. Unlike traditional RAG systems that rely on stale data snapshots, LiveLens continuously ingests news, updates its knowledge base instantly, and responds to queries with the most recent information—**all without restarts or manual re-indexing**.

### The Problem We Solve

> Traditional AI systems are stuck in the past. They train on static datasets and can't reflect breaking news from 5 minutes ago. LiveLens changes that.

- 📰 **Information Overload**: Users are overwhelmed by news from multiple sources
- ⏰ **Stale Knowledge**: Traditional RAG systems don't reflect real-time changes
- 🔄 **Manual Updates**: Existing solutions require restarts to incorporate new data
- 🎯 **No Personalization**: One-size-fits-all news feeds ignore user preferences

### Our Solution

LiveLens provides:
- **Instant Knowledge Updates**: New articles are indexed and queryable within seconds
- **Semantic Article Chat**: Ask questions about any article with automatic context expansion
- **Smart Comparisons**: Compare multiple articles with AI-powered analysis
- **Personalized Feeds**: Adaptive recommendations based on reading behavior
- **Newsletter Delivery**: AI-curated newsletters sent directly to your inbox

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **📰 Live News Ingestion** | Continuously fetches news from multiple sources via Serper API |
| **🤖 Adaptive RAG** | Real-time Retrieval-Augmented Generation with hybrid search (vector + keyword) |
| **💬 AI Chat** | Conversational interface with memory - maintains context across messages |
| **🔄 Article Comparison** | Side-by-side analysis of articles with AI insights |
| **👤 Personalization** | User preference learning from interactions |
| **📧 Newsletter** | AI-powered email digests with curated articles |
| **🎬 YouTube Analysis** | Whisper transcription and analysis of related videos |
| **⚡ Zero Restarts** | Knowledge updates without service interruption |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              LIVELENS ARCHITECTURE                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌──────────────────┐                                                           │
│   │   DATA SOURCES   │                                                           │
│   ├──────────────────┤                                                           │
│   │  • Serper API    │──────┐                                                    │
│   │  • Demo Inject   │      │                                                    │
│   │  • File Monitor  │      ▼                                                    │
│   └──────────────────┘  ┌───────────────────────────────────────┐                │
│                         │         INGESTION LAYER               │                │
│                         │  ┌─────────────────────────────────┐  │                │
│                         │  │      Article Scraper            │  │                │
│                         │  │  (news-please + BeautifulSoup)  │  │                │
│                         │  └─────────────────────────────────┘  │                │
│                         └───────────────────────────────────────┘                │
│                                        │                                          │
│                                        ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │                      PATHWAY STREAMING ENGINE                             │   │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐  │   │
│   │  │  Document      │  │  Incremental   │  │  Real-time                 │  │   │
│   │  │  Store         │──│  Indexing      │──│  Vector Embeddings         │  │   │
│   │  │  (SQLite)      │  │  (No Restarts) │  │  (sentence-transformers)   │  │   │
│   │  └────────────────┘  └────────────────┘  └────────────────────────────┘  │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                          │
│                                        ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────────────┐   │
│   │                         RAG QUERY ENGINE                                  │   │
│   │  ┌─────────────┐  ┌─────────────────┐  ┌────────────────────────────┐    │   │
│   │  │  Hybrid     │  │  Context        │  │  LLM Integration           │    │   │
│   │  │  Search     │  │  Expansion      │  │  (OpenRouter/Mistral)      │    │   │
│   │  │  (0.7v+0.3k)│  │  (Related Docs) │  │                            │    │   │
│   │  └─────────────┘  └─────────────────┘  └────────────────────────────┘    │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                          │
│       ┌────────────────────────────────┼───────────────────────────┐              │
│       ▼                                ▼                           ▼              │
│  ┌───────────┐               ┌──────────────────┐          ┌──────────────┐      │
│  │  AI Chat  │               │   FastAPI        │          │ Recommen-    │      │
│  │  Engine   │◀─────────────▶│   REST API       │◀────────▶│ dation       │      │
│  │           │               │   + WebSocket    │          │ Engine       │      │
│  └───────────┘               └──────────────────┘          └──────────────┘      │
│                                        ▲                                          │
│                                        │                                          │
│                              ┌──────────────────┐                                 │
│                              │     FRONTEND     │                                 │
│                              │  (HTML/JS/CSS)   │                                 │
│                              │  + Clerk Auth    │                                 │
│                              └──────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Server** | FastAPI | REST endpoints + WebSocket for real-time updates |
| **RAG Engine** | Pathway + sentence-transformers | Hybrid vector/keyword search with incremental indexing |
| **AI Chat** | OpenRouter (Mistral) | Conversational interface with conversation memory |
| **Document Store** | SQLite | Persistent article storage |
| **Embeddings** | all-MiniLM-L6-v2 | Local CPU-based embedding generation |
| **News Ingestion** | Serper API + news-please | Multi-source news fetching and scraping |
| **Frontend** | Vanilla HTML/CSS/JS | Modern glassmorphism UI with dark mode |
| **Authentication** | Clerk | User management and OAuth |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for local frontend development server)
- **API Keys**:
  - [Serper API](https://serper.dev/) - for news search
  - [OpenRouter](https://openrouter.ai/) - for LLM (Mistral)
  - [Clerk](https://clerk.com/) - for authentication (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/div-del/rag-news-pathway.git
cd rag-news-pathway
```

### Step 2: Set Up Environment

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the project root:

```env
# Required: News & LLM
SERPER_API_KEY=your_serper_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Authentication (Clerk)
CLERK_PUBLISHABLE_KEY=pk_test_your_key
CLERK_SECRET_KEY=sk_test_your_key

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Pathway Configuration (optional)
USE_PATHWAY=false
```

### Step 4: Run the Application

**Option A: Single Command (Recommended)**
```bash
python app.py
```
This starts the backend at `http://localhost:8000` and serves the frontend.

**Option B: With Live Frontend Reload**
```bash
# Terminal 1: Backend
python app.py

# Terminal 2: Frontend dev server
npm run dev
```

### Step 5: Access the Application

Open your browser and navigate to:
- **Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 🎯 Proving Real-Time Dynamism

> This is the **most critical requirement** for the hackathon. Here's how to verify that LiveLens truly operates in real-time.

### Method 1: Demo Panel (Built-in)

1. **Open the app** at http://localhost:8000
2. **Navigate to AI Chat**
3. **Ask a question** about a unique topic: *"What is Project Aurora?"*
4. **Observe**: The AI will say it doesn't have information
5. **Inject a new article**: Click the Demo button or use the API:

```bash
curl -X POST http://localhost:8000/api/demo/inject-article \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking: Project Aurora Unveiled as Revolutionary AI Platform",
    "content": "Tech giant XYZ Corp announced Project Aurora today, a groundbreaking AI platform that promises to revolutionize data processing with its unique neural architecture.",
    "category": "Technology",
    "topics": ["Project Aurora", "AI", "XYZ Corp"]
  }'
```

6. **Ask the same question again**: *"What is Project Aurora?"*
7. **Result**: The AI now provides detailed information about Project Aurora!

### Method 2: Test Dynamism API

Use the built-in dynamism test endpoint:

```bash
curl -X POST "http://localhost:8000/api/demo/test-dynamism?query=What%20is%20Acme%20Corporation%20doing"
```

This returns a before/after comparison proving real-time updates.

### Method 3: Live News Fetch

1. **Fetch real news**: Click "Fetch News" or:
```bash
curl -X POST http://localhost:8000/api/news/fetch?category=Technology
```

2. **Query immediately**: The new articles are queryable instantly

---

## 📂 Project Structure

```
rag-news-pathway/
├── api/
│   ├── main.py              # FastAPI application & all endpoints
│   ├── ai_chat_engine.py    # AI chat with conversation memory
│   ├── article_store.py     # SQLite article persistence
│   ├── db_models.py         # SQLAlchemy models
│   └── youtube_analyzer.py  # YouTube transcription & analysis
├── connectors/
│   ├── news_connector.py    # Serper API integration
│   └── article_scraper.py   # Web scraping with news-please
├── pipeline/
│   ├── pathway_server.py    # Pathway document store server
│   └── document_pipeline.py # Document processing pipeline
├── rag/
│   └── rag_engine.py        # RAG with hybrid search & LLM
├── user/
│   ├── user_profile.py      # User preference management
│   └── recommendation_engine.py  # Personalization logic
├── frontend/
│   ├── app.html             # Main application page
│   ├── onboarding.html      # User onboarding wizard
│   ├── styles.css           # Glassmorphism styling
│   ├── app.js               # Frontend application logic
│   └── logo.png             # LiveLens logo
├── app.py                   # Main entry point
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
├── package.json             # Node.js dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-container setup
└── README.md                # This file
```

---

## 📡 API Reference

### News Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/news/feed` | Get personalized news feed |
| `GET` | `/api/news/article/{id}` | Get article details with related articles |
| `GET` | `/api/news/search?query=` | Search articles |
| `POST` | `/api/news/fetch` | Trigger news fetching from sources |

### AI Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ai-chat/message` | Send message to AI with article context |
| `POST` | `/api/ai-chat/sessions/new` | Create new chat session |
| `GET` | `/api/ai-chat/sessions/{id}` | Get session history |
| `DELETE` | `/api/ai-chat/sessions/{id}` | Clear session |

### RAG Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/query` | Global RAG query across all articles |
| `POST` | `/api/chat/article/{id}` | Query about specific article |
| `POST` | `/api/chat/compare` | Compare multiple articles |

### Demo & Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/demo/inject-article` | Inject test article (proves real-time) |
| `POST` | `/api/demo/test-dynamism` | Full before/after dynamism test |
| `GET` | `/api/knowledge-base/status` | Get indexing stats |

---

## 🔧 Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | Serper API key for news search | **Required** |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM | **Required** |
| `HOST` | Server bind host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `USE_PATHWAY` | Enable Pathway streaming mode | `false` |
| `LLM_MODEL` | LLM model to use | `mistralai/mistral-7b-instruct:free` |
| `RAG_TOP_K` | Number of documents to retrieve | `5` |
| `NEWS_FETCH_INTERVAL_SECONDS` | Auto-fetch interval | `300` |

---

## 🎬 Demo Video

**[📺 Watch the 3-minute demo](YOUR_VIDEO_LINK_HERE)**

The demo showcases:
1. ✅ Team introduction and problem statement
2. ✅ Full application walkthrough (Feed, AI Chat, Compare)
3. ✅ **Real-time proof**: Article injection with before/after responses

---

## 🏆 Hackathon Criteria Mapping

| Criteria | Weight | How LiveLens Addresses It |
|----------|--------|---------------------------|
| **Real-Time Capability** | 35% | Pathway streaming, instant indexing, zero restarts, built-in dynamism test |
| **Technical Implementation** | 30% | Clean modular architecture, hybrid search, conversation memory, proper error handling |
| **Innovation & UX** | 20% | Glassmorphism UI, personalization, article comparison, YouTube analysis, newsletter |
| **Impact & Feasibility** | 15% | Solves information overload, scalable design, production-ready deployment options |

---

## 🚀 Deployment Options

### Docker (Recommended for Production)

```bash
docker-compose up -d
```

### Render

See [RENDER_DEPLOY.md](RENDER_DEPLOY.md) for one-click deployment instructions.

### Railway

```bash
railway up
```

---

## 👥 Team

**DataQuest 2026 - IIT Kharagpur**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🔮 LiveLens - Where News Meets Real-Time AI</strong><br>
  <em>Built with ❤️ using Pathway</em>
</p>
