# Setup Notes

## Python Version Requirement

**Important:** Pathway requires Python 3.10 or newer. This project uses **Docker with Python 3.11** to ensure compatibility.

## Quick Start

### Start with Docker

```bash
# Start PostgreSQL and the application
docker-compose up -d

# View logs
docker-compose logs -f app

# Access the application
open http://localhost:8000
```

### Development Mode (with hot reload)

```bash
# Start only postgres
docker-compose up -d postgres

# Start the app in dev mode
docker-compose --profile dev up app-dev
```

## Current Status

- ✅ All dependencies configured (requirements.txt)
- ✅ Docker environment ready (Dockerfile, docker-compose.yml)
- ✅ Database models and utilities created
- ✅ News connectors implemented (Serper API + news-please)
- ✅ Pathway document pipeline created
- ✅ RAG engine with multi-context support
- ✅ User personalization system
- ✅ Recommendation engine
- ✅ FastAPI REST & WebSocket endpoints
- ✅ Simple frontend (HTML/CSS/JS)
- ✅ Main application entry point
- ✅ README with documentation

## To Test

1. Start Docker Desktop
2. Run `docker-compose up -d`
3. Open http://localhost:8000
4. Click "Fetch News" to load articles
5. Try the AI chat and comparison features

## API Endpoints

See README.md for full API documentation.
