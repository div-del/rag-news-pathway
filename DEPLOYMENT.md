# Live Lens - Free Deployment Guide

Deploy your Live Lens RAG News Platform for **$0** using Railway (backend) + Vercel (frontend).

---

## Prerequisites

1. **GitHub Account** - Push your code to a GitHub repository
2. **Railway Account** - [Sign up at railway.app](https://railway.app) (free tier: $5 credit/month)
3. **Vercel Account** - [Sign up at vercel.com](https://vercel.com) (free tier: unlimited static sites)
4. **API Keys** (free tiers available):
   - [Serper API](https://serper.dev/) - for news search
   - [OpenRouter](https://openrouter.ai/) - for LLM (free Mistral model)

---

## Step 1: Deploy Backend to Railway (10 minutes)

### 1.1 Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select your `rag-news-pathway-main` repository
4. Railway will auto-detect the `Dockerfile` and start building

### 1.2 Add PostgreSQL Database

1. In your Railway project, click **"New"** → **"Database"** → **"PostgreSQL"**
2. Railway will provision a PostgreSQL instance automatically
3. The database URL will be available as `DATABASE_URL`

### 1.3 Configure Environment Variables

In your Railway service settings → **Variables** tab, add:

| Variable | Value |
|----------|-------|
| `SERPER_API_KEY` | Your Serper API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `POSTGRES_CONNECTION_STRING` | `${{Postgres.DATABASE_URL}}` (Railway auto-fills this) |
| `PORT` | `8000` |
| `USE_PATHWAY` | `true` |

### 1.4 Deploy

1. Railway will automatically redeploy with your environment variables
2. Wait for the build to complete (3-5 minutes)
3. Click **"Generate Domain"** to get your backend URL
4. **Copy your Railway URL** (e.g., `https://live-lens-production.up.railway.app`)

### 1.5 Test Backend

Visit `https://your-railway-url.railway.app/health` - you should see:
```json
{"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

---

## Step 2: Deploy Frontend to Vercel (5 minutes)

### 2.1 Update Frontend Configuration

Before deploying, update the backend URL in `frontend/app.html`:

```html
<script>
    // Set your Railway backend URL here
    window.LIVE_LENS_BACKEND_URL = 'https://your-railway-url.railway.app';
</script>
```

Commit and push this change to GitHub.

### 2.2 Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **"New Project"** → Import your repository
3. Configure:
   - **Root Directory**: `.` (leave as is)
   - **Framework Preset**: `Other`
   - **Build Command**: Leave empty (static files)
   - **Output Directory**: `frontend`
4. Click **"Deploy"**

### 2.3 Alternative: Use Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from project root)
vercel --prod
```

---

## Step 3: Connect Frontend to Backend

### 3.1 Update Railway CORS

In Railway, add environment variable:

| Variable | Value |
|----------|-------|
| `FRONTEND_URL` | `https://your-vercel-url.vercel.app` |

### 3.2 Verify Connection

1. Visit your Vercel frontend URL
2. The app should load and connect to your Railway backend
3. Check browser console for any CORS errors

---

## Quick Reference: URLs

After deployment, you'll have:

| Component | URL |
|-----------|-----|
| **Frontend** | `https://your-app.vercel.app` |
| **Backend API** | `https://your-app.railway.app/api` |
| **API Docs** | `https://your-app.railway.app/docs` |
| **Health Check** | `https://your-app.railway.app/health` |
| **WebSocket** | `wss://your-app.railway.app/ws/feed` |

---

## Alternative: Single Deployment (Render)

If you prefer everything on one platform:

### Deploy to Render (All-in-One)

1. Go to [render.com](https://render.com) and sign in
2. Create a **Web Service** from your GitHub repo
3. Configure:
   - **Environment**: Docker
   - **Instance Type**: Free
4. Add a **PostgreSQL** database (free for 90 days)
5. Set environment variables in the dashboard

⚠️ **Note**: Render's free tier spins down after 15 min of inactivity (cold starts ~30-60 seconds)

---

## Troubleshooting

### CORS Errors
- Ensure `FRONTEND_URL` is set in Railway
- Check that the URL includes `https://` and has no trailing slash

### WebSocket Connection Failed
- Railway supports WebSocket on the same port
- Verify the frontend is using `wss://` for HTTPS backends

### Database Connection Issues
- Use `${{Postgres.DATABASE_URL}}` syntax in Railway
- Ensure the Postgres service is running

### Build Fails
- Check Railway build logs
- Ensure `requirements.txt` is up to date
- Try adding `--no-cache-dir` to pip install

---

## Free Tier Limits

### Railway
- $5 free credit/month
- Typically enough for 500+ hours of runtime
- No sleep/spin-down on free tier

### Vercel
- Unlimited static site deployments
- 100GB bandwidth/month
- Automatic HTTPS

### API Keys
- **Serper**: 2,500 free searches/month
- **OpenRouter**: Free models available (Mistral)

---

## Demo Day Tips

For the DataQuest 2026 hackathon demo:

1. **Pre-warm the system**: Fetch some news before your demo
2. **Test the dynamism endpoint**: `POST /api/demo/test-dynamism`
3. **Inject a demo article**: `POST /api/demo/inject-article`
4. **Show real-time updates**: Keep the WebSocket feed visible

Your demo should clearly show:
1. Ask a question → Get an answer
2. Inject new data → Show it appearing in real-time
3. Ask the same question → Get an updated answer

This proves the **real-time RAG capability** that judges are looking for!

---

## Support

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- Pathway Docs: https://pathway.com/developers
