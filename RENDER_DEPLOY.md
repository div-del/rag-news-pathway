# Live Lens - Render Deployment Guide (100% Free)

Deploy Live Lens to Render's free tier in under 15 minutes.

---

## Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Render Account** - [Sign up free at render.com](https://render.com) (no credit card required)
3. **API Keys** (free tiers available):
   - [Serper API](https://serper.dev/) - 2,500 free searches/month
   - [OpenRouter](https://openrouter.ai/) - Free Mistral model

---

## Option A: One-Click Deploy (Recommended)

### Step 1: Click Deploy Button

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Or visit: `https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/rag-news-pathway`

### Step 2: Configure Environment Variables

When prompted, enter:
- `SERPER_API_KEY`: Your Serper API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Step 3: Wait for Deployment

Render will automatically:
1. Create a PostgreSQL database (free for 90 days)
2. Build your Docker container
3. Deploy and start the service

**Estimated time: 5-10 minutes**

---

## Option B: Manual Deploy

### Step 1: Sign Up & Connect GitHub

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Authorize Render to access your repositories

### Step 2: Create PostgreSQL Database

1. Click **"New +"** → **"PostgreSQL"**
2. Configure:
   - **Name**: `live-lens-db`
   - **Database**: `live_lens`
   - **User**: `live_lens_user`
   - **Region**: Choose closest to your users
   - **Plan**: **Free**
3. Click **"Create Database"**
4. **Copy the "Internal Database URL"** (you'll need this)

### Step 3: Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `live-lens-api`
   - **Region**: Same as database
   - **Branch**: `main`
   - **Runtime**: **Docker**
   - **Plan**: **Free**

### Step 4: Set Environment Variables

In the Web Service settings, add these environment variables:

| Key | Value |
|-----|-------|
| `SERPER_API_KEY` | Your Serper API key |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |
| `POSTGRES_CONNECTION_STRING` | Internal Database URL from Step 2 |
| `PORT` | `8000` |
| `HOST` | `0.0.0.0` |
| `USE_PATHWAY` | `true` |
| `LLM_MODEL` | `mistralai/devstral-2512:free` |
| `PYTHONUNBUFFERED` | `1` |

### Step 5: Deploy

1. Click **"Create Web Service"**
2. Wait for the build to complete (5-10 minutes)
3. Your app will be live at `https://live-lens-api.onrender.com`

---

## Post-Deployment

### Verify Deployment

1. **Health Check**: Visit `https://YOUR-APP.onrender.com/health`
   ```json
   {"status": "healthy", "timestamp": "...", "version": "1.0.0"}
   ```

2. **API Docs**: Visit `https://YOUR-APP.onrender.com/docs`

3. **Frontend**: Visit `https://YOUR-APP.onrender.com/`

### Test Real-Time Dynamism

```bash
# Test the dynamism endpoint
curl -X POST https://YOUR-APP.onrender.com/api/demo/test-dynamism

# Inject a demo article
curl -X POST https://YOUR-APP.onrender.com/api/demo/inject-article \
  -H "Content-Type: application/json" \
  -d '{"title": "Breaking News", "content": "Test content", "category": "Demo"}'
```

---

## Free Tier Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Web Service | 750 hrs/month | Spins down after 15 min inactivity |
| PostgreSQL | Free for 90 days | Then $7/month or migrate data |
| Bandwidth | 100 GB/month | More than enough for hackathon |
| Build Minutes | 500/month | Plenty for updates |

### Cold Starts

Free tier services spin down after 15 minutes of inactivity. First request after spin-down takes **30-60 seconds** to respond.

**Solutions for demos:**
1. Visit your app 2-3 minutes before presenting
2. Use [UptimeRobot](https://uptimerobot.com) to ping every 14 minutes (free)
3. Use [cron-job.org](https://cron-job.org) to hit `/health` endpoint

---

## Troubleshooting

### Build Fails

**Check Dockerfile:**
```bash
# Test locally first
docker build -t live-lens .
docker run -p 8000:8000 live-lens
```

**Common fixes:**
- Ensure `requirements.txt` is up to date
- Check for missing system dependencies in Dockerfile

### Database Connection Error

1. Verify `POSTGRES_CONNECTION_STRING` is set
2. Use the **Internal Database URL** (not External)
3. Check database is in same region as web service

### WebSocket Not Connecting

- Render supports WebSocket on the same port
- Ensure frontend uses `wss://` for HTTPS
- Check browser console for errors

### App Crashes on Start

1. Check Render logs: Dashboard → Service → Logs
2. Common issues:
   - Missing environment variables
   - Database not ready yet
   - Port mismatch (must use `PORT` env var)

---

## Updating Your App

After pushing changes to GitHub:

1. Render auto-deploys from `main` branch
2. Or manually trigger: Dashboard → Service → **"Manual Deploy"**

---

## Custom Domain (Optional)

1. Go to Service → **Settings** → **Custom Domains**
2. Add your domain (e.g., `livelens.yourdomain.com`)
3. Add CNAME record in your DNS:
   ```
   CNAME livelens -> YOUR-APP.onrender.com
   ```

---

## Demo Day Checklist

- [ ] Deploy completed successfully
- [ ] Health endpoint returns `healthy`
- [ ] Frontend loads correctly
- [ ] WebSocket connection shows "Connected"
- [ ] Can fetch news articles
- [ ] Dynamism test works (`/api/demo/test-dynamism`)
- [ ] App is "warm" (visited recently to avoid cold start)

---

## Cost Summary

| Component | Cost |
|-----------|------|
| Render Web Service | **FREE** |
| Render PostgreSQL | **FREE** (90 days) |
| Serper API | **FREE** (2,500/month) |
| OpenRouter LLM | **FREE** (Mistral model) |
| **Total** | **$0** |

---

## Support

- Render Docs: https://render.com/docs
- Render Status: https://status.render.com
- Community: https://community.render.com
