/**
 * Live AI News Platform - Frontend JavaScript
 * Premium Interactive Experience
 */

const API_BASE = '/api';
let wsConnection = null;
let currentArticleId = null;
let articles = [];
let comparisonData = { 1: [], 2: [] };

// ============ Live Data Config ============
const BACKGROUND_REFRESH_INTERVAL = 2 * 60 * 1000; // 2 minutes
let lastArticleCount = 0;
let backgroundRefreshTimer = null;

// ============ Initialization ============

document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeEventListeners();

    // Initial load of existing articles
    loadNewsFeed();
    loadStats();
    connectWebSocket();

    // [LIVE DATA] Auto-fetch fresh news on page load
    autoFetchOnLoad();

    // [LIVE DATA] Start background refresh timer
    startBackgroundRefresh();
});

function initializeNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const section = e.currentTarget.dataset.section;
            switchSection(section);
        });
    });
}

function switchSection(section) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === section);
    });

    // Update sections
    document.querySelectorAll('.section').forEach(sec => {
        sec.classList.toggle('active', sec.id === `${section}-section`);
    });
}

function initializeEventListeners() {
    // Fetch news button
    document.getElementById('fetch-news-btn').addEventListener('click', fetchNews);

    // Search
    document.getElementById('search-input').addEventListener('input', debounce(filterArticles, 300));
    document.getElementById('category-filter').addEventListener('change', () => {
        const category = document.getElementById('category-filter').value;
        loadNewsFeed(category);
    });

    // Chat
    document.getElementById('send-chat-btn').addEventListener('click', sendChatMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });

    // Article modal chat
    document.getElementById('send-article-chat-btn').addEventListener('click', sendArticleChatMessage);
    document.getElementById('article-chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendArticleChatMessage();
    });

    // Comparison
    document.getElementById('compare-btn').addEventListener('click', compareArticles);
    document.getElementById('compare-select-1').addEventListener('change', updateComparePreview);
    document.getElementById('compare-select-1').addEventListener('change', updateComparePreview);
    document.getElementById('compare-select-2').addEventListener('change', updateComparePreview);

    // Auto-load comparison articles when category changes
    document.getElementById('compare-cat-1').addEventListener('change', () => loadComparisonArticles(1));
    document.getElementById('compare-cat-2').addEventListener('change', () => loadComparisonArticles(2));
    document.getElementById('compare-query').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') compareArticles();
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// ============ Notifications ============

/**
 * Show a toast notification
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style based on type
    const bgColor = type === 'success' ? 'linear-gradient(135deg, #10b981, #059669)'
        : type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)'
            : 'linear-gradient(135deg, #6366f1, #8b5cf6)';

    notification.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${bgColor};
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        max-width: 300px;
    `;

    document.body.appendChild(notification);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100px)';
        notification.style.transition = 'all 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// ============ Live Data Functions ============

/**
 * Auto-fetch fresh news when page loads
 */
async function autoFetchOnLoad() {
    console.log('[LIVE] Auto-fetching fresh news on page load...');
    try {
        // Fetch a small batch of fresh articles silently
        const result = await fetchAPI('/news/fetch?num_results=5', { method: 'POST' });

        if (result.count > 0) {
            showNotification(`🔴 LIVE: ${result.count} fresh articles loaded`, 'success');
            await loadNewsFeed();
            await loadStats();
            lastArticleCount = articles.length;
        }
    } catch (error) {
        console.error('[LIVE] Auto-fetch failed:', error);
        // Silent fail - don't disturb user
    }
}

/**
 * Start background refresh timer
 */
function startBackgroundRefresh() {
    // Clear any existing timer
    if (backgroundRefreshTimer) {
        clearInterval(backgroundRefreshTimer);
    }

    console.log(`[LIVE] Background refresh enabled (every ${BACKGROUND_REFRESH_INTERVAL / 1000}s)`);

    backgroundRefreshTimer = setInterval(async () => {
        await checkForNewArticles();
    }, BACKGROUND_REFRESH_INTERVAL);
}

/**
 * Check for new articles in background and notify user
 */
async function checkForNewArticles() {
    console.log('[LIVE] Checking for new articles...');

    try {
        // Silently fetch new articles
        const result = await fetchAPI('/news/fetch?num_results=3', { method: 'POST' });

        if (result.count > 0) {
            // New articles available!
            showNewArticlesNotification(result.count);
        }

        // Refresh the feed silently
        await loadNewsFeed(document.getElementById('category-filter').value || null);
        await loadStats();

        // Check if count increased
        const newCount = articles.length;
        if (newCount > lastArticleCount && lastArticleCount > 0) {
            const diff = newCount - lastArticleCount;
            showNotification(`🔴 LIVE: +${diff} new articles added to feed`, 'success');
        }
        lastArticleCount = newCount;

    } catch (error) {
        console.error('[LIVE] Background check failed:', error);
    }
}

/**
 * Show prominent "New Articles" notification
 */
function showNewArticlesNotification(count) {
    // Create a special live notification
    const notification = document.createElement('div');
    notification.className = 'notification live-notification';
    notification.innerHTML = `
        <span class="live-pulse"></span>
        <span>🔴 LIVE: ${count} new articles just arrived!</span>
    `;
    notification.style.cssText = `
        position: fixed;
        top: 80px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: 600;
        z-index: 10000;
        animation: slideDown 0.3s ease-out, pulse 2s infinite;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    `;

    notification.onclick = () => {
        notification.remove();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideUp 0.3s ease-out forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// ============ API Functions ============

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ============ Stats ============

async function loadStats() {
    try {
        const stats = await fetchAPI('/stats');
        document.getElementById('indexed-count').textContent = `${stats.rag_engine.indexed_documents} articles indexed`;
        document.getElementById('article-count').textContent = `${stats.rag_engine.indexed_documents} articles`;
    } catch (error) {
        console.error('Failed to load stats');
    }
}

// ============ News Feed ============

async function loadNewsFeed(category = null) {
    const feedContainer = document.getElementById('news-feed');
    feedContainer.innerHTML = '<div class="loading">Loading articles</div>';

    // Construct URL with optional category
    // Default limit 20 for mixed feed, 10 for specific category
    let limit = 20;
    if (category) {
        limit = 10;
    }

    let url = `/news/feed?limit=${limit}`;
    if (category) {
        url += `&category=${encodeURIComponent(category)}`;
    }

    try {
        const data = await fetchAPI(url);
        articles = data.articles || [];

        renderArticles(articles);
        updateCompareDropdowns();
        updateStats();

    } catch (error) {
        feedContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">⚠️</div>
                <h3>Unable to load articles</h3>
                <p>Make sure the backend is running and try again</p>
            </div>
        `;
    }
}

function renderArticles(articlesToRender) {
    const feedContainer = document.getElementById('news-feed');

    if (articlesToRender.length === 0) {
        feedContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📰</div>
                <h3>No articles yet</h3>
                <p>Click "Fetch News" to load the latest articles</p>
            </div>
        `;
        return;
    }

    feedContainer.innerHTML = articlesToRender.map(article => createNewsCard(article)).join('');
}

function createNewsCard(article) {
    const topics = article.topics || [];
    const topicTags = topics.slice(0, 4).map(t => `<span class="topic-tag">${escapeHtml(t)}</span>`).join('');
    const score = article.relevance_score || article.score || 0;

    return `
        <article class="news-card" onclick="openArticle('${article.article_id}')">
            <div class="news-card-content">
                <div class="news-card-meta">
                    <span class="news-category">${escapeHtml(article.category || 'News')}</span>
                    <span class="news-source">${escapeHtml(article.source || '')}</span>
                </div>
                <h3>${escapeHtml(article.title || 'Untitled')}</h3>
                ${topicTags ? `<div class="news-topics">${topicTags}</div>` : ''}
                ${score > 0.5 ? `
                    <div class="relevance-score">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                        <span>${Math.round(score * 100)}% relevant for you</span>
                    </div>
                ` : ''}
            </div>
        </article>
    `;
}

function filterArticles() {
    const searchTerm = document.getElementById('search-input').value.toLowerCase();
    const category = document.getElementById('category-filter').value;

    // Allow local search filtering on top of server-side category
    // But reload feed if category changes
    if (articles.length > 0 && articles[0].category !== category && category !== "") {
        loadNewsFeed(category);
        return;
    } else if (category === "" && articles.length > 0 && articles.some(a => a.category !== articles[0].category)) {
        // If "All" selected but we currently have single-category view, reload all
        // This check is imperfect but simple: just reload if switching to "All" to be safe
        loadNewsFeed();
        return;
    }

    // Client-side filtering for search term
    let filtered = articles;

    if (searchTerm) {
        filtered = filtered.filter(a =>
            (a.title || '').toLowerCase().includes(searchTerm) ||
            (a.topics || []).some(t => t.toLowerCase().includes(searchTerm))
        );
    }

    // Also filter by category locally just in case we have mixed set
    if (category) {
        filtered = filtered.filter(a => a.category === category);
    }

    renderArticles(filtered);
}

// Ensure filterArticles handles the category change event correctly
// We need to modify the event listener logic slightly if we want it to trigger a reload
// The existing event listener calls filterArticles, so we'll handle the reload logic there
// or better, update the event listener to call loadNewsFeed directly for category changes.

function updateStats() {
    document.getElementById('article-count').textContent = `${articles.length} articles`;
    document.getElementById('indexed-count').textContent = `${articles.length} articles indexed`;
    document.getElementById('last-update').textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

async function fetchNews() {
    const btn = document.getElementById('fetch-news-btn');
    const originalContent = btn.innerHTML;

    // Get currently selected category
    const categoryFilter = document.getElementById('category-filter');
    const selectedCategory = categoryFilter.value; // Empty string if "All", or specific category

    const displayCategory = selectedCategory || 'All Categories';

    btn.innerHTML = `
        <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        <span>Fetching ${escapeHtml(displayCategory)}...</span>
    `;
    btn.disabled = true;

    try {
        // Build URL
        let url = '/news/fetch?num_results=5';
        if (selectedCategory) {
            url += `&category=${encodeURIComponent(selectedCategory)}`;
        }

        const result = await fetchAPI(url, { method: 'POST' });

        if (result.count > 0) {
            showNotification(`✅ Fetched ${result.count} new articles`);
        } else {
            showNotification('No new articles found');
        }

        await loadNewsFeed(selectedCategory); // Reload feed with current filter
        await loadStats();

    } catch (error) {
        showNotification('❌ Error fetching news', 'error');
    }

    btn.innerHTML = originalContent;
    btn.disabled = false;
}

// ============ Article Modal ============

async function openArticle(articleId) {
    currentArticleId = articleId;
    const modal = document.getElementById('article-modal');
    const content = document.getElementById('modal-article-content');

    modal.classList.add('active');
    content.innerHTML = '<div class="loading">Loading article</div>';

    try {
        const data = await fetchAPI(`/news/article/${articleId}`);
        const article = data.article;

        content.innerHTML = `
            <span class="news-category">${escapeHtml(article.category || 'News')}</span>
            <h2>${escapeHtml(article.title)}</h2>
            <div style="color: var(--text-secondary); margin-bottom: 1.5rem; font-size: 0.9rem;">
                ${escapeHtml(article.source || 'Unknown source')} • ${article.publish_date ? new Date(article.publish_date).toLocaleDateString() : 'Recent'}
            </div>
            <div class="article-content">${escapeHtml(article.content || 'No content available')}</div>
        `;

        // Show comparison suggestion
        const suggestionDiv = document.getElementById('comparison-suggestion');
        if (data.comparison_suggestion) {
            const s = data.comparison_suggestion;
            suggestionDiv.innerHTML = `
                <p><strong>💡 Want to compare?</strong></p>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">Similar article: "${escapeHtml(s.titles[1])}"</p>
                <button class="btn-primary" style="margin-top: 0.5rem" onclick="startComparison('${s.article_ids[0]}', '${s.article_ids[1]}')">
                    Compare Articles
                </button>
            `;
            suggestionDiv.style.display = 'block';
        } else {
            suggestionDiv.style.display = 'none';
        }

        document.getElementById('article-chat-messages').innerHTML = '';

    } catch (error) {
        content.innerHTML = '<div class="empty-state"><p>Error loading article</p></div>';
    }
}

function closeModal() {
    document.getElementById('article-modal').classList.remove('active');
    currentArticleId = null;
}

// Close modal on escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// ============ Chat Functions ============

function setQuery(query) {
    document.getElementById('chat-input').value = query;
    document.getElementById('chat-input').focus();
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();

    if (!query) return;

    input.value = '';
    const messagesDiv = document.getElementById('chat-messages');

    // Add user message
    messagesDiv.innerHTML += `
        <div class="message user">
            <div class="message-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
            <div class="message-content"><p>${escapeHtml(query)}</p></div>
        </div>
    `;
    scrollToBottom(messagesDiv);

    // Add loading message
    const loadingId = `loading-${Date.now()}`;
    messagesDiv.innerHTML += `
        <div class="message assistant" id="${loadingId}">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <circle cx="12" cy="12" r="10"/>
                </svg>
            </div>
            <div class="message-content"><div class="loading">Thinking</div></div>
        </div>
    `;
    scrollToBottom(messagesDiv);

    try {
        const result = await fetchAPI('/chat/query', {
            method: 'POST',
            body: JSON.stringify({ query })
        });

        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            `<p>${formatResponse(result.response)}</p>`;

    } catch (error) {
        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            '<p>Sorry, there was an error processing your question. Please try again.</p>';
    }

    scrollToBottom(messagesDiv);
}

async function sendArticleChatMessage() {
    if (!currentArticleId) return;

    const input = document.getElementById('article-chat-input');
    const query = input.value.trim();

    if (!query) return;

    input.value = '';
    const messagesDiv = document.getElementById('article-chat-messages');

    messagesDiv.innerHTML += `<div class="message user"><div class="message-content"><p>${escapeHtml(query)}</p></div></div>`;

    const loadingId = `loading-${Date.now()}`;
    messagesDiv.innerHTML += `<div class="message assistant" id="${loadingId}"><div class="message-content"><div class="loading">Thinking</div></div></div>`;

    try {
        const result = await fetchAPI(`/chat/article/${currentArticleId}`, {
            method: 'POST',
            body: JSON.stringify({ query, expand_context: true })
        });

        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            `<p>${formatResponse(result.response)}</p>`;

    } catch (error) {
        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            '<p>Sorry, there was an error.</p>';
    }

    scrollToBottom(messagesDiv);
}

// ============ Comparison Functions ============

// Update all compare dropdowns
async function updateCompareDropdowns() {
    // Initial load for both slots based on their default/current categories
    await loadComparisonArticles(1);
    await loadComparisonArticles(2);
}

// Fetch articles for a specific comparison slot
async function loadComparisonArticles(slotId) {
    const catSelect = document.getElementById(`compare-cat-${slotId}`);
    if (!catSelect) return;

    const category = catSelect.value;
    const articleSelect = document.getElementById(`compare-select-${slotId}`);
    const originalText = articleSelect.options[0] ? articleSelect.options[0].text : 'Select an article...';

    // Show loading state in dropdown
    articleSelect.innerHTML = '<option>Loading...</option>';
    articleSelect.disabled = true;

    // Use limit 20 to get a good selection
    let url = `/news/feed?limit=20`;
    if (category) {
        url += `&category=${encodeURIComponent(category)}`;
    }

    try {
        const data = await fetchAPI(url);
        comparisonData[slotId] = data.articles || [];
        renderCompareDropdown(slotId);
    } catch (error) {
        console.error(`Error loading comparison articles for slot ${slotId}:`, error);
        articleSelect.innerHTML = '<option>Error loading articles</option>';
    } finally {
        articleSelect.disabled = false;
    }
}

// Render specific compare dropdown from local comparisonData
function renderCompareDropdown(slotId) {
    const catSelect = document.getElementById(`compare-cat-${slotId}`);
    const articleSelect = document.getElementById(`compare-select-${slotId}`);

    if (!catSelect || !articleSelect) return;

    const category = catSelect.value;
    const currentVal = articleSelect.value;

    // Filter articles from our specific slot data
    // (They should already be filtered by the API, but double check)
    let slotArticles = comparisonData[slotId] || [];

    if (category) {
        slotArticles = slotArticles.filter(a => a.category === category);
    }

    if (slotArticles.length === 0) {
        articleSelect.innerHTML = '<option value="">No articles found for this category</option>';
        return;
    }

    const options = slotArticles.map(a =>
        `<option value="${a.article_id}">${escapeHtml(truncate(a.title, 60))}</option>`
    ).join('');

    articleSelect.innerHTML = `<option value="">Select an article...</option>${options}`;

    // Restore selection if valid
    if (currentVal && slotArticles.some(a => a.article_id === currentVal)) {
        articleSelect.value = currentVal;
    }
}



async function fetchCompareNews(slotId) {
    const btn = document.getElementById(`compare-fetch-${slotId}`);
    const catSelect = document.getElementById(`compare-cat-${slotId}`);
    const category = catSelect.value;
    const displayCategory = category || 'All Categories';

    const originalContent = btn.innerHTML;
    btn.innerHTML = '...';
    btn.disabled = true;

    try {
        let url = '/news/fetch?num_results=5';
        if (category) {
            url += `&category=${encodeURIComponent(category)}`;
        }

        const result = await fetchAPI(url, { method: 'POST' });

        if (result.count > 0) {
            showNotification(`Fetched ${result.count} new articles`);
        } else {
            showNotification('No new articles options found');
        }

        // Reload just this slot
        await loadComparisonArticles(slotId);

        // The loadNewsFeed call is no longer needed here as we handle slots independently

    } catch (error) {
        showNotification('Error fetching news', 'error');
    }

    btn.innerHTML = originalContent;
    btn.disabled = false;
}

function updateComparePreview() {
    const id1 = document.getElementById('compare-select-1').value;
    const id2 = document.getElementById('compare-select-2').value;

    const preview1 = document.getElementById('compare-preview-1');
    const preview2 = document.getElementById('compare-preview-2');

    const article1 = comparisonData[1].find(a => a.article_id === id1) || articles.find(a => a.article_id === id1);
    const article2 = comparisonData[2].find(a => a.article_id === id2) || articles.find(a => a.article_id === id2);

    preview1.innerHTML = article1 ? `<strong>${escapeHtml(article1.category || 'News')}</strong><br>${escapeHtml(truncate(article1.title, 80))}` : 'No article selected';
    preview2.innerHTML = article2 ? `<strong>${escapeHtml(article2.category || 'News')}</strong><br>${escapeHtml(truncate(article2.title, 80))}` : 'No article selected';
}

function startComparison(id1, id2) {
    closeModal();
    switchSection('compare');

    document.getElementById('compare-select-1').value = id1;
    document.getElementById('compare-select-2').value = id2;
    updateComparePreview();
}

async function compareArticles() {
    const id1 = document.getElementById('compare-select-1').value;
    const id2 = document.getElementById('compare-select-2').value;
    const query = document.getElementById('compare-query').value.trim() || 'Compare these two articles in detail';

    if (!id1 || !id2) {
        showNotification('Please select two articles to compare', 'warning');
        return;
    }

    const resultDiv = document.getElementById('compare-result');
    resultDiv.innerHTML = '<div class="loading">Analyzing articles</div>';

    try {
        const result = await fetchAPI('/chat/compare', {
            method: 'POST',
            body: JSON.stringify({
                article_ids: [id1, id2],
                query
            })
        });

        resultDiv.innerHTML = `<p>${formatResponse(result.response)}</p>`;

    } catch (error) {
        resultDiv.innerHTML = '<p style="color: var(--accent-danger)">Error comparing articles. Please try again.</p>';
    }
}

// ============ WebSocket ============

function connectWebSocket() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('connection-status');

    try {
        wsConnection = new WebSocket(`ws://${window.location.host}/ws/feed`);

        wsConnection.onopen = () => {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        };

        wsConnection.onmessage = (event) => {
            if (event.data !== 'pong') {
                const data = JSON.parse(event.data);
                if (data.type === 'new_article') {
                    loadNewsFeed();
                }
            }
        };

        wsConnection.onclose = () => {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
            setTimeout(connectWebSocket, 5000);
        };

        wsConnection.onerror = () => {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Connection error';
        };

    } catch (error) {
        statusText.textContent = 'WebSocket unavailable';
    }
}

// Keep WebSocket alive
setInterval(() => {
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send('ping');
    }
}, 30000);

// ============ YouTube Analysis Functions ============

let currentYouTubeVideos = [];
let currentArticleTitle = '';
let currentArticleContent = '';

async function fetchYouTubeVideos() {
    const btn = document.getElementById('fetch-youtube-btn');
    const listDiv = document.getElementById('youtube-videos-list');
    const loadingDiv = document.getElementById('youtube-analysis-loading');
    const resultDiv = document.getElementById('youtube-analysis-result');

    // Get article title from modal
    const modalContent = document.getElementById('modal-article-content');
    const titleElement = modalContent.querySelector('h2');
    currentArticleTitle = titleElement ? titleElement.textContent : '';
    const contentElement = modalContent.querySelector('.article-content');
    currentArticleContent = contentElement ? contentElement.textContent.substring(0, 500) : '';

    if (!currentArticleTitle) {
        showNotification('No article loaded', 'error');
        return;
    }

    // Update button state
    btn.disabled = true;
    btn.innerHTML = `
        <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        <span>Searching...</span>
    `;

    // Hide previous results
    resultDiv.style.display = 'none';
    loadingDiv.style.display = 'none';

    try {
        const data = await fetchAPI('/youtube/search', {
            method: 'POST',
            body: JSON.stringify({
                query: currentArticleTitle,
                num_results: 5
            })
        });

        currentYouTubeVideos = data.videos || [];

        if (currentYouTubeVideos.length === 0) {
            listDiv.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 1rem;">No related YouTube videos found</p>';
        } else {
            listDiv.innerHTML = currentYouTubeVideos.map((video, index) => `
                <div class="youtube-video-card" onclick="selectVideo(${index})">
                    <div class="video-thumbnail">
                        ${video.thumbnail
                    ? `<img src="${escapeHtml(video.thumbnail)}" alt="Video thumbnail" onerror="this.parentElement.innerHTML='<svg width=\\'24\\' height=\\'24\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'2\\'><polygon points=\\'23 7 16 12 23 17 23 7\\'/><rect x=\\'1\\' y=\\'5\\' width=\\'15\\' height=\\'14\\' rx=\\'2\\' ry=\\'2\\'/></svg>'"/>`
                    : `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>`
                }
                    </div>
                    <div class="video-info">
                        <h4>${escapeHtml(video.title)}</h4>
                        <div class="video-channel">${escapeHtml(video.channel)}</div>
                        ${video.duration ? `<div class="video-duration">${escapeHtml(video.duration)}</div>` : ''}
                    </div>
                    <button class="video-analyze-btn" onclick="event.stopPropagation(); analyzeYouTubeVideo('${escapeHtml(video.url)}', '${escapeHtml(video.title)}')">
                        Analyze
                    </button>
                </div>
            `).join('');
        }

        listDiv.style.display = 'flex';

    } catch (error) {
        console.error('YouTube search error:', error);
        listDiv.innerHTML = '<p style="color: var(--accent-danger); text-align: center; padding: 1rem;">Error searching YouTube. Please try again.</p>';
        listDiv.style.display = 'flex';
    }

    // Reset button
    btn.disabled = false;
    btn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7"/>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
        </svg>
        <span>Find Related Videos</span>
    `;
}

function selectVideo(index) {
    // Highlight selected video
    document.querySelectorAll('.youtube-video-card').forEach((card, i) => {
        card.classList.toggle('selected', i === index);
    });
}

async function analyzeYouTubeVideo(videoUrl, videoTitle) {
    const listDiv = document.getElementById('youtube-videos-list');
    const loadingDiv = document.getElementById('youtube-analysis-loading');
    const resultDiv = document.getElementById('youtube-analysis-result');

    // Show loading
    listDiv.style.display = 'none';
    loadingDiv.style.display = 'block';
    resultDiv.style.display = 'none';

    try {
        const data = await fetchAPI('/youtube/analyze', {
            method: 'POST',
            body: JSON.stringify({
                video_url: videoUrl,
                article_title: currentArticleTitle,
                article_content: currentArticleContent
            })
        });

        // Display results
        displayYouTubeAnalysis(data);

    } catch (error) {
        console.error('YouTube analysis error:', error);
        loadingDiv.style.display = 'none';
        listDiv.style.display = 'flex';
        showNotification('Error analyzing video. Make sure yt-dlp and ffmpeg are installed.', 'error');
    }
}

function displayYouTubeAnalysis(data) {
    const loadingDiv = document.getElementById('youtube-analysis-loading');
    const resultDiv = document.getElementById('youtube-analysis-result');

    loadingDiv.style.display = 'none';
    resultDiv.style.display = 'block';

    // Set sentiment badge
    const sentimentBadge = document.getElementById('video-sentiment');
    sentimentBadge.textContent = data.sentiment || 'neutral';
    sentimentBadge.className = `sentiment-badge ${data.sentiment || 'neutral'}`;

    // Set key points
    const keyPointsDiv = document.getElementById('video-key-points');
    if (data.key_points && data.key_points.length > 0) {
        keyPointsDiv.innerHTML = `
            <ul>
                ${data.key_points.map(point => `<li>${escapeHtml(point)}</li>`).join('')}
            </ul>
        `;
    } else {
        keyPointsDiv.innerHTML = '';
    }

    // Set analysis text
    const analysisText = document.getElementById('video-analysis-text');
    analysisText.textContent = data.analysis || 'No analysis available.';

    // Set transcript
    const transcriptDiv = document.getElementById('video-transcript');
    transcriptDiv.textContent = data.transcript || 'No transcript available.';

    showNotification('✅ Video analysis complete!');
}

// ============ Utilities ============

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, length) {
    if (!text) return '';
    return text.length > length ? text.substring(0, length) + '...' : text;
}

function formatResponse(text) {
    if (!text) return '';
    // Convert line breaks to <br> and escape HTML
    return escapeHtml(text).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
}

function scrollToBottom(element) {
    element.scrollTop = element.scrollHeight;
}

function showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'error' ? 'var(--accent-danger)' : type === 'warning' ? 'var(--accent-warning)' : 'var(--accent-success)'};
        color: white;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 500;
        z-index: 1001;
        animation: slideIn 0.3s ease;
        box-shadow: var(--shadow-lg);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS for spin animation
const style = document.createElement('style');
style.textContent = `
    .spin { animation: spin 1s linear infinite; }
    @keyframes fadeOut { to { opacity: 0; transform: translateX(20px); } }
`;
document.head.appendChild(style);
