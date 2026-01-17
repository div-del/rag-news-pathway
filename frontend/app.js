/**
 * Live AI News Platform - Frontend JavaScript
 * Premium Interactive Experience
 */

// Use configurable API base (supports separate frontend/backend deployment)
const API_BASE = window.LiveLensConfig?.API_BASE
    ? `${window.LiveLensConfig.API_BASE}/api`
    : '/api';
const WS_BASE = window.LiveLensConfig?.WS_BASE || '';

let wsConnection = null;
let currentArticleId = null;
let articles = [];

// AI Chat state
let aiChatSessionId = null;

// ============ Initialization ============

document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    initializeEventListeners();
    initializeTheme();
    initializeAIChat();
    initializePushNotifications();
    initializeNewsletter();
    loadNewsFeed();
    loadStats();
    connectWebSocket();

    // URL Routing: Open article if specified in query params
    const urlParams = new URLSearchParams(window.location.search);
    const articleId = urlParams.get('article');
    if (articleId) {
        // Wait slightly for feed to initialize, but loadArticle handles its own fetch
        setTimeout(() => openArticle(articleId), 500);
    }
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

    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

    // Search
    document.getElementById('search-input').addEventListener('input', debounce(filterArticles, 300));
    document.getElementById('category-filter').addEventListener('change', () => {
        const category = document.getElementById('category-filter').value;
        loadNewsFeed(category);
    });

    // AI Chat
    document.getElementById('send-ai-chat-btn').addEventListener('click', sendAIChatMessage);
    document.getElementById('ai-chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendAIChatMessage();
    });
    document.getElementById('new-chat-btn').addEventListener('click', createNewAIChat);

    // Article modal chat
    document.getElementById('send-article-chat-btn').addEventListener('click', sendArticleChatMessage);
    document.getElementById('article-chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendArticleChatMessage();
    });

    // Comparison
    document.getElementById('compare-btn').addEventListener('click', compareArticles);
    document.getElementById('compare-select-1').addEventListener('change', updateComparePreview);
    document.getElementById('compare-select-2').addEventListener('change', updateComparePreview);
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

// ============ API Functions ============

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                // Required for ngrok free tier to bypass browser warning page
                'ngrok-skip-browser-warning': 'true',
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
    const articleCount = document.getElementById('article-count');
    if (articleCount) {
        articleCount.textContent = `${articles.length} articles`;
    }

    // Also update the KB count if available but don't overwrite with just feed length if KB status is better
    // This function runs on feed load, so it only knows about loaded articles

    const indexedCount = document.getElementById('indexed-count');
    if (indexedCount) {
        indexedCount.textContent = `${articles.length} articles indexed`;
    }

    const lastUpdate = document.getElementById('last-update');
    if (lastUpdate) {
        lastUpdate.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    }
}

function prependNewArticle(article) {
    // Add to local articles array
    const existingIndex = articles.findIndex(a => a.article_id === article.article_id);
    if (existingIndex === -1) {
        articles.unshift(article);
    } else {
        return; // Already exists
    }

    // Prepend to feed visually
    const feedContainer = document.getElementById('news-feed');

    // Remove empty state if present
    const emptyState = feedContainer.querySelector('.empty-state');
    if (emptyState) {
        feedContainer.innerHTML = '';
    }

    // Create and prepend the card
    const cardHTML = createNewsCard(article);
    feedContainer.insertAdjacentHTML('afterbegin', cardHTML);

    // Add highlight animation
    const newCard = feedContainer.firstElementChild;
    if (newCard) {
        newCard.classList.add('new-article');
        setTimeout(() => newCard.classList.remove('new-article'), 3000);
    }

    // Update stats and dropdowns
    updateStats();
    updateCompareDropdowns();

    // Show notification
    showNotification(`📰 New: ${truncate(article.title, 40)}`);
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

// ============ AI Chat Functions ============

function setAIChatQuery(query) {
    document.getElementById('ai-chat-input').value = query;
    document.getElementById('ai-chat-input').focus();
}

async function initializeAIChat() {
    // Get or create session ID from localStorage
    aiChatSessionId = localStorage.getItem('aiChatSessionId');

    if (!aiChatSessionId) {
        await createNewAIChat();
    }
}

async function createNewAIChat() {
    try {
        const result = await fetchAPI('/ai-chat/sessions/new', { method: 'POST' });
        aiChatSessionId = result.session_id;
        localStorage.setItem('aiChatSessionId', aiChatSessionId);

        // Clear the chat messages UI
        const messagesDiv = document.getElementById('ai-chat-messages');
        messagesDiv.innerHTML = `
            <div class="message assistant">
                <div class="message-avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
                        <circle cx="12" cy="12" r="3" />
                    </svg>
                </div>
                <div class="message-content">
                    <div class="markdown-content">
                        <p>👋 <strong>Hello! I'm your AI News Assistant.</strong></p>
                        <p>I can help you:</p>
                        <ul>
                            <li>Answer questions about current news topics</li>
                            <li>Summarize articles on specific subjects</li>
                            <li>Find connections between different news stories</li>
                            <li>Explain complex news developments</li>
                        </ul>
                        <p>I'll automatically search our article database - and if I don't have enough information, I'll fetch new articles for you!</p>
                        <p><em>Try asking about technology, finance, business, or any current topic.</em></p>
                    </div>
                </div>
            </div>
        `;

        // Hide sources panel
        document.getElementById('ai-chat-sources').style.display = 'none';

        showNotification('✨ Started new conversation');
    } catch (error) {
        console.error('Error creating new chat:', error);
        showNotification('Error creating new chat', 'error');
    }
}

async function sendAIChatMessage() {
    const input = document.getElementById('ai-chat-input');
    const query = input.value.trim();

    if (!query) return;

    // Ensure we have a session
    if (!aiChatSessionId) {
        await initializeAIChat();
    }

    input.value = '';
    const messagesDiv = document.getElementById('ai-chat-messages');

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
            <div class="message-content">
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="loading-text">Searching articles and generating response...</span>
            </div>
        </div>
    `;
    scrollToBottom(messagesDiv);

    try {
        const result = await fetchAPI('/ai-chat/message', {
            method: 'POST',
            body: JSON.stringify({
                session_id: aiChatSessionId,
                message: query
            })
        });

        // Render the response with markdown
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            let responseHtml = result.response || '';

            // Use marked.js if available for markdown rendering
            if (typeof marked !== 'undefined') {
                responseHtml = marked.parse(responseHtml);
            } else {
                // Fallback formatting
                responseHtml = formatResponse(responseHtml);
            }

            loadingMsg.querySelector('.message-content').innerHTML = `
                <div class="markdown-content">${responseHtml}</div>
                ${result.articles_fetched > 0 ? `
                    <div class="fetch-notice">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"/>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                        </svg>
                        <span>Fetched ${result.articles_fetched} new articles for this query</span>
                    </div>
                ` : ''}
            `;
        }

        // Update sources panel
        if (result.sources && result.sources.length > 0) {
            displayAIChatSources(result.sources);
        }

    } catch (error) {
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            loadingMsg.querySelector('.message-content').innerHTML = `
                <p class="error-message">Sorry, I encountered an error processing your question. Please try again.</p>
            `;
        }
        console.error('AI Chat error:', error);
    }

    scrollToBottom(messagesDiv);
}

function displayAIChatSources(sources) {
    const sourcesPanel = document.getElementById('ai-chat-sources');
    const sourcesList = document.getElementById('sources-list');

    if (sources.length === 0) {
        sourcesPanel.style.display = 'none';
        return;
    }

    sourcesPanel.style.display = 'block';
    sourcesList.innerHTML = sources.map(source => `
        <div class="source-item" onclick="openArticle('${source.article_id}')">
            <div class="source-title">${escapeHtml(source.title || 'Untitled')}</div>
            <div class="source-meta">${escapeHtml(source.source || 'Unknown source')}</div>
        </div>
    `).join('');
}

// Legacy function for backwards compatibility
function setQuery(query) {
    setAIChatQuery(query);
}

// Keep old sendChatMessage for backwards compatibility if needed
async function sendChatMessage() {
    // Redirect to new AI Chat
    sendAIChatMessage();
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

        let responseHtml = result.response || '';
        if (typeof marked !== 'undefined') {
            responseHtml = marked.parse(responseHtml);
        } else {
            responseHtml = formatResponse(responseHtml);
        }

        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            `<div class="markdown-content">${responseHtml}</div>`;

    } catch (error) {
        document.getElementById(loadingId).querySelector('.message-content').innerHTML =
            '<p>Sorry, there was an error.</p>';
    }

    scrollToBottom(messagesDiv);
}

// ============ Newsletter Modal ============

function openNewsletterModal() {
    document.getElementById('newsletter-modal').classList.add('active');
}

window.closeNewsletterModal = function () {
    document.getElementById('newsletter-modal').classList.remove('active');
    document.getElementById('app-newsletter-message').style.display = 'none';
};

function initializeNewsletter() {
    const btn = document.getElementById('nav-newsletter-btn');
    if (btn) {
        btn.addEventListener('click', openNewsletterModal);
    }

    const form = document.getElementById('app-newsletter-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const category = document.getElementById('app-newsletter-category').value;
            const msgDiv = document.getElementById('app-newsletter-message');
            const submitBtn = form.querySelector('button');

            // Get user email
            let email = '';
            if (window.Clerk && window.Clerk.user) {
                email = window.Clerk.user.emailAddresses[0].emailAddress;
            } else {
                showNotification('Error: User not identified', 'error');
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerHTML = '📨 Sending article...';

            try {
                const result = await fetchAPI('/newsletter/subscribe', {
                    method: 'POST',
                    body: JSON.stringify({
                        email: email,
                        name: window.Clerk.user.firstName,
                        category: category
                    })
                });

                if (result.success) {
                    msgDiv.innerHTML = `✅ <strong>Success!</strong> ${category} article sent to ${email}. Check your inbox!`;
                    msgDiv.style.background = 'rgba(0, 255, 150, 0.1)';
                    msgDiv.style.color = '#00ff96';
                    msgDiv.style.border = '1px solid rgba(0, 255, 150, 0.2)';

                    showNotification('Newsletter article sent!');
                    setTimeout(() => window.closeNewsletterModal(), 4000);
                } else {
                    throw new Error(result.message);
                }
            } catch (error) {
                msgDiv.innerHTML = `❌ ${error.message || 'Failed to subscribe'}`;
                msgDiv.style.background = 'rgba(255, 107, 107, 0.1)';
                msgDiv.style.color = '#ff6b6b';
                msgDiv.style.border = '1px solid rgba(255, 107, 107, 0.2)';
            }

            msgDiv.style.display = 'block';
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Subscribe & Get Article';
        });
    }
}

// ============ Comparison Functions ============

// Update all compare dropdowns
function updateCompareDropdowns() {
    updateCompareDropdown(1);
    updateCompareDropdown(2);
}

// Update specific compare dropdown based on its category filter
function updateCompareDropdown(slotId) {
    const catSelect = document.getElementById(`compare-cat-${slotId}`);
    const articleSelect = document.getElementById(`compare-select-${slotId}`);

    if (!catSelect || !articleSelect) return;

    const category = catSelect.value;
    const currentVal = articleSelect.value;

    // Filter articles
    let filteredArticles = articles;
    if (category) {
        filteredArticles = articles.filter(a => a.category === category);
    }

    const options = filteredArticles.map(a =>
        `<option value="${a.article_id}">${escapeHtml(truncate(a.title, 60))}</option>`
    ).join('');

    articleSelect.innerHTML = `<option value="">Select an article...</option>${options}`;

    // Restore selection if valid
    if (currentVal && filteredArticles.some(a => a.article_id === currentVal)) {
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

        // Reload global articles (resets main feed, but ensures we have data)
        // We pass the current main filter to loadNewsFeed so we don't disrupt the main feed too much if user goes back
        const mainCategory = document.getElementById('category-filter').value;
        await loadNewsFeed(mainCategory);

        // The loadNewsFeed calls updateCompareDropdowns, so the dropdowns will refresh automatically

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

    const article1 = articles.find(a => a.article_id === id1);
    const article2 = articles.find(a => a.article_id === id2);

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

        // Use marked.js to render markdown properly
        if (typeof marked !== 'undefined') {
            resultDiv.innerHTML = `<div class="markdown-content">${marked.parse(result.response || '')}</div>`;
        } else {
            resultDiv.innerHTML = `<p>${formatResponse(result.response)}</p>`;
        }

    } catch (error) {
        resultDiv.innerHTML = '<p style="color: var(--accent-danger)">Error comparing articles. Please try again.</p>';
    }
}

// ============ WebSocket ============

function connectWebSocket() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('connection-status');

    try {
        // Support configurable WebSocket URL for separate frontend/backend deployment
        const wsUrl = WS_BASE
            ? `${WS_BASE}/ws/feed`
            : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/feed`;

        wsConnection = new WebSocket(wsUrl);

        wsConnection.onopen = () => {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        };

        wsConnection.onmessage = (event) => {
            if (event.data !== 'pong') {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'new_article' && data.article) {
                        // Prepend new article to the feed
                        prependNewArticle(data.article);
                    }
                } catch (e) {
                    console.debug('Non-JSON WebSocket message:', event.data);
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

// ============ Theme Functions ============

function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');

    if (theme === 'light') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

// ============ Push Notifications ============

let pushSubscription = null;

async function initializePushNotifications() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
        console.log('Push notifications not supported');
        return;
    }

    try {
        // Register service worker
        const registration = await navigator.serviceWorker.register('/static/sw.js');
        console.log('Service Worker registered:', registration.scope);

        // Check existing subscription
        pushSubscription = await registration.pushManager.getSubscription();
        updateNotificationUI(!!pushSubscription);

    } catch (error) {
        console.error('Service Worker registration failed:', error);
    }
}

async function enablePushNotifications() {
    try {
        // Request permission
        const permission = await Notification.requestPermission();
        if (permission !== 'granted') {
            showToast('Notification permission denied', 'error');
            return false;
        }

        // Get service worker registration
        const registration = await navigator.serviceWorker.ready;

        // Get VAPID public key from server
        const response = await fetch('/api/push/vapid-key');
        const { publicKey } = await response.json();

        // Convert VAPID key to Uint8Array
        const vapidKey = urlBase64ToUint8Array(publicKey);

        // Subscribe to push
        pushSubscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: vapidKey
        });

        // Send subscription to server
        const subscribeResponse = await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subscription: pushSubscription.toJSON(),
                user_id: window.currentUserId
            })
        });

        if (subscribeResponse.ok) {
            showToast('Notifications enabled!', 'success');
            updateNotificationUI(true);
            return true;
        }

    } catch (error) {
        console.error('Error enabling notifications:', error);
        showToast('Failed to enable notifications', 'error');
    }
    return false;
}

async function disablePushNotifications() {
    try {
        if (pushSubscription) {
            await pushSubscription.unsubscribe();

            // Notify server
            await fetch(`/api/push/unsubscribe?user_id=${window.currentUserId}`, {
                method: 'DELETE'
            });

            pushSubscription = null;
            updateNotificationUI(false);
            showToast('Notifications disabled', 'info');
        }
    } catch (error) {
        console.error('Error disabling notifications:', error);
    }
}

async function sendTestNotification() {
    try {
        const token = await window.Clerk?.session?.getToken();
        const response = await fetch('/api/push/test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                title: '🎉 Test Notification',
                body: 'Push notifications are working! You will receive breaking news alerts here.'
            })
        });

        if (response.ok) {
            showToast('Test notification sent!', 'success');
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to send test', 'error');
        }
    } catch (error) {
        console.error('Error sending test notification:', error);
        showToast('Failed to send test notification', 'error');
    }
}

function updateNotificationUI(enabled) {
    const btn = document.getElementById('notification-toggle-btn');
    const status = document.getElementById('notification-status');

    if (btn) {
        btn.textContent = enabled ? 'Disable Notifications' : 'Enable Notifications';
        btn.classList.toggle('btn-danger', enabled);
    }
    if (status) {
        status.textContent = enabled ? 'Enabled' : 'Disabled';
        status.style.color = enabled ? 'var(--accent-success)' : 'var(--text-tertiary)';
    }
}

// Convert VAPID key from base64 to Uint8Array
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
        .replace(/-/g, '+')
        .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

// Listen for messages from service worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data.type === 'OPEN_ARTICLE') {
            openArticle(event.data.articleId);
        }
    });
}

// ============ Demo Mode Functions ============

function toggleDemoPanel() {
    const panel = document.getElementById('demo-panel');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
    } else {
        panel.style.display = 'none';
    }
}

// Initialize demo toggle button
document.addEventListener('DOMContentLoaded', () => {
    const demoBtn = document.getElementById('demo-toggle-btn');
    if (demoBtn) {
        demoBtn.addEventListener('click', toggleDemoPanel);
    }

    // Start polling knowledge base status
    loadKnowledgeBaseStatus();
    setInterval(loadKnowledgeBaseStatus, 10000); // Poll every 10 seconds
});

async function injectDemoArticle() {
    const titleInput = document.getElementById('demo-title');
    const contentInput = document.getElementById('demo-content');
    const categorySelect = document.getElementById('demo-category');
    const topicsInput = document.getElementById('demo-topics');
    const injectBtn = document.getElementById('inject-demo-btn');

    const title = titleInput.value.trim();
    const content = contentInput.value.trim();
    const category = categorySelect.value;
    const topicsStr = topicsInput.value.trim();
    const topics = topicsStr ? topicsStr.split(',').map(t => t.trim()) : [];

    if (!title || !content) {
        showNotification('Please enter a title and content', 'warning');
        return;
    }

    // Disable button during injection
    const originalContent = injectBtn.innerHTML;
    injectBtn.innerHTML = `
        <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
        </svg>
        Injecting...
    `;
    injectBtn.disabled = true;

    try {
        const result = await fetchAPI('/demo/inject-article', {
            method: 'POST',
            body: JSON.stringify({
                title,
                content,
                category,
                topics
            })
        });

        showNotification(`✅ Article injected! ID: ${result.article_id}`, 'success');

        // Clear inputs
        titleInput.value = '';
        contentInput.value = '';
        topicsInput.value = '';

        // Trigger pulse animation
        triggerKnowledgeBasePulse();

        // Refresh knowledge base status
        await loadKnowledgeBaseStatus();

        // Auto-switch to AI Chat with a suggestion
        showNotification(`💡 Tip: Ask the AI about "${title.split(' ').slice(0, 4).join(' ')}..."`, 'info');

    } catch (error) {
        console.error('Demo injection error:', error);
        showNotification('❌ Failed to inject article', 'error');
    }

    injectBtn.innerHTML = originalContent;
    injectBtn.disabled = false;
}

// ============ Knowledge Base Status Functions ============

async function loadKnowledgeBaseStatus() {
    try {
        const status = await fetchAPI('/knowledge-base/status');

        // Update article count
        const countEl = document.getElementById('kb-article-count');
        if (countEl) {
            const oldCount = parseInt(countEl.textContent) || 0;
            const newCount = status.article_count || 0;
            countEl.textContent = newCount;

            // Trigger pulse if count increased
            if (newCount > oldCount && oldCount > 0) {
                triggerKnowledgeBasePulse();
            }
        }

        // Update search method badge
        const methodBadge = document.getElementById('search-method-badge');
        if (methodBadge) {
            const method = status.search_method || 'keyword';
            methodBadge.textContent = method;
            methodBadge.setAttribute('data-method', method);
            methodBadge.title = `Search: ${method === 'hybrid' ? 'Vector + Keyword' : method === 'vector' ? 'Vector Only' : 'Keyword Only'}`;
        }

        // Also update legacy article count elements
        const articleCountEl = document.getElementById('article-count');
        if (articleCountEl) {
            articleCountEl.textContent = `${status.article_count} articles`;
        }

        const indexedCountEl = document.getElementById('indexed-count');
        if (indexedCountEl) {
            const embeddingInfo = status.embeddings_available ? ` (${status.embeddings_count} embeddings)` : '';
            indexedCountEl.textContent = `${status.article_count} articles indexed${embeddingInfo}`;
        }

    } catch (error) {
        console.debug('Could not load knowledge base status:', error);
    }
}

function triggerKnowledgeBasePulse() {
    const pulseEl = document.getElementById('kb-pulse');
    if (pulseEl) {
        // Remove and re-add the class to restart animation
        pulseEl.classList.remove('active');
        void pulseEl.offsetWidth; // Force reflow
        pulseEl.classList.add('active');

        // Remove class after animation completes
        setTimeout(() => {
            pulseEl.classList.remove('active');
        }, 1000);
    }
}

// Enhanced prependNewArticle to trigger pulse
const originalPrependNewArticle = prependNewArticle;
function prependNewArticle(article) {
    // Call original function if it exists
    if (typeof originalPrependNewArticle === 'function') {
        originalPrependNewArticle(article);
    }

    // Trigger knowledge base pulse
    triggerKnowledgeBasePulse();

    // Update knowledge base status
    loadKnowledgeBaseStatus();
}

// ============ Pathway Status Functions ============

function updatePathwayStatus(status) {
    const indicator = document.getElementById('pathway-indicator');
    const label = document.getElementById('pathway-label');
    const container = document.getElementById('pathway-status');

    if (!indicator || !label || !container) return;

    const pathway = status.pathway || {};
    const isRunning = pathway.running || false;
    const isEnabled = pathway.enabled || false;

    if (isRunning) {
        indicator.classList.add('active');
        label.textContent = 'Pathway Active';
        container.title = `Pathway Streaming Engine - Running\nDocuments: ${pathway.documents_indexed || 0}\nAvg Latency: ${pathway.avg_query_latency_ms || 0}ms`;
        container.style.background = 'linear-gradient(135deg, rgba(0, 198, 255, 0.2), rgba(0, 114, 255, 0.2))';
    } else if (isEnabled) {
        indicator.classList.remove('active');
        label.textContent = 'Pathway Starting...';
        container.title = 'Pathway Streaming Engine - Starting';
        container.style.background = 'rgba(255, 193, 7, 0.2)';
    } else {
        indicator.classList.remove('active');
        label.textContent = 'Local RAG';
        container.title = 'Using local RAG engine (Pathway disabled)';
        container.style.background = 'rgba(108, 117, 125, 0.2)';
    }
}

// Update loadKnowledgeBaseStatus to also update Pathway status
const originalLoadKnowledgeBaseStatus = loadKnowledgeBaseStatus;
async function loadKnowledgeBaseStatus() {
    try {
        const status = await fetchAPI('/knowledge-base/status');

        // Update article count
        const countEl = document.getElementById('kb-article-count');
        if (countEl) {
            const oldCount = parseInt(countEl.textContent) || 0;
            const newCount = status.article_count || 0;
            countEl.textContent = newCount;

            // Trigger pulse if count increased
            if (newCount > oldCount && oldCount > 0) {
                triggerKnowledgeBasePulse();
            }
        }

        // Update search method badge
        const methodBadge = document.getElementById('search-method-badge');
        if (methodBadge) {
            const method = status.search_method || 'keyword';
            // Show "pathway" if Pathway is being used
            const displayMethod = status.pathway?.running ? 'pathway' : method;
            methodBadge.textContent = displayMethod;
            methodBadge.setAttribute('data-method', displayMethod);
            methodBadge.title = displayMethod === 'pathway' ? 'Pathway Real-Time RAG' :
                `Search: ${method === 'hybrid' ? 'Vector + Keyword' : method === 'vector' ? 'Vector Only' : 'Keyword Only'}`;
        }

        // Update Pathway status
        updatePathwayStatus(status);

        // Also update legacy article count elements
        const articleCountEl = document.getElementById('article-count');
        if (articleCountEl) {
            articleCountEl.textContent = `${status.article_count} articles`;
        }

        const indexedCountEl = document.getElementById('indexed-count');
        if (indexedCountEl) {
            const embeddingInfo = status.embeddings_available ? ` (${status.embeddings_count} embeddings)` : '';
            indexedCountEl.textContent = `${status.article_count} articles indexed${embeddingInfo}`;
        }

    } catch (error) {
        console.debug('Could not load knowledge base status:', error);
    }
}

// ============ Dynamism Test Functions ============

async function testDynamism() {
    const modal = document.getElementById('dynamism-modal');
    const loading = document.getElementById('dynamism-loading');
    const result = document.getElementById('dynamism-result');

    if (!modal) return;

    // Show modal with loading state
    modal.style.display = 'flex';
    loading.style.display = 'flex';
    result.style.display = 'none';

    try {
        const response = await fetchAPI('/demo/test-dynamism', {
            method: 'POST'
        });

        // Hide loading, show result
        loading.style.display = 'none';
        result.style.display = 'block';

        // Populate results
        const demo = response.demonstration;
        const proof = response.proof_of_dynamism;

        // Set verdict
        const verdictEl = document.getElementById('dynamism-verdict');
        if (proof.answer_changed || proof.new_data_reflected) {
            verdictEl.innerHTML = '✅ SUCCESS: Real-time update demonstrated!';
            verdictEl.className = 'dynamism-verdict success';
        } else {
            verdictEl.innerHTML = '⚠️ Article indexed, answer may not have changed significantly';
            verdictEl.className = 'dynamism-verdict warning';
        }

        // Before response
        document.getElementById('before-response').textContent = demo.before.response;
        document.getElementById('before-meta').textContent =
            `${demo.before.documents_found} docs | ${demo.before.search_method} | ${demo.before.latency_ms}ms`;

        // After response
        document.getElementById('after-response').textContent = demo.after.response;
        document.getElementById('after-meta').textContent =
            `${demo.after.documents_found} docs | ${demo.after.search_method} | ${demo.after.latency_ms}ms` +
            (demo.after.new_article_in_context ? ' | ✓ New article used' : '');

        // Injection info
        document.getElementById('injection-info').innerHTML = `
            <strong>Injected:</strong><br>
            "${demo.injection.title.substring(0, 30)}..."<br>
            <small>${demo.injection.indexing_latency_ms}ms | Pathway: ${demo.injection.pathway_indexed ? '✓' : '✗'}</small>
        `;

        // Total time
        document.getElementById('total-time').textContent =
            `Total time: ${proof.total_time_ms}ms`;

        // Refresh knowledge base status
        loadKnowledgeBaseStatus();

        showNotification('⚡ Dynamism test completed!', 'success');

    } catch (error) {
        console.error('Dynamism test error:', error);
        loading.style.display = 'none';
        result.style.display = 'block';
        document.getElementById('dynamism-verdict').innerHTML = '❌ Test failed: ' + error.message;
        document.getElementById('dynamism-verdict').className = 'dynamism-verdict error';
    }
}

function closeDynamismModal() {
    const modal = document.getElementById('dynamism-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('dynamism-modal');
    if (modal && e.target === modal) {
        closeDynamismModal();
    }
});

