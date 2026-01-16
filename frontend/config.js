/**
 * Live Lens - Frontend Configuration
 * Configure API endpoints for different environments
 */

// Detect environment and set API base URL
const getApiBaseUrl = () => {
    // Check for explicit environment variable (set in HTML or build process)
    if (window.LIVE_LENS_API_URL) {
        return window.LIVE_LENS_API_URL;
    }

    // Check if we're on a Vercel deployment (frontend only)
    const hostname = window.location.hostname;

    // Production: Frontend on Vercel, Backend running locally
    // When you deploy frontend to Vercel but run backend locally,
    // you'll need to use ngrok or similar to expose your local backend
    if (hostname.includes('vercel.app') || hostname.includes('netlify.app')) {
        // Set this in the browser console or via script tag:
        // window.LIVE_LENS_BACKEND_URL = 'https://your-ngrok-url.ngrok.io'
        // Or for local-only demos, run everything locally
        return window.LIVE_LENS_BACKEND_URL || 'http://localhost:8000';
    }

    // Local development: same origin
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return '';  // Same origin, use relative paths
    }

    // Default: same origin (backend serves frontend)
    return '';
};

// WebSocket URL helper
const getWsBaseUrl = () => {
    const apiBase = getApiBaseUrl();

    if (!apiBase) {
        // Same origin - use current host
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}`;
    }

    // Convert HTTP URL to WebSocket URL
    return apiBase.replace('https://', 'wss://').replace('http://', 'ws://');
};

// Export configuration
window.LiveLensConfig = {
    API_BASE: getApiBaseUrl(),
    WS_BASE: getWsBaseUrl(),

    // Helper to get full API URL
    apiUrl: (endpoint) => `${getApiBaseUrl()}/api${endpoint}`,

    // Helper to get full WebSocket URL
    wsUrl: (endpoint) => `${getWsBaseUrl()}${endpoint}`
};

// Log configuration in development
if (window.location.hostname === 'localhost') {
    console.log('Live Lens Config:', window.LiveLensConfig);
}
