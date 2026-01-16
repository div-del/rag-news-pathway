/**
 * LiveLens Service Worker
 * Handles push notifications for breaking news
 */

const CACHE_NAME = 'livelens-v1';

// Install event
self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');
    self.skipWaiting();
});

// Activate event
self.addEventListener('activate', (event) => {
    console.log('[SW] Service worker activated');
    event.waitUntil(clients.claim());
});

// Push notification received
self.addEventListener('push', (event) => {
    console.log('[SW] Push notification received');

    let data = {
        title: 'LiveLens News',
        body: 'You have a new notification',
        icon: '/static/icon-192.png',
        badge: '/static/badge-72.png',
        url: '/app'
    };

    if (event.data) {
        try {
            data = { ...data, ...event.data.json() };
        } catch (e) {
            data.body = event.data.text();
        }
    }

    const options = {
        body: data.body,
        icon: data.icon || '/static/icon-192.png',
        badge: data.badge || '/static/badge-72.png',
        vibrate: [100, 50, 100],
        data: {
            url: data.url || '/app',
            articleId: data.articleId
        },
        actions: [
            { action: 'open', title: 'Read Article' },
            { action: 'dismiss', title: 'Dismiss' }
        ],
        tag: data.tag || 'livelens-notification',
        renotify: true,
        requireInteraction: data.breaking || false
    };

    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked:', event.action);

    event.notification.close();

    if (event.action === 'dismiss') {
        return;
    }

    const urlToOpen = event.notification.data?.url || '/app';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((clientList) => {
                // Check if app is already open
                for (const client of clientList) {
                    if (client.url.includes('/app') && 'focus' in client) {
                        // Navigate to article if specified
                        if (event.notification.data?.articleId) {
                            client.postMessage({
                                type: 'OPEN_ARTICLE',
                                articleId: event.notification.data.articleId
                            });
                        }
                        return client.focus();
                    }
                }
                // Open new window
                if (clients.openWindow) {
                    return clients.openWindow(urlToOpen);
                }
            })
    );
});

// Handle messages from main app
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});
