/**
 * Onboarding Questionnaire - JavaScript Logic
 * Handles navigation, response collection, and API integration
 */

// State management
const state = {
    currentQuestion: 1,
    totalQuestions: 9,
    responses: {
        categories: [],
        reading_depth: null,
        daily_time: null,
        content_formats: [],
        primary_reason: null,
        industry: null,
        regions: [],
        ai_summary_preference: null,
        importance_timely: 3,
        importance_accurate: 3,
        importance_engaging: 3
    },
    userId: null
};

// DOM Elements
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const questionsWrapper = document.getElementById('questions-wrapper');
const btnBack = document.getElementById('btn-back');
const btnNext = document.getElementById('btn-next');
const btnSubmit = document.getElementById('btn-submit');
const successScreen = document.getElementById('success-screen');
const btnStart = document.getElementById('btn-start');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initUserId();
    checkExistingOnboarding();
    setupEventListeners();
    setupSliders();
});

/**
 * Generate or retrieve user ID
 */
function initUserId() {
    let userId = localStorage.getItem('news_ai_user_id');
    if (!userId) {
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('news_ai_user_id', userId);
    }
    state.userId = userId;
}

/**
 * Check if user already completed onboarding
 */
async function checkExistingOnboarding() {
    try {
        const response = await fetch(`/api/onboarding/check/${state.userId}`);
        const data = await response.json();

        if (data.completed) {
            // Redirect to main app
            window.location.href = '/';
        }
    } catch (error) {
        console.log('No existing onboarding found, proceeding...');
    }
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Multi-select pills
    document.querySelectorAll('.multi-select .option-pill').forEach(pill => {
        pill.addEventListener('click', () => handleMultiSelect(pill));
    });

    // Single-select pills (industry)
    document.querySelectorAll('.single-select-grid .option-pill').forEach(pill => {
        pill.addEventListener('click', () => handleSingleSelectPill(pill));
    });

    // Single-select cards
    document.querySelectorAll('.single-select .option-card').forEach(card => {
        card.addEventListener('click', () => handleSingleSelect(card));
    });

    // Navigation
    btnBack.addEventListener('click', goBack);
    btnNext.addEventListener('click', goNext);
    btnSubmit.addEventListener('click', submitOnboarding);
    btnStart.addEventListener('click', () => {
        window.location.href = '/';
    });
}

/**
 * Setup slider interactions
 */
function setupSliders() {
    const sliders = {
        'slider-timely': 'timely-value',
        'slider-accurate': 'accurate-value',
        'slider-engaging': 'engaging-value'
    };

    Object.entries(sliders).forEach(([sliderId, valueId]) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);

        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value;

                // Update state
                if (sliderId === 'slider-timely') {
                    state.responses.importance_timely = parseInt(e.target.value);
                } else if (sliderId === 'slider-accurate') {
                    state.responses.importance_accurate = parseInt(e.target.value);
                } else if (sliderId === 'slider-engaging') {
                    state.responses.importance_engaging = parseInt(e.target.value);
                }
            });
        }
    });
}

/**
 * Handle multi-select option click
 */
function handleMultiSelect(pill) {
    pill.classList.toggle('selected');

    const field = pill.parentElement.dataset.field;
    const value = pill.dataset.value;

    if (pill.classList.contains('selected')) {
        if (!state.responses[field].includes(value)) {
            state.responses[field].push(value);
        }
    } else {
        state.responses[field] = state.responses[field].filter(v => v !== value);
    }

    // Add visual feedback
    pill.style.transform = 'scale(0.95)';
    setTimeout(() => {
        pill.style.transform = '';
    }, 150);
}

/**
 * Handle single-select pill click (industry)
 */
function handleSingleSelectPill(pill) {
    const parent = pill.parentElement;

    // Remove selected from siblings
    parent.querySelectorAll('.option-pill').forEach(p => p.classList.remove('selected'));

    // Select this one
    pill.classList.add('selected');

    const field = parent.dataset.field;
    state.responses[field] = pill.dataset.value;
}

/**
 * Handle single-select card click
 */
function handleSingleSelect(card) {
    const parent = card.parentElement;

    // Remove selected from siblings
    parent.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));

    // Select this one
    card.classList.add('selected');

    const field = parent.dataset.field;
    state.responses[field] = card.dataset.value;

    // Auto-advance after selection (with delay)
    if (state.currentQuestion < state.totalQuestions) {
        setTimeout(() => goNext(), 400);
    }
}

/**
 * Update progress bar and text
 */
function updateProgress() {
    const percentage = (state.currentQuestion / state.totalQuestions) * 100;
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `Question ${state.currentQuestion} of ${state.totalQuestions}`;
}

/**
 * Show question by number
 */
function showQuestion(num) {
    const questions = document.querySelectorAll('.question-card');
    const currentCard = document.querySelector('.question-card.active');
    const nextCard = document.querySelector(`.question-card[data-question="${num}"]`);

    if (!nextCard) return;

    // Determine direction
    const goingForward = num > state.currentQuestion;

    // Animate out current
    if (currentCard) {
        currentCard.classList.remove('active');
        currentCard.classList.add(goingForward ? 'exit-left' : '');
        setTimeout(() => {
            currentCard.classList.remove('exit-left');
        }, 400);
    }

    // Animate in new
    nextCard.classList.add('active');

    state.currentQuestion = num;
    updateProgress();
    updateNavButtons();
}

/**
 * Update navigation button states
 */
function updateNavButtons() {
    // Back button
    btnBack.disabled = state.currentQuestion === 1;

    // Next/Submit buttons
    if (state.currentQuestion === state.totalQuestions) {
        btnNext.classList.add('hidden');
        btnSubmit.classList.remove('hidden');
    } else {
        btnNext.classList.remove('hidden');
        btnSubmit.classList.add('hidden');
    }
}

/**
 * Go to previous question
 */
function goBack() {
    if (state.currentQuestion > 1) {
        showQuestion(state.currentQuestion - 1);
    }
}

/**
 * Go to next question
 */
function goNext() {
    if (!validateCurrentQuestion()) {
        shakeButton(btnNext);
        return;
    }

    if (state.currentQuestion < state.totalQuestions) {
        showQuestion(state.currentQuestion + 1);
    }
}

/**
 * Validate current question has a response
 */
function validateCurrentQuestion() {
    const currentCard = document.querySelector('.question-card.active');
    const field = currentCard.querySelector('[data-field]')?.dataset.field;

    if (!field) return true; // Slider question, always valid

    const response = state.responses[field];

    if (Array.isArray(response)) {
        return response.length > 0;
    }

    return response !== null && response !== undefined;
}

/**
 * Shake button animation for validation feedback
 */
function shakeButton(btn) {
    btn.style.animation = 'shake 0.5s ease-in-out';
    setTimeout(() => {
        btn.style.animation = '';
    }, 500);
}

// Add shake keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-5px); }
        40%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

/**
 * Submit onboarding data
 */
async function submitOnboarding() {
    if (!validateCurrentQuestion()) {
        shakeButton(btnSubmit);
        return;
    }

    btnSubmit.classList.add('loading');
    btnSubmit.innerHTML = '<span>Saving...</span>';

    try {
        const payload = {
            user_id: state.userId,
            categories: state.responses.categories,
            reading_depth: state.responses.reading_depth,
            daily_time: state.responses.daily_time,
            content_formats: state.responses.content_formats,
            primary_reason: state.responses.primary_reason,
            industry: state.responses.industry,
            regions: state.responses.regions,
            ai_summary_preference: state.responses.ai_summary_preference,
            importance_timely: state.responses.importance_timely,
            importance_accurate: state.responses.importance_accurate,
            importance_engaging: state.responses.importance_engaging
        };

        const response = await fetch('/api/onboarding', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            // Show success screen
            document.querySelector('.onboarding-container').classList.add('hidden');
            successScreen.classList.remove('hidden');
        } else {
            throw new Error(data.message || 'Failed to save');
        }
    } catch (error) {
        console.error('Error saving onboarding:', error);
        btnSubmit.classList.remove('loading');
        btnSubmit.innerHTML = `
            <span>Let's Go!</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
        `;
        alert('Something went wrong. Please try again.');
    }
}

// Initialize on load
updateProgress();
updateNavButtons();
