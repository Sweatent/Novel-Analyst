document.addEventListener('DOMContentLoaded', () => {
    const sentenceText = document.getElementById('sentence-text');
    const sentenceSpeaker = document.getElementById('sentence-speaker');
    const chapterTitle = document.getElementById('chapter-title');
    const refreshBtn = document.getElementById('refresh-btn');
    const favoriteBtn = document.getElementById('favorite-btn');
    const viewSourceBtn = document.getElementById('view-source-btn');
    const copyBtn = document.getElementById('copy-btn');
    const favoritesContainer = document.getElementById('favorites-container');
    const favoritesList = document.getElementById('favorites-list');
    const closeFavoritesBtn = document.getElementById('close-favorites-btn');
    const showFavoritesBtn = document.getElementById('show-favorites-btn');
    const toastContainer = document.getElementById('toast-container');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('settings-panel');
    const closeSettingsBtn = document.getElementById('close-settings-btn');
    const increaseFontSizeBtn = document.getElementById('increase-font-size-btn');
    const decreaseFontSizeBtn = document.getElementById('decrease-font-size-btn');
    const resetFontSizeBtn = document.getElementById('reset-font-size-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    let sentences = [];
    let currentSentence = {};
    let favorites = JSON.parse(localStorage.getItem('favorites')) || [];

    function loadSettings() {
        const savedFontSize = localStorage.getItem('fontSize');
        if (savedFontSize) {
            sentenceText.style.fontSize = savedFontSize;
        } else {
            sentenceText.style.fontSize = '2em'; // Default size
        }
    }

    function saveFontSize() {
        localStorage.setItem('fontSize', sentenceText.style.fontSize);
    }

    function changeFontSize(amount) {
        const currentSize = parseFloat(window.getComputedStyle(sentenceText, null).getPropertyValue('font-size'));
        sentenceText.style.fontSize = `${currentSize + amount}px`;
        saveFontSize();
    }

    fetch('test.json')
        .then(response => response.json())
        .then(data => {
            sentences = data;
            displayRandomSentence();
        })
        .catch(error => {
            console.error('Error fetching sentences:', error);
            sentenceText.textContent = '加载一言失败，请检查 test.json 文件是否存在。';
        });

    function displayRandomSentence() {
        if (sentences.length === 0) {
            return;
        }
        const randomIndex = Math.floor(Math.random() * sentences.length);
        currentSentence = sentences[randomIndex];
        sentenceText.textContent = currentSentence.sentence;
        sentenceSpeaker.textContent = `—— ${currentSentence.speaker}`;
        chapterTitle.textContent = `出自: ${currentSentence.chapter_title}`;
        chapterTitle.style.display = 'none';
        updateFavoriteButton();
    }

    function updateFavoriteButton() {
        const isFavorited = favorites.some(fav => fav.sentence === currentSentence.sentence);
        favoriteBtn.textContent = isFavorited ? '❤️' : '⭐';
    }

    function toggleFavorite() {
        const isFavorited = favorites.some(fav => fav.sentence === currentSentence.sentence);
        if (isFavorited) {
            favorites = favorites.filter(fav => fav.sentence !== currentSentence.sentence);
        } else {
            favorites.push(currentSentence);
        }
        localStorage.setItem('favorites', JSON.stringify(favorites));
        updateFavoriteButton();
        renderFavorites();
    }

    function renderFavorites() {
        favoritesList.innerHTML = '';
        favorites.forEach((fav, index) => {
            const li = document.createElement('li');
            li.className = 'favorite-item';

            const textSpan = document.createElement('span');
            textSpan.className = 'favorite-item-text';
            textSpan.textContent = `${fav.sentence} —— ${fav.speaker}`;

            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'favorite-item-actions';

            const copyFavBtn = document.createElement('button');
            copyFavBtn.textContent = '📋';
            copyFavBtn.className = 'icon-btn-small';
            copyFavBtn.setAttribute('data-tooltip', '复制');
            copyFavBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const textToCopy = `${fav.sentence} —— ${fav.speaker} (来源: https://goldsay.twiap.dpdns.org/)`;
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showToast('已复制到剪贴板');
                }).catch(err => {
                    console.error('复制失败: ', err);
                    showToast('复制失败');
                });
            });

            const removeFavBtn = document.createElement('button');
            removeFavBtn.textContent = '🗑️';
            removeFavBtn.className = 'icon-btn-small';
            removeFavBtn.setAttribute('data-tooltip', '移除');
            removeFavBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeFavorite(index);
            });

            buttonContainer.appendChild(copyFavBtn);
            buttonContainer.appendChild(removeFavBtn);

            li.appendChild(textSpan);
            li.appendChild(buttonContainer);

            favoritesList.appendChild(li);
        });
    }

    function removeFavorite(index) {
        favorites.splice(index, 1);
        localStorage.setItem('favorites', JSON.stringify(favorites));
        renderFavorites();
        updateFavoriteButton();
        showToast('已从收藏夹移除');
    }

    function toggleSource() {
        chapterTitle.style.display = chapterTitle.style.display === 'none' ? 'block' : 'none';
    }

    refreshBtn.addEventListener('click', displayRandomSentence);
    favoriteBtn.addEventListener('click', toggleFavorite);
    viewSourceBtn.addEventListener('click', toggleSource);

    copyBtn.addEventListener('click', () => {
        if (currentSentence.sentence) {
            const textToCopy = `${currentSentence.sentence} —— ${currentSentence.speaker} (来源: https://goldsay.twiap.dpdns.org/)`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                showToast('已复制到剪贴板');
            }).catch(err => {
                console.error('复制失败: ', err);
                showToast('复制失败');
            });
        }
    });

    function animatePanel(panel, show) {
        if (show) {
            panel.style.display = 'block';
            anime({
                targets: panel,
                translateX: ['100%', '0%'],
                easing: 'easeOutQuad',
                duration: 400
            });
        } else {
            anime({
                targets: panel,
                translateX: ['0%', '100%'],
                easing: 'easeInQuad',
                duration: 400,
                complete: () => {
                    panel.style.display = 'none';
                }
            });
        }
    }

    settingsBtn.addEventListener('click', () => {
        animatePanel(settingsPanel, true);
    });

    closeSettingsBtn.addEventListener('click', () => {
        animatePanel(settingsPanel, false);
    });

    increaseFontSizeBtn.addEventListener('click', () => changeFontSize(1));
    decreaseFontSizeBtn.addEventListener('click', () => changeFontSize(-1));
    resetFontSizeBtn.addEventListener('click', () => {
        sentenceText.style.fontSize = '2em';
        localStorage.removeItem('fontSize');
    });

    showFavoritesBtn.addEventListener('click', () => {
        renderFavorites();
        animatePanel(favoritesContainer, true);
    });

    closeFavoritesBtn.addEventListener('click', () => {
        animatePanel(favoritesContainer, false);
    });

    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('show');
        }, 100);

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                toastContainer.removeChild(toast);
            }, 500);
        }, 10000);
    }

    window.addEventListener('keydown', (e) => {
        if (e.key === ' ' || e.key === 'ArrowRight' || e.key === 'ArrowDown') {
            e.preventDefault();
            displayRandomSentence();
        }
    });

    function applyTheme(theme) {
        if (theme === 'light') {
            document.body.classList.add('light-theme');
        } else {
            document.body.classList.remove('light-theme');
        }
    }

    function toggleTheme() {
        const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        applyTheme(newTheme);
        localStorage.setItem('theme', newTheme);
    }

    themeToggleBtn.addEventListener('click', toggleTheme);

    loadSettings();
    renderFavorites();
    showToast('来源于番茄小说《心跳引擎》，sweatent设计与筛选 (快捷键: 空格/→/↓)');

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme);
});