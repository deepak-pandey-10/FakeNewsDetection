// =============================================
//  Fake News Detector — Frontend Logic
// =============================================

const DOM = {
    // Tabs
    tabs: document.querySelectorAll(".tab"),
    panelDetect: document.getElementById("panel-detect"),
    panelStats: document.getElementById("panel-stats"),
    // Detect
    input: document.getElementById("news-input"),
    charCount: document.getElementById("char-count"),
    analyzeBtn: document.getElementById("analyze-btn"),
    clearBtn: document.getElementById("clear-btn"),
    inputSection: document.getElementById("input-section"),
    resultSection: document.getElementById("result-section"),
    resultCard: document.getElementById("result-card"),
    resultIconWrap: document.getElementById("result-icon-wrap"),
    resultLabel: document.getElementById("result-label"),
    resultDescription: document.getElementById("result-description"),
    confidenceValue: document.getElementById("confidence-value"),
    confidenceFill: document.getElementById("confidence-fill"),
    wordBars: document.getElementById("word-bars"),
    tryAgainBtn: document.getElementById("try-again-btn"),
    // Stats
    statsLoading: document.getElementById("stats-loading"),
    statsGrid: document.getElementById("stats-grid"),
    modelInfoGrid: document.getElementById("model-info-grid"),
    statTotal: document.getElementById("stat-total"),
    statReal: document.getElementById("stat-real"),
    statFake: document.getElementById("stat-fake"),
    statAvgConf: document.getElementById("stat-avg-conf"),
    donutReal: document.getElementById("donut-real"),
    donutFake: document.getElementById("donut-fake"),
    donutWrap: document.getElementById("donut-wrap"),
    sessionEmpty: document.getElementById("session-empty"),
    fakeFeatList: document.getElementById("fake-features-list"),
    realFeatList: document.getElementById("real-features-list"),
    historyBody: document.getElementById("history-body"),
    historyTableWrap: document.getElementById("history-table-wrap"),
    historyEmpty: document.getElementById("history-empty"),
};

// SVG icons
const ICONS = {
    real: `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
        <path d="M22 4L12 14.01l-3-3"/>
    </svg>`,
    fake: `<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>`,
};

const CIRCUMFERENCE = 2 * Math.PI * 50; // donut circle r=50

// =============================================
// TAB SWITCHING
// =============================================
DOM.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        const target = tab.dataset.tab;

        DOM.tabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");

        document.querySelectorAll(".tab-content").forEach((p) => p.classList.remove("active"));
        document.getElementById(`panel-${target}`).classList.add("active");

        if (target === "stats") loadStats();
    });
});

// =============================================
// DETECT TAB
// =============================================

// Character counter
DOM.input.addEventListener("input", () => {
    const len = DOM.input.value.length;
    DOM.charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? "s" : ""}`;
});

// Analyze
DOM.analyzeBtn.addEventListener("click", async () => {
    const text = DOM.input.value.trim();
    if (!text) {
        showError("Please paste some text to analyze.");
        return;
    }

    DOM.analyzeBtn.classList.add("loading");
    DOM.analyzeBtn.disabled = true;

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || "Server error");
        }

        const data = await res.json();
        showResult(data);
    } catch (err) {
        showError(err.message || "Something went wrong. Please try again.");
    } finally {
        DOM.analyzeBtn.classList.remove("loading");
        DOM.analyzeBtn.disabled = false;
    }
});

// Show Result
function showResult({ prediction, confidence, top_words }) {
    const isReal = prediction === "Real";
    const cls = isReal ? "real" : "fake";

    DOM.resultCard.classList.remove("real", "fake");
    DOM.resultCard.classList.add(cls);

    DOM.resultIconWrap.innerHTML = isReal ? ICONS.real : ICONS.fake;

    DOM.resultLabel.textContent = isReal ? "Likely Real News" : "Likely Fake News";
    DOM.resultDescription.textContent = isReal
        ? "Our AI model indicates this article appears to be credible and authentic."
        : "Our AI model flags this article as potentially misleading or fabricated.";

    DOM.confidenceValue.textContent = `${confidence}%`;
    DOM.confidenceFill.style.width = "0%";

    // Word contributions
    renderWordBars(top_words || []);

    // Show result, hide input
    DOM.inputSection.style.display = "none";
    DOM.resultSection.style.display = "block";

    // Re-trigger animations
    DOM.resultCard.style.animation = "none";
    DOM.resultCard.offsetHeight;
    DOM.resultCard.style.animation = "";
    DOM.resultIconWrap.style.animation = "none";
    DOM.resultIconWrap.offsetHeight;
    DOM.resultIconWrap.style.animation = "";

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            DOM.confidenceFill.style.width = `${confidence}%`;
        });
    });
}

// Render word contribution bars
function renderWordBars(words) {
    if (!words.length) {
        DOM.wordBars.innerHTML = '<p style="color:var(--text-muted);font-size:0.82rem;">No significant word contributions found.</p>';
        return;
    }

    const maxAbs = Math.max(...words.map((w) => Math.abs(w.contribution)));

    DOM.wordBars.innerHTML = words
        .map((w) => {
            const pct = maxAbs > 0 ? (Math.abs(w.contribution) / maxAbs) * 100 : 0;
            const cls = w.contribution > 0 ? "positive" : "negative";
            const direction = w.contribution > 0 ? "→ Real" : "→ Fake";
            return `
                <div class="word-bar-row">
                    <span class="word-bar-label">${escapeHtml(w.word)}</span>
                    <div class="word-bar-track">
                        <div class="word-bar-fill ${cls}" style="width:0%;" data-w="${pct}"></div>
                    </div>
                    <span class="word-bar-value">${direction}</span>
                </div>`;
        })
        .join("");

    // Animate bars
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            DOM.wordBars.querySelectorAll(".word-bar-fill").forEach((el) => {
                el.style.width = el.dataset.w + "%";
            });
        });
    });
}

// Try Again
DOM.tryAgainBtn.addEventListener("click", () => {
    DOM.resultSection.style.display = "none";
    DOM.inputSection.style.display = "block";
    DOM.input.focus();
});

// Clear
DOM.clearBtn.addEventListener("click", () => {
    DOM.input.value = "";
    DOM.charCount.textContent = "0 characters";
    DOM.input.focus();
});

// Keyboard shortcut
DOM.input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        DOM.analyzeBtn.click();
    }
});

// =============================================
// STATS TAB
// =============================================
let statsLoaded = false;

async function loadStats() {
    DOM.statsLoading.style.display = "flex";
    DOM.statsGrid.style.display = "none";

    try {
        const res = await fetch("/stats");
        if (!res.ok) throw new Error("Failed to load stats");
        const data = await res.json();
        renderStats(data);
    } catch (err) {
        showError("Could not load model statistics.");
    }
}

function renderStats({ model, session }) {
    // ── Model Info ──
    // Friendly display names
    const modelTypeNames = {
        LogisticRegression: "Logistic Regression",
        RandomForestClassifier: "Random Forest",
        SVC: "Support Vector Machine",
        MultinomialNB: "Multinomial Naive Bayes",
        GradientBoostingClassifier: "Gradient Boosting",
    };
    const solverNames = {
        lbfgs: "L-BFGS (Quasi-Newton)",
        liblinear: "LIBLINEAR (Coordinate Descent)",
        newton_cg: "Newton-CG (Conjugate Gradient)",
        sag: "SAG (Stochastic Avg. Gradient)",
        saga: "SAGA (Improved SAG)",
    };
    const classNames = { 0: "Fake (0)", 1: "Real (1)" };

    const infoItems = [
        { label: "Model Type", value: modelTypeNames[model.model_type] || model.model_type },
        { label: "Solver", value: solverNames[model.solver] || model.solver },
        { label: "Regularization (C)", value: model.regularization },
        { label: "Max Iterations", value: model.max_iter },
        { label: "Vocabulary Size", value: Number(model.vocabulary_size).toLocaleString() },
        { label: "Total Features", value: Number(model.n_features).toLocaleString() },
        { label: "Classes", value: model.classes.map(c => classNames[c] || c).join(", ") },
    ];

    DOM.modelInfoGrid.innerHTML = infoItems
        .map(
            (item) => `
        <div class="info-chip">
            <span class="info-chip-label">${item.label}</span>
            <span class="info-chip-value">${item.value}</span>
        </div>`
        )
        .join("");

    // ── Session Counters ──
    DOM.statTotal.textContent = session.total_predictions;
    DOM.statReal.textContent = session.real_count;
    DOM.statFake.textContent = session.fake_count;
    DOM.statAvgConf.textContent = session.avg_confidence + "%";

    const hasSession = session.total_predictions > 0;
    DOM.sessionEmpty.style.display = hasSession ? "none" : "block";
    DOM.donutWrap.style.display = hasSession ? "flex" : "none";

    // ── Donut Chart ──
    if (hasSession) {
        const total = session.total_predictions;
        const realPct = session.real_count / total;
        const fakePct = session.fake_count / total;

        DOM.donutReal.setAttribute("stroke-dasharray", `${realPct * CIRCUMFERENCE} ${CIRCUMFERENCE}`);
        DOM.donutFake.setAttribute("stroke-dasharray", `${fakePct * CIRCUMFERENCE} ${CIRCUMFERENCE}`);
        DOM.donutFake.setAttribute("stroke-dashoffset", `${-realPct * CIRCUMFERENCE}`);
    }

    // ── Feature Lists ──
    renderFeatureList(DOM.fakeFeatList, model.top_fake_features, "fake-bar");
    renderFeatureList(DOM.realFeatList, model.top_real_features, "real-bar");

    // ── History Table ──
    if (session.history.length > 0) {
        DOM.historyEmpty.style.display = "none";
        DOM.historyTableWrap.style.display = "block";
        DOM.historyBody.innerHTML = session.history
            .slice()
            .reverse()
            .map(
                (h, i) => `
            <tr>
                <td>${session.history.length - i}</td>
                <td>${escapeHtml(h.text)}</td>
                <td><span class="pill pill-${h.prediction.toLowerCase()}">${h.prediction}</span></td>
                <td>${h.confidence}%</td>
            </tr>`
            )
            .join("");
    } else {
        DOM.historyEmpty.style.display = "block";
        DOM.historyTableWrap.style.display = "none";
    }

    // Show grid
    DOM.statsLoading.style.display = "none";
    DOM.statsGrid.style.display = "grid";
}

function renderFeatureList(container, features, barClass) {
    const maxW = Math.max(...features.map((f) => Math.abs(f.weight)));

    container.innerHTML = features
        .map((f, i) => {
            const pct = maxW > 0 ? (Math.abs(f.weight) / maxW) * 100 : 0;
            return `
            <div class="feature-row">
                <span class="feature-rank">${i + 1}</span>
                <span class="feature-word">${escapeHtml(f.word)}</span>
                <div class="feature-bar-track">
                    <div class="feature-bar-fill ${barClass}" style="width:0%;" data-w="${pct}"></div>
                </div>
                <span class="feature-weight">${Math.abs(f.weight).toFixed(3)}</span>
            </div>`;
        })
        .join("");

    // Animate bars after render
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            container.querySelectorAll(".feature-bar-fill").forEach((el) => {
                el.style.width = el.dataset.w + "%";
            });
        });
    });
}

// =============================================
// UTILITIES
// =============================================
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function showError(message) {
    const existing = document.querySelector(".error-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "error-toast";
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            toast.classList.add("visible");
        });
    });

    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => toast.remove(), 400);
    }, 3500);
}
