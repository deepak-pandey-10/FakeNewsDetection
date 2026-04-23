document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const textInput = document.getElementById('text');
    const factBtn = document.getElementById('fact-btn');
    const loadingUI = document.getElementById('loading');
    const resultsUI = document.getElementById('results');
    const errorUI = document.getElementById('error-container');
    const errorMsg = document.getElementById('error-message');

    // Results Elements
    const verdictHero = document.getElementById('verdict-hero');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictSummary = document.getElementById('verdict-summary');
    const verdictTimestamp = document.getElementById('verdict-timestamp');

    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    const confLabel = document.getElementById('conf-label');
    const predText = document.getElementById('prediction-text');
    const sourceCount = document.getElementById('source-count');
    const evidenceCount = document.getElementById('evidence-count');

    const reasoningPanel = document.getElementById('reasoning-panel');
    const reasoningBody = document.getElementById('reasoning-body');
    const evidenceGrid = document.getElementById('evidence-grid');

    // Sidebar tab switching
    const tabText = document.getElementById('tab-text');
    const tabUrl = document.getElementById('tab-url');
    const tabDoc = document.getElementById('tab-doc');
    const textGroup = document.getElementById('text-input-group');
    const urlGroup = document.getElementById('url-input-group');
    const docGroup = document.getElementById('doc-input-group');

    [tabText, tabUrl, tabDoc].forEach(tab => {
        tab?.addEventListener('click', () => {
            document.querySelectorAll('.sidebar-section li').forEach(li => li.classList.remove('active'));
            tab.classList.add('active');
            textGroup.classList.add('hidden');
            urlGroup.classList.add('hidden');
            docGroup.classList.add('hidden');
            if (tab === tabText) textGroup.classList.remove('hidden');
            if (tab === tabUrl) urlGroup.classList.remove('hidden');
            if (tab === tabDoc) docGroup.classList.remove('hidden');
        });
    });

    // History Panel
    const historyLink = document.getElementById('history-link');
    const historyPanel = document.getElementById('history-panel');
    const closeHistory = document.getElementById('close-history');
    const clearHistory = document.getElementById('clear-history');
    const historyList = document.getElementById('history-list');

    historyLink?.addEventListener('click', (e) => { e.preventDefault(); historyPanel.classList.toggle('hidden'); renderHistory(); });
    closeHistory?.addEventListener('click', () => historyPanel.classList.add('hidden'));
    clearHistory?.addEventListener('click', () => { localStorage.removeItem('truthguard_history'); renderHistory(); });

    function saveToHistory(text, data) {
        const history = JSON.parse(localStorage.getItem('truthguard_history') || '[]');
        history.unshift({ text: text.substring(0, 80), prediction: data.prediction || 'API ONLY', timestamp: new Date().toISOString() });
        if (history.length > 20) history.pop();
        localStorage.setItem('truthguard_history', JSON.stringify(history));
    }

    function renderHistory() {
        const history = JSON.parse(localStorage.getItem('truthguard_history') || '[]');
        historyList.innerHTML = '';
        if (history.length === 0) {
            historyList.innerHTML = '<li style="text-align:center; color: var(--text-muted);">No analysis history.</li>';
            return;
        }
        history.forEach(item => {
            const li = document.createElement('li');
            const date = new Date(item.timestamp).toLocaleString();
            li.innerHTML = `<div style="font-size:13px; color:var(--text-primary); margin-bottom:4px;">${item.text}${item.text.length >= 80 ? '...' : ''}</div>
                <div style="font-size:11px; color:var(--text-muted); display:flex; justify-content:space-between;"><span>${item.prediction}</span><span>${date}</span></div>`;
            historyList.appendChild(li);
        });
    }

    // New analysis button
    document.getElementById('new-analysis-btn')?.addEventListener('click', () => {
        resultsUI.classList.add('hidden');
        textInput.value = '';
        textInput.focus();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Export
    document.getElementById('export-pdf-btn')?.addEventListener('click', () => window.print());

    // Click Handlers
    factBtn?.addEventListener('click', () => triggerAnalysis());

    async function triggerAnalysis() {
        const activeTab = document.querySelector('.sidebar-section li.active').id;
        
        let fetchUrl = '/api/analyze';
        let options = { method: 'POST' };
        let historyLabel = '';

        if (activeTab === 'tab-text') {
            const textValue = document.getElementById('text').value;
            if (!textValue.trim()) {
                textInput.style.borderColor = 'var(--danger)';
                setTimeout(() => textInput.style.borderColor = '', 2000);
                return;
            }
            historyLabel = textValue;
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify({ text: textValue });
        } else if (activeTab === 'tab-url') {
            const urlValue = document.getElementById('url').value;
            if (!urlValue.trim()) {
                document.getElementById('url').style.borderColor = 'var(--danger)';
                setTimeout(() => document.getElementById('url').style.borderColor = '', 2000);
                return;
            }
            const fullUrl = urlValue.startsWith('http') ? urlValue : 'https://' + urlValue;
            historyLabel = fullUrl;
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify({ url: fullUrl });
        } else if (activeTab === 'tab-doc') {
            const fileInput = document.getElementById('file');
            if (!fileInput.files.length) {
                fileInput.style.borderColor = 'var(--danger)';
                setTimeout(() => fileInput.style.borderColor = '', 2000);
                return;
            }
            fetchUrl = '/api/analyze-file';
            historyLabel = fileInput.files[0].name;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            options.body = formData;
        }

        // Reset UI
        resultsUI.classList.add('hidden');
        errorUI.classList.add('hidden');
        loadingUI.classList.remove('hidden');
        if (factBtn) factBtn.disabled = true;

        try {
            const response = await fetch(fetchUrl, options);
            const data = await response.json();

            if (!response.ok || data.error) throw new Error(data.error || 'Request failed');

            saveToHistory(historyLabel, data);
            renderReport(data);
        } catch (err) {
            errorMsg.textContent = err.message;
            errorUI.classList.remove('hidden');
            loadingUI.classList.add('hidden');
        } finally {
            if (factBtn) factBtn.disabled = false;
        }
    }

    function renderReport(data) {
        loadingUI.classList.add('hidden');
        resultsUI.classList.remove('hidden');

        const factData = data.fact_verification || { agent_verdict: 'N/A', evidence: [] };
        const factResult = factData.agent_verdict || '';
        const evidence = factData.evidence || [];

        // ─── 1. Hero Verdict ───
        let heroClass, heroIcon, heroLabel, heroSummary;

        if (factResult.includes('UNVERIFIED')) {
            heroClass = 'verdict-warning';
            heroIcon = 'fa-shield-halved';
            heroLabel = 'UNVERIFIED';
            heroSummary = 'AI fact-checker is unavailable. The claim could not be verified for accuracy.';
        } else if (factResult.includes('VERIFIED') && !factResult.includes('UNVERIFIED')) {
            heroClass = 'verdict-safe';
            heroIcon = 'fa-circle-check';
            heroLabel = 'VERIFIED';
            heroSummary = 'This claim has been verified as accurate by AI analysis.';
        } else if (factResult.includes('FAKE')) {
            heroClass = 'verdict-danger';
            heroIcon = 'fa-circle-xmark';
            heroLabel = 'FAKE';
            heroSummary = 'This claim contains false or inaccurate information.';
        } else if (factResult.includes('MISLEADING')) {
            heroClass = 'verdict-warning';
            heroIcon = 'fa-triangle-exclamation';
            heroLabel = 'MISLEADING';
            heroSummary = 'This claim may be misleading or contains suspicious patterns.';
        } else {
            heroClass = 'verdict-neutral';
            heroIcon = 'fa-circle-question';
            heroLabel = 'ANALYSIS COMPLETE';
            heroSummary = 'Review the detailed findings below.';
        }

        verdictHero.className = `verdict-hero ${heroClass}`;
        verdictIcon.className = `fa-solid ${heroIcon}`;
        verdictLabel.textContent = heroLabel;
        verdictSummary.textContent = heroSummary;
        verdictTimestamp.textContent = new Date().toLocaleString();

        // ─── 2. Confidence Gauge (Static 100% since it's API reasoning) ───
        gaugeValue.textContent = `100%`;
        gaugeFill.style.strokeDashoffset = 0;
        gaugeFill.style.stroke = 'var(--success)';
        confLabel.textContent = 'API Reasoning Active';

        // ─── 3. ML Prediction Badge (Reused for API status) ───
        predText.textContent = 'API ONLY';

        // ─── 4. Source Count ───
        sourceCount.textContent = evidence.length;
        evidenceCount.textContent = `${evidence.length} Source${evidence.length !== 1 ? 's' : ''}`;

        // ─── 5. AI Reasoning ───
        const reasoningBadge = document.querySelector('.reasoning-badge');
        if (factResult && factResult !== 'N/A' && !factResult.includes('Analysis not requested')) {
            reasoningPanel.classList.remove('hidden');
            
            if (factResult.includes('API quota exceeded') || factResult.includes('UNVERIFIED')) {
                if (reasoningBadge) reasoningBadge.textContent = '⚠ Fallback Mode';
                if (reasoningBadge) reasoningBadge.style.background = 'rgba(245, 158, 11, 0.15)';
                if (reasoningBadge) reasoningBadge.style.color = 'var(--warning)';
            } else {
                if (reasoningBadge) reasoningBadge.textContent = 'Gemini 2.0 Flash';
                if (reasoningBadge) reasoningBadge.style.background = '';
                if (reasoningBadge) reasoningBadge.style.color = '';
            }
            
            let formattedReasoning = factResult
                .replace(/\[VERIFIED\]/g, '✅ VERIFIED')
                .replace(/\[FAKE\]/g, '❌ FAKE')
                .replace(/\[MISLEADING\]/g, '⚠️ MISLEADING')
                .replace(/\[UNVERIFIED\]/g, '🛡️ UNVERIFIED')
                .replace(/\[Likely Real\]/g, '✅ Likely Real')
                .replace(/\[Investigation Req\]/g, '🔍 Investigation Required');
            reasoningBody.textContent = formattedReasoning;
        } else {
            reasoningPanel.classList.add('hidden');
        }

        // ─── 6. Evidence Cards ───
        evidenceGrid.innerHTML = '';
        if (evidence.length === 0) {
            evidenceGrid.innerHTML = `
                <div class="evidence-empty">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <p>No web evidence was found for this claim.</p>
                </div>`;
        } else {
            evidence.forEach(ev => {
                const cred = ev.credibility || 'unknown';
                const credIcon = cred === 'high' ? 'fa-circle-check' : cred === 'medium' ? 'fa-circle-half-stroke' : 'fa-circle-info';
                const snippet = ev.snippet || ev.title || 'No details available.';
                const sourceName = ev.source || 'Unknown';
                const sourceUrl = ev.url || '';

                const card = document.createElement('div');
                card.className = 'evidence-card';
                card.innerHTML = `
                    <div class="evidence-card-icon cred-${cred}">
                        <i class="fa-solid ${credIcon}"></i>
                    </div>
                    <div class="evidence-card-body">
                        <div class="evidence-card-source">
                            ${sourceUrl ? `<a href="${sourceUrl}" target="_blank">${sourceName}</a>` : sourceName}
                            <i class="fa-solid fa-arrow-up-right-from-square" style="font-size: 10px; opacity: 0.5;"></i>
                        </div>
                        <div class="evidence-card-text">${snippet}</div>
                    </div>
                    <span class="evidence-card-cred ${cred}">${cred}</span>
                `;
                evidenceGrid.appendChild(card);
            });
        }

        resultsUI.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
