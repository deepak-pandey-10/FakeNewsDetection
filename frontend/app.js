document.addEventListener('DOMContentLoaded', () => {
    // UI Panels
    const textInput = document.getElementById('text');
    const factBtn = document.getElementById('fact-btn');
    const newsBtn = document.getElementById('news-btn');
    
    const loadingUI = document.getElementById('loading');
    const resultsUI = document.getElementById('results');
    const errorUI = document.getElementById('error-container');
    const errorMsg = document.getElementById('error-message');

    // Result Nodes
    const newsBadge = document.getElementById('news-verdict-badge');
    const factBadge = document.getElementById('fact-verdict-badge');
    const confBar = document.getElementById('conf-bar');
    const confText = document.getElementById('conf-text');
    const evidenceList = document.getElementById('evidence-list');
    const evidenceCount = document.getElementById('evidence-count');

    // Click Handlers
    factBtn.addEventListener('click', () => triggerAnalysis('fact'));
    newsBtn.addEventListener('click', () => triggerAnalysis('news'));

    async function triggerAnalysis(mode) {
        const textValue = textInput.value;
        if (!textValue.trim()) {
            alert("Please provide content to analyze.");
            return;
        }

        // Reset UI
        resultsUI.classList.add('hidden');
        errorUI.classList.add('hidden');
        loadingUI.classList.remove('hidden');
        factBtn.disabled = true;
        newsBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textValue, mode: mode })
            });
            const data = await response.json();

            if (!response.ok || data.error) throw new Error(data.error || 'Request failed');

            renderDashboard(data, mode);
        } catch (err) {
            errorMsg.textContent = err.message;
            errorUI.classList.remove('hidden');
            loadingUI.classList.add('hidden');
        } finally {
            factBtn.disabled = false;
            newsBtn.disabled = false;
        }
    }

    function renderDashboard(data, mode) {
        loadingUI.classList.add('hidden');
        resultsUI.classList.remove('hidden');

        // 1. News Verification (Model)
        const newsData = data.news_verification || { prediction: "N/A", confidence: "0" };
        const prediction = newsData.prediction;
        let nClass = prediction === 'Real' ? 'is-real' : prediction === 'Fake' ? 'is-fake' : 'is-warn';
        newsBadge.className = `verdict-banner ${nClass}`;
        newsBadge.innerHTML = `<i class="fa-solid fa-microchip"></i> ${prediction}`;

        // 2. Fact Verification (Agent)
        const factData = data.fact_verification || { agent_verdict: "N/A", evidence: [] };
        const factResult = factData.agent_verdict;
        let fClass = factResult.includes('VERIFIED') || factResult.includes('credible') || factResult.includes('Real') ? 'is-real' : factResult.includes('Analysis not requested') ? 'is-warn' : 'is-fake';
        factBadge.className = `verdict-banner ${fClass}`;
        factBadge.innerHTML = `<i class="fa-solid fa-robot"></i> Audit: ${factResult}`;

        // 3. Confidence Meter
        const numVal = parseFloat(newsData.confidence) || 0;
        confText.textContent = `${numVal}%`;
        confBar.style.width = `${numVal}%`;

        // 4. Evidence Table
        evidenceList.innerHTML = '';
        const items = data.fact_verification.evidence || [];
        evidenceCount.textContent = `${items.length} Sources`;

        if (items.length === 0) {
            evidenceList.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No live corroboration performed.</td></tr>`;
        } else {
            items.forEach(ev => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${ev.source}</td>
                    <td>${ev.title.length > 50 ? ev.title.substring(0,47)+'...' : ev.title}</td>
                    <td><span class="cred-indicator cred-standard">Standard</span></td>
                `;
                evidenceList.appendChild(tr);
            });
        }
    }
});
