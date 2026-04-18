document.addEventListener('DOMContentLoaded', () => {
    // Nav/Tabs
    const tabText = document.getElementById('tab-text');
    const tabUrl = document.getElementById('tab-url');
    const textGroup = document.getElementById('text-input-group');
    const urlGroup = document.getElementById('url-input-group');
    
    // UI Panels
    const form = document.getElementById('analyze-form');
    const loadingUI = document.getElementById('loading');
    const resultsUI = document.getElementById('results');
    const errorUI = document.getElementById('error-container');
    const errorMsg = document.getElementById('error-message');
    const submitBtn = document.getElementById('submit-btn');

    // Result Nodes
    const verdictBadge = document.getElementById('verdict-badge');
    const confBar = document.getElementById('conf-bar');
    const confText = document.getElementById('conf-text');
    const evidenceList = document.getElementById('evidence-list');
    const evidenceCount = document.getElementById('evidence-count');

    let activeTab = 'text';

    // Interactions
    tabText.addEventListener('click', () => {
        activeTab = 'text';
        tabText.classList.add('active');
        tabUrl.classList.remove('active');
        textGroup.classList.remove('hidden');
        urlGroup.classList.add('hidden');
    });

    tabUrl.addEventListener('click', () => {
        activeTab = 'url';
        tabUrl.classList.add('active');
        tabText.classList.remove('active');
        urlGroup.classList.remove('hidden');
        textGroup.classList.add('hidden');
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Payload
        let payload = {};
        if (activeTab === 'text') {
            const text = document.getElementById('text').value;
            if (!text.trim()) return;
            payload = { text };
        } else {
            const url = document.getElementById('url').value;
            if (!url.trim()) return;
            // The HTML prefix has 'https://' as a label, we prepend if missing
            payload = { url: url.startsWith('http') ? url : `https://${url}` };
        }

        // Reset
        resultsUI.classList.add('hidden');
        errorUI.classList.add('hidden');
        loadingUI.classList.remove('hidden');
        submitBtn.disabled = true;

        startLogAnimation();

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Network response was not ok');
            }

            renderDashboard(data);

        } catch (err) {
            errorMsg.textContent = err.message;
            errorUI.classList.remove('hidden');
            loadingUI.classList.add('hidden');
        } finally {
            submitBtn.disabled = false;
        }
    });

    function startLogAnimation() {
        const lines = document.querySelectorAll('.log-line');
        lines.forEach(l => {
            l.style.opacity = '0';
            l.classList.remove('pending');
        });
        
        let i = 0;
        function showLine() {
            if (i < lines.length) {
                lines[i].style.opacity = '1';
                if (i === lines.length - 1) {
                    lines[i].classList.add('pending');
                }
                i++;
                setTimeout(showLine, 400 + Math.random() * 500);
            }
        }
        showLine();
    }

    function renderDashboard(data) {
        loadingUI.classList.add('hidden');
        resultsUI.classList.remove('hidden');

        // Verdict
        let vClass = 'is-warn';
        if (data.prediction.includes('Real')) vClass = 'is-real';
        if (data.prediction.includes('Fake')) vClass = 'is-fake';

        verdictBadge.className = `verdict-banner ${vClass}`;
        verdictBadge.innerHTML = `<i class="fa-solid fa-flag-checkered"></i> ${data.prediction} &mdash; ${data.decision}`;

        // Confidence
        const numVal = parseFloat(data.confidence) || 0;
        confText.textContent = `${Math.round(numVal)}%`;
        
        setTimeout(() => {
            confBar.style.width = `${numVal}%`;
            if (numVal > 75) confBar.style.backgroundColor = 'var(--status-success)';
            else if (numVal > 40) confBar.style.backgroundColor = 'var(--status-warning)';
            else confBar.style.backgroundColor = 'var(--status-error)';
        }, 100);

        // Sources Table
        evidenceList.innerHTML = '';
        const items = data.evidence || [];
        evidenceCount.textContent = `${items.length} Sources`;

        if (items.length === 0) {
            evidenceList.innerHTML = `<tr><td colspan="3" style="text-align: center; font-style: italic;">No external cross-references found.</td></tr>`;
        } else {
            items.forEach(ev => {
                const cId = ev.credibility;
                const cLabel = cId === 'high' ? 'High Authority' : cId === 'medium' ? 'Medium Auth' : 'Unknown / Low';
                const cClass = `cred-${cId}`;
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${ev.source}</td>
                    <td>${ev.title.length > 50 ? ev.title.substring(0,47)+'...' : ev.title}</td>
                    <td><span class="cred-indicator ${cClass}">${cLabel}</span></td>
                `;
                evidenceList.appendChild(tr);
            });
        }
    }
});
