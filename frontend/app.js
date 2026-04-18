document.addEventListener('DOMContentLoaded', () => {
    // Nav/Tabs
    const tabText = document.getElementById('tab-text');
    const tabUrl = document.getElementById('tab-url');
    const tabDoc = document.getElementById('tab-doc');
    const textGroup = document.getElementById('text-input-group');
    const urlGroup = document.getElementById('url-input-group');
    const docGroup = document.getElementById('doc-input-group');
    
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

    // History Nodes
    const historyLink = document.getElementById('history-link');
    const historyPanel = document.getElementById('history-panel');
    const closeHistoryBtn = document.getElementById('close-history');
    const clearHistoryBtn = document.getElementById('clear-history');
    const historyList = document.getElementById('history-list');

    // Export PDF
    const exportPdfBtn = document.getElementById('export-pdf-btn');
    exportPdfBtn.addEventListener('click', () => {
        window.print();
    });

    let activeTab = 'text';

    // Interactions
    function resetTabs() {
        tabText.classList.remove('active');
        tabUrl.classList.remove('active');
        tabDoc.classList.remove('active');
        textGroup.classList.add('hidden');
        urlGroup.classList.add('hidden');
        docGroup.classList.add('hidden');
    }

    tabText.addEventListener('click', () => {
        resetTabs(); activeTab = 'text';
        tabText.classList.add('active'); textGroup.classList.remove('hidden');
    });

    tabUrl.addEventListener('click', () => {
        resetTabs(); activeTab = 'url';
        tabUrl.classList.add('active'); urlGroup.classList.remove('hidden');
    });

    tabDoc.addEventListener('click', () => {
        resetTabs(); activeTab = 'doc';
        tabDoc.classList.add('active'); docGroup.classList.remove('hidden');
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let endpoint = '/api/analyze';
        let bodyPayload = null;
        let isFormData = false;
        let historyTitle = "";

        if (activeTab === 'text') {
            const text = document.getElementById('text').value;
            if (!text.trim()) return;
            bodyPayload = JSON.stringify({ text });
            historyTitle = text.substring(0, 40) + "...";
        } else if (activeTab === 'url') {
            let url = document.getElementById('url').value;
            if (!url.trim()) return;
            url = url.startsWith('http') ? url : `https://${url}`;
            bodyPayload = JSON.stringify({ url });
            historyTitle = url;
        } else {
            // PDF Document
            const fileInput = document.getElementById('file');
            if (fileInput.files.length === 0) return;
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append("file", file);
            bodyPayload = formData;
            isFormData = true;
            endpoint = '/api/analyze-file';
            historyTitle = "Document: " + file.name;
        }

        // Reset
        resultsUI.classList.add('hidden');
        errorUI.classList.add('hidden');
        loadingUI.classList.remove('hidden');
        submitBtn.disabled = true;

        startLogAnimation();

        try {
            const headers = {};
            if (!isFormData) headers['Content-Type'] = 'application/json';

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: bodyPayload
            });
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || data.detail || 'Network response was not ok');
            }

            renderDashboard(data);
            saveToHistory(historyTitle, data.prediction);

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

    // --- History Logic ---

    function loadHistory() {
        const hist = JSON.parse(localStorage.getItem('truthguardHistory') || '[]');
        historyList.innerHTML = '';
        if(hist.length === 0) {
            historyList.innerHTML = '<li style="color:var(--text-muted);font-size:12px;">No past analyses.</li>';
        } else {
            hist.forEach(item => {
                const li = document.createElement('li');
                const vColor = item.prediction.includes('Real') ? 'var(--status-success)' : item.prediction.includes('Fake') ? 'var(--status-error)' : 'var(--status-warning)';
                li.innerHTML = `
                    <div style="font-weight:500;margin-bottom:4px;word-break:break-all;">${item.title}</div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;">
                        <span style="color:${vColor}">${item.prediction}</span>
                        <span style="color:var(--text-muted)">${new Date(item.date).toLocaleTimeString()}</span>
                    </div>
                `;
                historyList.appendChild(li);
            });
        }
    }

    function saveToHistory(title, prediction) {
        const hist = JSON.parse(localStorage.getItem('truthguardHistory') || '[]');
        hist.unshift({ title, prediction, date: new Date().toISOString() });
        if(hist.length > 10) hist.pop(); // keep last 10
        localStorage.setItem('truthguardHistory', JSON.stringify(hist));
        loadHistory();
    }

    historyLink.addEventListener('click', (e) => {
        e.preventDefault();
        historyPanel.classList.remove('hidden');
        loadHistory();
    });

    closeHistoryBtn.addEventListener('click', () => {
        historyPanel.classList.add('hidden');
    });

    clearHistoryBtn.addEventListener('click', () => {
        localStorage.removeItem('truthguardHistory');
        loadHistory();
    });

    // Init history on load
    loadHistory();
});
