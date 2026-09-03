let fileQueue = [];

document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('send-btn');
    const query = document.getElementById('query-input');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const startIndexing = document.getElementById('start-indexing');
    const clearChat = document.getElementById('clear-chat');
    const chatWindow = document.getElementById('chat-window');
    const ingestionsList = document.getElementById('ingestions-list');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.background = 'rgba(255, 255, 255, 0.1)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.background = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.background = '';
    addFilesToQueue(e.dataTransfer.files);
});


dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    addFilesToQueue(e.target.files);
});

function addFilesToQueue(files) {
    const activeQueue = document.querySelector('.active-queue');
    if(files.length === 0) return;
    //IF PyMuPDF cannot load txt or docx, remove these conditions I just need to check later
    for(let file of files){
        if(file.name.endsWith('.txt') || file.name.endsWith('.pdf') || file.name.endsWith('.docx') || file.name.endsWith('.md')){
            fileQueue.push(file);

            const card = document.createElement('div');
            card.className = 'queue-card';
            card.innerHTML = `
                <div class="queue-card-info">
                    <span class="queue-card-name"></span>
                    <span class="queue-card-size">${(file.size / 1024).toFixed(1)} KB</span>
                </div>
                <button class="queue-card-remove">✕</button>
            `;
            card.querySelector('.queue-card-name').textContent = file.name;

            card.querySelector('.queue-card-remove').addEventListener('click', () => {
                fileQueue = fileQueue.filter(f => f !== file);
                card.remove();
            });

            activeQueue.appendChild(card);
        }
    }
}

// Ships the queue at /ingest one file at a time, the backend does the chunking,
// embedding and bm25 indexing per file anyway so theres no point batching.
startIndexing.addEventListener('click', async () => {
    if(fileQueue.length === 0) return;

    startIndexing.disabled = true;
    const originalText = startIndexing.textContent;

    const queue = [...fileQueue];
    for(let i = 0; i < queue.length; i++){
        const file = queue[i];
        startIndexing.textContent = `Indexing ${i + 1}/${queue.length}...`;

        const form = new FormData();
        form.append('file', file);

        try {
            const response = await fetch('/ingest', { method: 'POST', body: form });
            const result = await response.json();
            if(result.status === 'error'){
                addMessageToUI(`Could not index ${file.name}: ${result.error}`, 'bot-message');
            }
        } catch (err) {
            addMessageToUI(`Could not index ${file.name}: ${err}`, 'bot-message');
        }
    }

    fileQueue = [];
    document.querySelector('.active-queue').innerHTML = '';
    startIndexing.textContent = originalText;
    startIndexing.disabled = false;
    refreshIngestions();
});

async function refreshIngestions() {
    const response = await fetch('/ingestions');
    const files = await response.json();

    ingestionsList.innerHTML = '';
    if(files.length === 0){
        const empty = document.createElement('div');
        empty.className = 'ingestion-empty';
        empty.textContent = 'Nothing indexed yet';
        ingestionsList.appendChild(empty);
        return;
    }

    for(const file of files){
        const item = document.createElement('div');
        item.className = 'ingestion-item';

        const name = document.createElement('span');
        name.className = 'ingestion-name';
        name.textContent = file.filename;

        const count = document.createElement('span');
        count.className = 'ingestion-chunks';
        count.textContent = `${file.num_chunks} chunks`;

        item.append(name, count);
        ingestionsList.appendChild(item);
    }
}

async function sendQuery() {
    const q = query.value.trim();
    if(q === '') return;

    addMessageToUI(q, 'message');
    query.value = '';

    const pending = addMessageToUI('Searching and thinking...', 'bot-message pending');

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: q })
        });
        const data = await response.json();
        pending.remove();
        addAnswerToUI(data.answer, data.citations || []);
    } catch (err) {
        pending.remove();
        addMessageToUI(`Something went wrong: ${err}`, 'bot-message');
    }
}

sendBtn.addEventListener('click', sendQuery);
query.addEventListener('keydown', (e) => {
    if(e.key === 'Enter') sendQuery();
});

clearChat.addEventListener('click', () => {
    chatWindow.innerHTML = '';
});

// The model is told to cite with [n] markers that line up with the order of the
// chunks we sent it, so we can turn them into chips pointing at the source cards.
function addAnswerToUI(answer, citations) {
    const wrapper = document.createElement('div');
    wrapper.className = 'bot-message';

    const body = document.createElement('div');
    body.className = 'answer-body';
    const byMarker = new Map(citations.map(c => [c.marker, c]));

    for(const part of answer.split(/(\[\d+\])/g)){
        const match = part.match(/^\[(\d+)\]$/);
        if(match && byMarker.has(Number(match[1]))){
            const chip = document.createElement('span');
            chip.className = 'citation-marker';
            chip.textContent = match[1];
            chip.title = sourceLabel(byMarker.get(Number(match[1])));
            chip.addEventListener('click', () => highlightSource(wrapper, Number(match[1])));
            body.appendChild(chip);
        } else {
            body.appendChild(document.createTextNode(part));
        }
    }
    wrapper.appendChild(body);

    if(citations.length > 0){
        wrapper.appendChild(buildSources(citations));
    }

    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return wrapper;
}

function sourceLabel(citation) {
    const page = citation.page === null ? '' : ` p.${citation.page + 1}`;
    return `${citation.source}${page}`;
}

function buildSources(citations) {
    const details = document.createElement('details');
    details.className = 'sources';

    const summary = document.createElement('summary');
    summary.textContent = `${citations.length} source${citations.length === 1 ? '' : 's'} used`;
    details.appendChild(summary);

    for(const citation of citations){
        const card = document.createElement('div');
        card.className = 'source-card';
        card.dataset.marker = citation.marker;

        const head = document.createElement('div');
        head.className = 'source-head';

        const label = document.createElement('span');
        label.textContent = `[${citation.marker}] ${sourceLabel(citation)}`;

        // the reranker score, 1.0 means the cross encoder was sure this chunk answers it
        const score = document.createElement('span');
        score.className = 'source-score';
        score.textContent = citation.score.toFixed(3);

        head.append(label, score);

        const snippet = document.createElement('div');
        snippet.className = 'source-snippet';
        snippet.textContent = citation.snippet;

        card.append(head, snippet);
        details.appendChild(card);
    }

    return details;
}

function highlightSource(wrapper, marker) {
    const details = wrapper.querySelector('.sources');
    if(details === null) return;

    details.open = true;
    const card = details.querySelector(`.source-card[data-marker="${marker}"]`);
    if(card === null) return;

    card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1200);
}

function addMessageToUI(message, className) {
    const messageElem = document.createElement('div');
    messageElem.className = className;
    messageElem.textContent = message;
    chatWindow.appendChild(messageElem);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return messageElem;
}

refreshIngestions();
});
