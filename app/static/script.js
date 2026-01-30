let fileQueue = [];

document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('send-btn');
    const query = document.getElementById('query-input');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

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
    console.log('File input changed:', e.target.files);
    addFilesToQueue(e.target.files);
});

function addFilesToQueue(files) {
    console.log('addFilesToQueue called with:', files);
    const activeQueue = document.querySelector('.active-queue');
    console.log('activeQueue element:', activeQueue);
    if(files.length === 0) return;
    //IF PyMuPDF cannot load txt or docx, remove these conditions I just need to check later
    for(let file of files){
        if(file.name.endsWith('.txt') || file.name.endsWith('.pdf') || file.name.endsWith('.docx')){
            fileQueue.push(file);

            const card = document.createElement('div');
            card.className = 'queue-card';
            card.innerHTML = `
                <div class="queue-card-info">
                    <span class="queue-card-name">${file.name}</span>
                    <span class="queue-card-size">${(file.size / 1024).toFixed(1)} KB</span>
                </div>
                <button class="queue-card-remove">✕</button>
            `;

            card.querySelector('.queue-card-remove').addEventListener('click', () => {
                fileQueue = fileQueue.filter(f => f !== file);
                card.remove();
            });

            activeQueue.appendChild(card);
        }
    }


}
sendBtn.addEventListener('click', () => {
    const q = query.value;
    addMessageToUI(q, 'message');
    query.value = '';

    const send = fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: q }) 

    })
.then(response => response.json())
.then(data => {
    addMessageToUI(data.answer, 'bot-message');

        });
    });

function addMessageToUI(message, className) {
        const chatBox = document.getElementById('chat-box');
        const messageElem = document.createElement('div');
        messageElem.className = className;
        messageElem.textContent = message;
        chatBox.appendChild(messageElem);
    }
});


