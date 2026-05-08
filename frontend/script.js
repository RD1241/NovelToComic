const MIN_CHARS = 20;
const MAX_CHARS = 3000;

// --- Character counter ---
const novelInput  = document.getElementById('novel-input');
const charCounter = document.getElementById('char-counter');
novelInput.addEventListener('input', () => {
    const len = novelInput.value.length;
    charCounter.textContent = `${len} / ${MAX_CHARS}`;
    charCounter.style.color =
        len < MIN_CHARS         ? '#ef4444' :
        len > MAX_CHARS * 0.9   ? '#f59e0b' : '#a78bfa';
});

// --- Helpers ---
function showError(msg) {
    document.getElementById('error-text').textContent = msg;
    document.getElementById('error-banner').classList.remove('hidden');
}

function setStep(stepId) {
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    const el = document.getElementById(stepId);
    if (el) el.classList.add('active');
}

/**
 * Build dynamic panel step spans inside #progress-steps.
 * Called once we know total panel count from the first Drawing progress message.
 */
let panelStepsBuilt = false;
function buildPanelSteps(totalPanels) {
    if (panelStepsBuilt) return;
    panelStepsBuilt = true;

    const container   = document.getElementById('progress-steps');
    const assembleEl  = document.getElementById('step-assemble');

    for (let i = 1; i <= totalPanels; i++) {
        const span = document.createElement('span');
        span.id        = `step-panel${i}`;
        span.className = 'step';
        span.textContent = `🎨 Panel ${i}`;
        // Insert before the Assembling step
        container.insertBefore(span, assembleEl);
    }
}

// Matches "Drawing panel 2/5..." and returns {current, total}
function parsePanelProgress(progress) {
    const match = progress.match(/panel\s+(\d+)\/(\d+)/i);
    if (match) return { current: parseInt(match[1]), total: parseInt(match[2]) };
    return null;
}

// --- Main generate handler ---
document.getElementById('generate-btn').addEventListener('click', async () => {
    const text  = novelInput.value.trim();
    const style = document.getElementById('style-select').value;  // "" means auto-detect

    // Client-side validation
    if (text.length < MIN_CHARS) {
        showError(`Text too short. Please enter at least ${MIN_CHARS} characters.`);
        return;
    }
    if (text.length > MAX_CHARS) {
        showError(`Text too long. Max ${MAX_CHARS} characters allowed.`);
        return;
    }

    // Reset dynamic panel steps for a fresh run
    panelStepsBuilt = false;
    // Remove any previously injected panel steps
    document.querySelectorAll('[id^="step-panel"]').forEach(el => el.remove());

    document.getElementById('error-banner').classList.add('hidden');
    document.getElementById('generate-btn').disabled  = true;
    document.getElementById('generate-btn').innerText = 'Generating...';
    document.getElementById('results-state').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');

    const loadingText = document.getElementById('loading-text');
    loadingText.innerText = 'Submitting to AI pipeline...';
    setStep('step-llm');

    try {
        // 1. Submit job — include style (empty string = auto-detect on server)
        const response = await fetch('/api/generate_comic', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text, style }),
        });

        const data = await response.json();
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message || data.detail || 'Server error');
        }

        const jobId = data.job_id || (data.data && data.data.job_id);
        if (!jobId) throw new Error('Server did not return a job ID.');

        // 2. Poll for status
        let resultData   = null;
        let lastProgress = '';

        while (true) {
            const statusRes  = await fetch(`/api/status/${jobId}`);
            const statusData = await statusRes.json();

            if (statusData.status === 'completed') {
                resultData = statusData.result;
                if (typeof resultData === 'string') resultData = JSON.parse(resultData);
                break;
            } else if (statusData.status === 'failed') {
                throw new Error(statusData.error || 'Generation failed.');
            }

            // Update progress display
            const progress = statusData.progress || '';
            if (progress !== lastProgress) {
                lastProgress           = progress;
                loadingText.innerText  = progress || 'AI is working...';

                if (progress.includes('Extracting')) {
                    setStep('step-llm');
                } else if (progress.toLowerCase().includes('drawing panel')) {
                    // Parse "Drawing panel N/M..." dynamically
                    const parsed = parsePanelProgress(progress);
                    if (parsed) {
                        buildPanelSteps(parsed.total);            // build steps on first panel message
                        setStep(`step-panel${parsed.current}`);
                    }
                } else if (progress.includes('Assembling')) {
                    setStep('step-assemble');
                }
            }

            await new Promise(r => setTimeout(r, 3000));
        }

        // 3. Render results
        setStep('step-assemble');
        document.getElementById('final-comic-img').src =
            resultData.final_page + '?t=' + Date.now();

        const panelsGrid = document.getElementById('panels-grid');
        panelsGrid.innerHTML = '';
        resultData.panels.forEach(panelPath => {
            const img = document.createElement('img');
            img.src   = panelPath + '?t=' + Date.now();
            img.alt   = 'Comic Panel';
            panelsGrid.appendChild(img);
        });

        document.getElementById('loading-state').classList.add('hidden');
        document.getElementById('results-state').classList.remove('hidden');

    } catch (error) {
        showError(error.message);
        document.getElementById('loading-state').classList.add('hidden');
    } finally {
        document.getElementById('generate-btn').disabled  = false;
        document.getElementById('generate-btn').innerText = 'Generate Comic';
    }
});
