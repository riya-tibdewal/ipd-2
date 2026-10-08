const video = document.querySelector('#video');
const run = document.querySelector('#run');
const message = document.querySelector('#message');
let sessionId;

video.addEventListener('change', () => {
  const file = video.files[0];
  document.querySelector('#file-name').textContent = file ? file.name : 'Choose a video file';
  run.disabled = !file;
});

run.addEventListener('click', async () => {
  const body = new FormData();
  body.append('video', video.files[0]);
  run.disabled = true;
  message.textContent = 'Uploading video…';
  try {
    const created = await fetch('/api/sessions', { method: 'POST', body }).then(check);
    sessionId = created.id;
    await fetch(`/api/sessions/${sessionId}/start`, { method: 'POST' }).then(check);
    document.querySelector('#progress-panel').classList.remove('hidden');
    document.querySelector('#results-panel').classList.add('hidden');
    message.textContent = '';
    poll();
  } catch (error) {
    message.textContent = error.message;
    run.disabled = false;
  }
});

async function poll() {
  try {
    const session = await fetch(`/api/sessions/${sessionId}`).then(check);
    document.querySelector('#status').textContent = session.status;
    document.querySelector('#progress').style.width = `${session.progress || 0}%`;
    document.querySelector('#progress-copy').textContent = `${session.processed_frames} / ${session.total_frames || '?'} frames`;
    if (session.status === 'completed') return showResults(session);
    if (session.status === 'failed') throw new Error(session.error || 'Analysis failed.');
    setTimeout(poll, 1200);
  } catch (error) {
    message.textContent = error.message;
    run.disabled = false;
  }
}

function showResults(session) {
  const player = document.querySelector('#annotated-player');
  if (session.annotated_video_url) {
    player.src = session.annotated_video_url;
    player.load();
  }

  const body = document.querySelector('#results');
  body.innerHTML = '';
  session.results.forEach(item => {
    let breakdown = [];
    if (item.label_counts) {
      for (const [lbl, cnt] of Object.entries(item.label_counts)) {
        breakdown.push(`${lbl}: ${cnt}`);
      }
    }
    const breakdownText = breakdown.length ? breakdown.join(', ') : `${item.observations} frames analyzed`;

    body.insertAdjacentHTML('beforeend', `
      <tr>
        <td><strong>${item.name}</strong></td>
        <td><span class="tag">${item.final_label}</span></td>
        <td>${Math.round(item.final_confidence * 100)}%</td>
        <td><strong>${item.average_attention}%</strong></td>
        <td><small style="color:var(--muted);">${breakdownText}</small></td>
      </tr>
    `);
  });

  const report = document.querySelector('#report');
  report.href = session.report_url;
  report.hidden = false;
  document.querySelector('#results-panel').classList.remove('hidden');
}

async function check(response) {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || 'Request failed.');
  }
  return response.json();
}

