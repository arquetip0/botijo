document.addEventListener('click', async (e) => {
  const el = e.target.closest('.btn-hotzone');
  if (!el) return;
  const action = el.dataset.action;
  if (action !== 'ctrlc') return;

  e.preventDefault();
  panelLogLine('⛔ Enviando Ctrl-C al proceso activo...');

  try {
    const res = await fetch('http://localhost:5000/interrupt', { method: 'POST' });
    const text = await res.text();
    if (res.ok) {
      panelLogLine(`✅ ${text}`);
    } else {
      panelLogLine(`⚠️ ${text}`);
    }
  } catch (err) {
    panelLogLine(`❌ Error al interrumpir: ${err}`);
  }
});
function appendLine(html) {
    const v = document.getElementById('content-window');
    v.innerHTML += `<div style="margin:0 10px; white-space:pre-wrap;">${html}</div>`;
    v.scrollTop = v.scrollHeight;
}

function prepConsole(bg = true) {
    const v = document.getElementById('content-window');
    if (bg) {
        v.style.backgroundColor = '#000';
        v.style.color = '#0f0';
        v.style.fontFamily = 'monospace';
        v.style.fontSize = '14px';
        v.style.overflowY = 'auto';
    }
    return v;
}

function runScript(scriptName) {
    console.log(`Ejecutando: ${scriptName}`);
    const ventana = prepConsole(true);
    ventana.innerHTML = `<div style="margin:10px;">Ejecutando ${scriptName}...</div>`;

    fetch(`http://localhost:5000/run_script?name=${encodeURIComponent(scriptName)}`)
      .then(response => {
        if (!response.body) { appendLine(`<span style="color:red">Sin salida</span>`); return; }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        (function read() {
          reader.read().then(({done, value}) => {
            if (done) return;
            const chunk = decoder.decode(value, {stream:true});
            appendLine(chunk);
            read();
          });
        })();
      })
      .catch(err => appendLine(`<span style="color:red">Error backend: ${err}</span>`));
}

function sendCtrlC() {
    prepConsole(false);
    appendLine('→ Enviando Ctrl-C al proceso actual…');
    fetch('http://localhost:5000/interrupt', {method:'POST'})
      .then(r => r.text()).then(t => appendLine(t))
      .catch(e => appendLine(`<span style="color:red">Error: ${e}</span>`));
}

function closeChromium() {
    prepConsole(false);
    appendLine('→ Cerrando Chromium…');
    fetch('http://localhost:5000/close_chromium', {method:'POST'})
      .then(r => r.text()).then(t => appendLine(t))
      .catch(e => appendLine(`<span style="color:red">Error: ${e}</span>`));
}

function initPanel() {
    const botones = document.querySelectorAll('.btn-hotzone');
    botones.forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action || 'run';
            if (action === 'ctrlc') return sendCtrlC();
            if (action === 'close-chromium') return closeChromium();
            const scriptName = btn.dataset.script;
            runScript(scriptName);
        });
    });
}
document.addEventListener('DOMContentLoaded', initPanel);
function panelLogLine(line) {
  const win = document.getElementById('content-window');
  if (!win) return;
  const p = document.createElement('p');
  p.textContent = line;
  win.appendChild(p);
  win.scrollTop = win.scrollHeight;
}