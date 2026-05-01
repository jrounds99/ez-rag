// Chat UI logic. Edit freely.
// All settings (model, retrieval, modifiers) come from the bundled
// data/config.toml — the server applies them. The page just talks to /api.

const messagesEl = document.getElementById("messages");
const inputEl    = document.getElementById("input");
const sendBtn    = document.getElementById("send");
const clearBtn   = document.getElementById("clear-btn");
const statusDot  = document.getElementById("status");
const statusText = document.getElementById("status-text");

let history = [];          // [{role, content}, ...]
let busy    = false;

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;

  const roleEl = document.createElement("span");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "Assistant";

  const bodyEl = document.createElement("span");
  bodyEl.className = "body";
  bodyEl.textContent = text;

  div.appendChild(roleEl);
  div.appendChild(bodyEl);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return { container: div, body: bodyEl };
}

function renderCitations(container, citations) {
  if (!citations || citations.length === 0) return;
  const row = document.createElement("div");
  row.className = "citations";
  citations.forEach((c, i) => {
    const chip = document.createElement("button");
    chip.className = "citation";
    chip.type = "button";
    chip.textContent = `[${i + 1}] ${shortName(c.path)}` + (c.page ? ` p.${c.page}` : "");
    chip.title = c.text || "click to view source";
    chip.addEventListener("click", () => openSource(i + 1, c));
    row.appendChild(chip);
  });
  container.appendChild(row);
}

function shortName(path) {
  if (!path) return "(unknown)";
  const parts = path.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path;
}

// ----- source-preview modal -----
// Shown when the user clicks a citation chip. For PDFs we fetch a rendered
// page image; for HTML/MD/TXT we fetch the file and show its raw content.
// Falls back to "source not bundled" message when the export was lean.
let serverHasSources = false;

const sourceModal = document.getElementById("source-modal");
const sourceTitle = document.getElementById("source-title");
const sourcePath  = document.getElementById("source-path");
const sourceBody  = document.getElementById("source-body");
const sourceClose = document.getElementById("source-close");
sourceClose.addEventListener("click", () => sourceModal.classList.remove("open"));
sourceModal.addEventListener("click", (e) => {
  if (e.target === sourceModal) sourceModal.classList.remove("open");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") sourceModal.classList.remove("open");
});

function isPdfPath(p) { return /\.pdf$/i.test(p || ""); }
function isImagePath(p) { return /\.(png|jpe?g|webp|gif|bmp)$/i.test(p || ""); }

async function openSource(idx, c) {
  sourceTitle.textContent = `[${idx}] ${shortName(c.path)}` + (c.page ? ` — page ${c.page}` : "");
  sourcePath.textContent = c.path + (c.section ? `  ·  ${c.section}` : "");
  sourceBody.innerHTML = "";
  sourceModal.classList.add("open");

  // Always show the chunk text — that's free regardless of bundle mode.
  if (c.text) {
    const ex = document.createElement("div");
    ex.className = "excerpt";
    ex.textContent = c.text;
    sourceBody.appendChild(document.createTextNode("Retrieved passage:"));
    sourceBody.appendChild(ex);
  }

  if (!serverHasSources) {
    const note = document.createElement("p");
    note.className = "muted";
    note.innerHTML = "<em>Source files weren't bundled with this chatbot. " +
      "Re-export from ez-rag with the <strong>Include source files</strong> " +
      "checkbox to see the original page or screenshot here.</em>";
    sourceBody.appendChild(note);
    return;
  }

  const path = c.path || "";
  const params = new URLSearchParams({ path });

  if (isPdfPath(path) && c.page) {
    // Render the cited page as an image.
    params.set("page", String(c.page));
    const url = `/api/page-image?${params.toString()}`;
    const dlName = `${shortName(path).replace(/\.pdf$/i, "")}-p${c.page}.png`;
    sourceBody.appendChild(makeImageViewer(url, `${path} page ${c.page}`, dlName));
  } else if (isImagePath(path)) {
    const url = `/api/source?${params.toString()}`;
    sourceBody.appendChild(makeImageViewer(url, path, shortName(path)));
  } else {
    // Text / HTML / MD — fetch and show first ~3 KB inline.
    try {
      const r = await fetch(`/api/source?${params.toString()}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const text = await r.text();
      const pre = document.createElement("pre");
      pre.className = "source-text";
      pre.textContent = text.length > 4000
        ? text.slice(0, 4000) + "\n\n…(truncated)"
        : text;
      sourceBody.appendChild(pre);
    } catch (err) {
      sourceBody.appendChild(makeError(`Couldn't load source: ${err.message}`));
    }
  }
}

function makeError(msg) {
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = msg;
  return div;
}

/**
 * Zoomable image viewer with a toolbar:
 *   [-] [reset] [+]   1.0x   [download]
 *
 * Click-and-drag to pan when zoomed in. Mouse wheel zooms toward the
 * cursor. Keyboard (+ / - / 0) works once the modal has focus.
 */
function makeImageViewer(src, alt, downloadName) {
  const wrap = document.createElement("div");
  wrap.className = "image-viewer";

  const stage = document.createElement("div");
  stage.className = "viewer-stage";

  // Wrapper handles the JS transform; the inner <img> keeps display:block
  // so we don't fight CSS centering. The wrapper itself is centered in
  // the stage via flex.
  const wrap2 = document.createElement("div");
  wrap2.className = "viewer-img-wrap";

  const img = document.createElement("img");
  img.alt = alt;
  img.draggable = false;
  img.className = "viewer-img";
  img.src = src;
  img.onerror = () => {
    stage.replaceWith(makeError(
      "Couldn't load this image. If it's a PDF, ensure pypdfium2 is " +
      "installed (`pip install pypdfium2`)."
    ));
  };
  wrap2.appendChild(img);

  // pan/zoom state
  let scale = 1.0, tx = 0, ty = 0;
  let dragging = false, lastX = 0, lastY = 0;

  function applyTransform() {
    wrap2.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    label.textContent = `${scale.toFixed(2)}x`;
  }

  function setScale(next, originX = null, originY = null) {
    next = Math.max(0.25, Math.min(8.0, next));
    if (originX !== null && originY !== null) {
      // Zoom toward the cursor: shift translation so the point under
      // the mouse stays under the mouse.
      const factor = next / scale;
      tx = (tx - originX) * factor + originX;
      ty = (ty - originY) * factor + originY;
    }
    scale = next;
    applyTransform();
  }

  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setScale(scale * factor, x, y);
  }, { passive: false });

  stage.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    stage.classList.add("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    tx += e.clientX - lastX;
    ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    applyTransform();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    stage.classList.remove("dragging");
  });

  // double-click to reset
  stage.addEventListener("dblclick", () => {
    scale = 1.0; tx = 0; ty = 0;
    applyTransform();
  });

  stage.appendChild(wrap2);

  // toolbar
  const tools = document.createElement("div");
  tools.className = "viewer-tools";

  const btn = (label, title, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "viewer-btn";
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", fn);
    return b;
  };

  const label = document.createElement("span");
  label.className = "viewer-scale";
  label.textContent = "1.00x";

  tools.appendChild(btn("−", "Zoom out (or scroll down)",
    () => setScale(scale / 1.25)));
  tools.appendChild(btn("Reset", "Reset zoom & pan (or double-click image)",
    () => { scale = 1.0; tx = 0; ty = 0; applyTransform(); }));
  tools.appendChild(btn("+", "Zoom in (or scroll up)",
    () => setScale(scale * 1.25)));
  tools.appendChild(label);
  tools.appendChild(btn("Fit width", "Fit the image to the modal width",
    () => {
      // Reset, then auto-scale so the natural width matches stage width.
      tx = 0; ty = 0;
      const stageW = stage.clientWidth - 16;
      const iw = img.naturalWidth || stageW;
      setScale(iw > 0 ? stageW / iw : 1.0);
    }));

  // Download — fetch the image as a blob and trigger a save
  const dlBtn = btn("⤓ Download", "Save this image to disk", async () => {
    try {
      const r = await fetch(src);
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = downloadName || alt.replace(/[^A-Za-z0-9._-]+/g, "_") + ".png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } catch (e) {
      alert("Download failed: " + e.message);
    }
  });
  dlBtn.classList.add("primary");
  tools.appendChild(dlBtn);

  wrap.appendChild(tools);
  wrap.appendChild(stage);
  return wrap;
}

function renderThinking(container, thinking) {
  if (!thinking || !thinking.trim()) return;
  const det = document.createElement("details");
  det.className = "thinking";
  const sum = document.createElement("summary");
  sum.textContent = "Reasoning";
  det.appendChild(sum);
  const pre = document.createElement("div");
  pre.textContent = thinking;
  det.appendChild(pre);
  container.appendChild(det);
}

async function setStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    serverHasSources = !!j.has_sources;
    const sourcesNote = serverHasSources ? "  ·  sources bundled" : "";
    statusText.textContent =
      `${j.backend} · ${j.model} · ${j.files} files / ${j.chunks} chunks${sourcesNote}`;
    statusDot.className = "dot " + (j.backend === "none" ? "err" : "ok");
  } catch {
    statusText.textContent = "server unreachable";
    statusDot.className = "dot err";
  }
}

async function send() {
  if (busy) return;
  const q = inputEl.value.trim();
  if (!q) return;

  busy = true;
  sendBtn.disabled = true;
  inputEl.value = "";

  appendMessage("user", q);
  const turn = appendMessage("assistant", "");
  turn.body.classList.add("streaming");
  history.push({ role: "user", content: q });

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history }),
    });

    if (!resp.ok) {
      turn.body.textContent = `Error: ${resp.status} ${resp.statusText}`;
      turn.body.classList.remove("streaming");
      busy = false;
      sendBtn.disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", answer = "", thinking = "";
    let citations = null;
    let thinkingEl = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });

      // ndjson — one event per line.
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        if (ev.kind === "content") {
          answer += ev.text;
          turn.body.textContent = answer;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.kind === "thinking") {
          thinking += ev.text;
          if (!thinkingEl) {
            renderThinking(turn.container, thinking);
            thinkingEl = turn.container.querySelector(".thinking div");
          } else {
            thinkingEl.textContent = thinking;
          }
        } else if (ev.kind === "citations") {
          citations = ev.items;
        } else if (ev.kind === "error") {
          answer += "\n\n[error] " + ev.text;
          turn.body.textContent = answer;
        }
      }
    }

    turn.body.classList.remove("streaming");
    if (citations) renderCitations(turn.container, citations);
    history.push({ role: "assistant", content: answer });
  } catch (err) {
    turn.body.textContent = "Network error: " + err.message;
    turn.body.classList.remove("streaming");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

clearBtn.addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = "";
  inputEl.focus();
});

inputEl.focus();
setStatus();
