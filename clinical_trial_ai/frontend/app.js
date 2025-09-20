const API_BASE = ""; // same origin; set like "http://localhost:8000" if needed.
const qs = (s, el=document) => el.querySelector(s);
const elMessages = qs("#messages");
const elPrompt = qs("#prompt");
const elSend = qs("#send");
const elClear = qs("#clearChat");
const elTopK = qs("#topk");
const elFile = qs("#fileInput");
const elUpload = qs("#uploadBtn");
const elDocList = qs("#docList");
const elRefreshDocs = qs("#refreshDocs");
const elThemeToggle = qs("#themeToggle");
const elSystemStatus = qs("#systemStatus");

// Theme
const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);
elThemeToggle.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

// Helpers
function fmtBytes(b) {
  if (b === 0) return "0 B";
  const k = 1024, sizes = ["B","KB","MB","GB","TB"];
  const i = Math.floor(Math.log(b)/Math.log(k));
  return parseFloat((b/Math.pow(k,i)).toFixed(2)) + " " + sizes[i];
}
function tsToLocal(ts) {
  try { return new Date(ts).toLocaleString(); } catch { return String(ts); }
}
function scrollBottom(container) { container.scrollTop = container.scrollHeight; }

// Chat rendering
function addMessage(role, text, meta={}) {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  const bubble = document.createElement("div");
  bubble.className = "bubble " + (role === "agent" ? "agent" : "user");
  bubble.textContent = text;
  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  metaEl.innerHTML = Object.entries(meta).map(([k,v]) => `<span>${k}: ${v}</span>`).join("");
  wrap.appendChild(avatar);
  const col = document.createElement("div");
  col.appendChild(bubble);
  if (Object.keys(meta).length) col.appendChild(metaEl);
  wrap.appendChild(col);
  elMessages.appendChild(wrap);
  scrollBottom(elMessages);
}

// Loading indicator
const thinking = [];
function addThinking() {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  wrap.dataset.loader = "1";
  wrap.innerHTML = `
    <div class="avatar"></div>
    <div>
      <div class="bubble agent"><span class="muted">Thinking...</span></div>
    </div>`;
  elMessages.appendChild(wrap);
  scrollBottom(elMessages);
  thinking.push(wrap);
  return thinking.length - 1;
}
function removeThinking(idx) {
  const el = thinking[idx];
  if (el && el.parentElement) el.parentElement.removeChild(el);
  thinking[idx] = null;
}

// Chat actions -> POST /api/v1/query
async function sendQuery(query) {
  const topk = Math.max(1, Math.min(20, Number(elTopK.value) || 5));
  addMessage("user", query);
  const thinkingIdx = addThinking();
  try {
    const res = await fetch(`${API_BASE}/api/v1/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, topk })
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }
    const data = await res.json();
    removeThinking(thinkingIdx);
    const timeVal = data.processing_time ?? data.processingtime ?? 0;
    addMessage("agent", data.response || "No response", {
      confidence: (data.confidence != null ? Number(data.confidence).toFixed(2) : "n/a"),
      time: (typeof timeVal === "number" ? timeVal.toFixed(2) + "s" : String(timeVal)),
      session: data.session_id || data.sessionid || "-"
    });
  } catch (e) {
    removeThinking(thinkingIdx);
    addMessage("agent", `Error: ${e.message}`, { status: "failed" });
  }
}

elSend.addEventListener("click", () => {
  const q = elPrompt.value.trim();
  if (!q) return;
  elPrompt.value = "";
  sendQuery(q);
});
elPrompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    elSend.click();
  }
});
elClear.addEventListener("click", () => {
  elMessages.innerHTML = "";
});

// Documents
async function listDocuments() {
  elDocList.innerHTML = `<div class="muted">Loading documents...</div>`;
  try {
    const res = await fetch(`${API_BASE}/api/v1/documents`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const docs = await res.json();
    renderDocs(docs);
  } catch (e) {
    elDocList.innerHTML = `<div class="error">Failed to load: ${e.message}</div>`;
  }
}

function renderDocs(docs) {
  if (!docs || docs.length === 0) {
    elDocList.innerHTML = `<div class="muted">No documents uploaded yet.</div>`;
    return;
  }
  elDocList.innerHTML = "";
  docs.forEach(d => {
    const row = document.createElement("div");
    row.className = "file";
    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = d.filename;
    const sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = `ID: ${d.documentid || d.document_id || "-"} • Size: ${fmtBytes(d.filesize)} • Uploaded: ${tsToLocal(d.uploadtime)}`;
    left.appendChild(name);
    left.appendChild(sub);

    const right = document.createElement("div");
    right.className = "row right";
    const status = document.createElement("span");
    status.className = "chip";
    status.textContent = `Status: ${d.processingstatus}`;
    right.appendChild(status);

    const btnSummary = document.createElement("button");
    btnSummary.className = "btn";
    btnSummary.textContent = "Get summary";
    btnSummary.disabled = (d.processingstatus !== "completed");
    btnSummary.addEventListener("click", () => summarizeDoc(d.documentid || d.document_id));
    right.appendChild(btnSummary);

    const btnDelete = document.createElement("button");
    btnDelete.className = "btn danger";
    btnDelete.textContent = "Delete";
    btnDelete.addEventListener("click", () => deleteDoc(d.documentid || d.document_id));
    right.appendChild(btnDelete);

    row.appendChild(left);
    row.appendChild(right);
    elDocList.appendChild(row);
  });
}

// Upload -> POST /api/v1/documents/upload
elUpload.addEventListener("click", async () => {
  const file = elFile.files[0];
  if (!file) return alert("Choose a file first.");
  const allowed = [".pdf",".txt",".csv"];
  const ok = allowed.some(ext => file.name.toLowerCase().endsWith(ext));
  if (!ok) return alert("Unsupported type. Allowed: .pdf, .txt, .csv");
  const form = new FormData();
  form.append("file", file);
  elUpload.disabled = true;
  elUpload.textContent = "Uploading...";
  try {
    const res = await fetch(`${API_BASE}/api/v1/documents/upload`, { method: "POST", body: form });
    const txt = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${txt}`);
    let data;
    try { data = JSON.parse(txt); } catch { data = {}; }
    alert(data.message || "Uploaded. Processing in background.");
    elFile.value = "";
    listDocuments();
  } catch (e) {
    alert("Upload failed: " + e.message);
  } finally {
    elUpload.disabled = false;
    elUpload.textContent = "Upload";
  }
});

// Delete -> DELETE /api/v1/documents/{id}
async function deleteDoc(id) {
  if (!confirm("Delete this document?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/documents/${id}`, { method: "DELETE" });
    const txt = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${txt}`);
    listDocuments();
  } catch (e) {
    alert("Delete failed: " + e.message);
  }
}

// Summary -> POST /api/v1/documents/{id}/summary and then show in chat
async function summarizeDoc(id) {
  const loadingIdx = addThinking();
  try {
    const res = await fetch(`${API_BASE}/api/v1/documents/${id}/summary`, { method: "POST" });
    const txt = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${txt}`);
    const data = JSON.parse(txt);
    removeThinking(loadingIdx);
    const timeVal = data.processing_time ?? data.processingtime ?? 0;
    addMessage("agent", data.summary || "No summary generated.", {
      confidence: (data.confidence != null ? Number(data.confidence).toFixed(2) : "n/a"),
      time: (typeof timeVal === "number" ? timeVal.toFixed(2) + "s" : String(timeVal)),
      doc: id
    });
  } catch (e) {
    removeThinking(loadingIdx);
    addMessage("agent", `Summary error: ${e.message}`, { doc: id });
  }
}

// System status -> GET /api/v1/systemstatus
elSystemStatus.addEventListener("click", async () => {
  try {
    const res = await fetch(`${API_BASE}/api/v1/systemstatus`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    alert(`System: ${data.success ? "OK" : "Issue"}\nLLM: ${JSON.stringify(data.llmstatus)}\nAgents: ${data.availableagents?.length || 0}`);
  } catch (e) {
    alert("Status check failed: " + e.message);
  }
});

// Init
qs("#refreshDocs").addEventListener("click", listDocuments);
listDocuments();
