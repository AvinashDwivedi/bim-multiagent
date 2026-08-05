const conversation = document.getElementById("conversation");
const messages = document.getElementById("messages");
const welcomePanel = document.getElementById("welcomePanel");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const sidebar = document.getElementById("sidebar");
let busy = false;

const agentWorkflow = document.getElementById("agentWorkflow");
const agentRows = new Map();

const agentDescriptions = {
  "BIM Supervisor": "Routing and final synthesis",
  "BIM Query Agent": "Ontology + typed BIM tools",
  "Verification Agent": "Independent evidence check"
};

function resetAgentWorkflow() {
  agentRows.clear();
  agentWorkflow.innerHTML = '<li class="workflow-empty"><span>·</span><div><strong>Waiting for a question</strong><small>Agents will appear in execution order</small></div></li>';
}

function startAgent(event) {
  agentWorkflow.querySelector(".workflow-empty")?.remove();
  const row = document.createElement("li");
  row.className = "active";
  row.dataset.eventId = event.id;
  const number = document.createElement("span");
  number.textContent = event.sequence;
  const content = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = event.agent;
  const detail = document.createElement("small");
  detail.textContent = agentDescriptions[event.agent] || "Specialist agent";
  content.append(name, detail);
  row.append(number, content);
  agentWorkflow.appendChild(row);
  agentRows.set(event.id, row);
  row.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function endAgent(event) {
  const row = agentRows.get(event.id);
  if (!row) return;
  row.classList.remove("active", "tool-active");
  row.classList.add("done");
  row.querySelector("small").textContent += ` · ${Number(event.elapsed_seconds).toFixed(1)}s`;
}

function markTool(event, active) {
  const candidates = [...agentRows.values()].reverse();
  const row = candidates.find(item => item.classList.contains("active") && item.querySelector("strong").textContent === event.agent);
  row?.classList.toggle("tool-active", active);
}

function scrollToBottom() {
  requestAnimationFrame(() => { conversation.scrollTop = conversation.scrollHeight; });
}

function addUserMessage(text) {
  const fragment = document.getElementById("userMessageTemplate").content.cloneNode(true);
  fragment.querySelector(".user-message").textContent = text;
  messages.appendChild(fragment);
  scrollToBottom();
}

function icon(name) {
  const paths = {
    warning: '<path d="M12 3 2.5 20h19Z"/><path d="M12 9v4M12 17h.01"/>',
    evidence: '<path d="M6 3h9l3 3v15H6Z"/><path d="M14 3v4h4M9 12h6M9 16h4"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>'
  };
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name]}</svg>`;
}

function addAssistantMessage(report) {
  const fragment = document.getElementById("assistantMessageTemplate").content.cloneNode(true);
  const article = fragment.querySelector(".assistant-row");
  const copy = fragment.querySelector(".answer-copy");
  const meta = fragment.querySelector(".answer-meta");
  copy.textContent = report.answer || "No answer was returned.";

  if (report.verification_status === "verified") {
    const label = document.createElement("div");
    label.className = "verified-label";
    label.innerHTML = `${icon("check")} Independently verified`;
    copy.appendChild(label);
  }

  if (report.limitations?.length) {
    const card = document.createElement("div");
    card.className = "meta-card warning";
    card.innerHTML = `<div class="meta-title">${icon("warning")} Model limitations</div>`;
    const list = document.createElement("ul");
    report.limitations.forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
    card.appendChild(list);
    meta.appendChild(card);
  }

  const evidenceIds = [...new Set((report.claims || []).flatMap(claim => claim.evidence_ids || []))];
  if (evidenceIds.length) {
    const card = document.createElement("div");
    card.className = "meta-card";
    card.innerHTML = `<div class="meta-title">${icon("evidence")} Evidence</div>`;
    const row = document.createElement("div");
    row.className = "evidence-row";
    evidenceIds.forEach(id => {
      const chip = document.createElement("span");
      chip.className = "evidence-chip";
      chip.textContent = id;
      row.appendChild(chip);
    });
    card.appendChild(row);
    meta.appendChild(card);
  }

  messages.appendChild(fragment);
  scrollToBottom();
  return article;
}

function addTypingMessage() {
  const fragment = document.getElementById("assistantMessageTemplate").content.cloneNode(true);
  const row = fragment.querySelector(".assistant-row");
  fragment.querySelector(".assistant-message").innerHTML = '<div class="typing"><i></i><i></i><i></i><span class="typing-status">Interpreting BIM question…</span></div>';
  messages.appendChild(fragment);
  scrollToBottom();
  return row;
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  sendButton.disabled = value;
}

async function askQuestion(question) {
  if (busy || !question.trim()) return;
  welcomePanel.classList.add("hidden");
  addUserMessage(question.trim());
  input.value = "";
  input.style.height = "auto";
  setBusy(true);
  resetAgentWorkflow();
  const typingRow = addTypingMessage();
  const status = typingRow.querySelector(".typing-status");

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim() })
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "The BIM service could not answer this question.");
    }
    if (!response.body) throw new Error("The browser could not open the live workflow stream.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalReport = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "agent_start") {
          startAgent(event);
          status.textContent = `${event.agent} is working…`;
        } else if (event.type === "agent_end") {
          endAgent(event);
        } else if (event.type === "tool_start") {
          markTool(event, true);
          status.textContent = `${event.agent} is using ${event.tool.replaceAll("_", " ")}…`;
        } else if (event.type === "tool_end") {
          markTool(event, false);
        } else if (event.type === "result") {
          finalReport = event.report;
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
      if (done) break;
    }
    if (!finalReport) throw new Error("The workflow ended without a final BIM answer.");
    typingRow.remove();
    addAssistantMessage(finalReport);
  } catch (error) {
    typingRow.remove();
    addAssistantMessage({
      answer: error.message,
      verification_status: "insufficient_evidence",
      limitations: ["The request did not complete. Check the server log for the failed stage."],
      claims: []
    });
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  askQuestion(input.value);
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});

document.getElementById("suggestions").addEventListener("click", event => {
  const button = event.target.closest("button[data-question]");
  if (button) askQuestion(button.dataset.question);
});

document.getElementById("newChatButton").addEventListener("click", () => {
  if (busy) return;
  messages.replaceChildren();
  welcomePanel.classList.remove("hidden");
  resetAgentWorkflow();
  input.focus();
});

document.getElementById("menuButton").addEventListener("click", () => sidebar.classList.toggle("open"));
document.addEventListener("click", event => {
  if (window.innerWidth <= 880 && sidebar.classList.contains("open") && !sidebar.contains(event.target) && !event.target.closest("#menuButton")) {
    sidebar.classList.remove("open");
  }
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!response.ok) throw new Error();
    document.getElementById("projectName").textContent = health.project_id || "Authorized BIM project";
    document.getElementById("connectionText").textContent = "Connected and authorized";
    document.getElementById("connectionDot").classList.add("online");
    document.getElementById("sourceCount").textContent = health.source_count ?? "—";
    document.getElementById("elementCount").textContent = Number(health.element_count || 0).toLocaleString();
    document.getElementById("contractVersion").textContent = `v${health.contract_version}`;
  } catch {
    document.getElementById("projectName").textContent = "BIM service unavailable";
    document.getElementById("connectionText").textContent = "Check server connection";
  }
}

loadHealth();
input.focus();
