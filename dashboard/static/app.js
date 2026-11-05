const sourcesBody = document.querySelector("#sources-table tbody");
const modelsBody = document.querySelector("#models-table tbody");
const jobSelect = document.getElementById("job-select");
const jobLog = document.getElementById("job-log");

let selectedJobId = null;
let knownJobIds = new Set();

function pill(level, text) {
  return `<span class="pill pill-${level}">${text}</span>`;
}

function sourcePill(source) {
  if (!source.available) return pill("warn", "unavailable");
  if (source.failed > 0) return pill("bad", `${source.failed} failed`);
  if (source.pending > 0) return pill("warn", `${source.pending} pending`);
  return pill("good", "ok");
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || response.statusText);
  }
  return response.json();
}

async function runSource(key, button) {
  button.disabled = true;
  try {
    const job = await fetchJSON(`/api/sources/${key}/run`, { method: "POST" });
    await refreshJobs();
    selectJob(job.id);
  } catch (error) {
    alert(`Failed to start ${key}: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function trainModel(modelType, button) {
  button.disabled = true;
  try {
    const job = await fetchJSON(`/api/models/${modelType}/train`, { method: "POST" });
    await refreshJobs();
    selectJob(job.id);
  } catch (error) {
    alert(`Failed to start training for ${modelType}: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function refreshSources() {
  const sources = await fetchJSON("/api/sources");
  sourcesBody.innerHTML = "";
  for (const source of sources) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${source.label}${source.note ? `<div style="color:var(--muted);font-size:0.75rem">${source.note}</div>` : ""}</td>
      <td>${sourcePill(source)}</td>
      <td>${source.complete}</td>
      <td>${source.pending}</td>
      <td>${source.failed}</td>
      <td>${source.latest ?? "none"}</td>
      <td></td>`;
    const actionCell = row.querySelector("td:last-child");
    const button = document.createElement("button");
    button.textContent = "Sync";
    button.disabled = !source.available;
    button.onclick = () => runSource(source.key, button);
    actionCell.appendChild(button);
    sourcesBody.appendChild(row);
  }
}

async function refreshModels() {
  const models = await fetchJSON("/api/models");
  modelsBody.innerHTML = "";
  for (const model of models) {
    const row = document.createElement("tr");
    const stable = model.stable ? `${model.stable.version} (${formatMetrics(model.stable.performance)})` : "none";
    const beta = model.beta ? `${model.beta.version} (${formatMetrics(model.beta.performance)})` : "none";
    row.innerHTML = `
      <td>${model.label}</td>
      <td>${stable}</td>
      <td>${beta}</td>
      <td>${model.history_count}</td>
      <td></td>`;
    const actionCell = row.querySelector("td:last-child");
    const button = document.createElement("button");
    button.textContent = "Train";
    button.disabled = !model.trainable;
    button.title = model.train_note;
    button.onclick = () => trainModel(model.model_type, button);
    actionCell.appendChild(button);
    modelsBody.appendChild(row);
  }
}

function formatMetrics(performance) {
  if (!performance || Object.keys(performance).length === 0) return "-";
  return Object.entries(performance).slice(0, 2).map(([k, v]) => `${k}=${v}`).join(", ");
}

async function refreshJobs() {
  const jobsList = await fetchJSON("/api/jobs");
  const previouslySelected = selectedJobId;
  jobSelect.innerHTML = "";
  for (const job of jobsList) {
    const option = document.createElement("option");
    option.value = job.id;
    const started = new Date(job.started_at * 1000).toLocaleTimeString();
    option.textContent = `[${job.status}] ${job.label} (${started})`;
    jobSelect.appendChild(option);
  }
  if (previouslySelected && jobsList.some((j) => j.id === previouslySelected)) {
    jobSelect.value = previouslySelected;
  } else if (jobsList.length > 0) {
    selectJob(jobsList[0].id);
  }
}

function selectJob(jobId) {
  selectedJobId = jobId;
  jobSelect.value = jobId;
  refreshJobLog();
}

async function refreshJobLog() {
  if (!selectedJobId) return;
  const job = await fetchJSON(`/api/jobs/${selectedJobId}`);
  jobLog.textContent = `$ ${job.command.join(" ")}\n\n${job.log}`;
}

jobSelect.addEventListener("change", () => selectJob(jobSelect.value));

async function tick() {
  await Promise.all([refreshSources(), refreshModels(), refreshJobs()]);
  if (selectedJobId) await refreshJobLog();
}

tick();
setInterval(tick, 4000);
