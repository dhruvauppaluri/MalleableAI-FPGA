/* Offline, dependency-free first IDE shell. No CDN or remote model execution. */
const el = id => document.getElementById(id);
let selected = null;
let jobsSignature = '';
let refreshing = false;
async function api(path, payload) {
  const result = await fetch(path, payload === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await result.json();
  if (!result.ok) throw Error(data.detail || 'Request failed');
  return data;
}
async function models() {
  const data = await api('/api/models');
  const old = el('model').value;
  el('model').replaceChildren(...data.map(m => {const o = document.createElement('option'); o.value = m.path; o.textContent = m.path; return o;}));
  if (data.some(m => m.path === old)) el('model').value = old;
}
async function submit(kind, payload) {
  try { const result = await api('/api/jobs/' + kind, payload); selected = result.id; el('message').textContent = 'Queued ' + kind + ' experiment.'; await refresh(); }
  catch (e) { el('message').textContent = e.message; }
}
function config() {return {artifact: el('model').value, lanes: Number(el('lanes').value), seed: Number(el('seed').value)};}
el('generate').onclick = () => submit('generate', {...config(), prompt: el('prompt').value, max_new: Number(el('tokens').value), backend: el('backend').value});
el('analyze').onclick = () => submit('analyze', config());
el('optimize').onclick = () => {
  try {const payload = {...config(), prompt: el('prompt').value, horizon: Number(el('horizon').value), budget: 5};
    if (el('switch').value.trim()) payload.switch_cost = JSON.parse(el('switch').value);
    submit('optimize', payload);
  } catch(e) {el('message').textContent = 'Invalid overhead JSON: ' + e.message;}
};
el('train').onclick = () => submit('train', {dataset: el('dataset').value, device: el('device').value, steps: Number(el('steps').value), family: el('family').value, seed: Number(el('seed').value)});
el('quartus').onclick = () => submit('quartus', {lanes: Number(el('lanes').value)});
el('refresh').onclick = () => models().catch(e => el('message').textContent = e.message);
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
  const jobs = await api('/api/jobs');
  const signature = JSON.stringify(jobs);
  if (signature !== jobsSignature) {
  jobsSignature = signature;
  el('jobs').replaceChildren(...jobs.map(job => {
    const row = document.createElement('div'); row.className = 'job';
    const show = document.createElement('button'); show.textContent = job.kind + ' · ' + job.status + ' · ' + job.id.slice(0,8);
    show.onclick = () => {selected = job.id; refresh().catch(e => el('message').textContent=e.message);}; row.append(show);
    if (['queued','running'].includes(job.status)) {
      const cancel = document.createElement('button'); cancel.textContent = 'Cancel'; cancel.onclick = () => api('/api/job/' + job.id + '/cancel',{}).then(refresh); row.append(cancel);
    } else if (job.kind === 'train' && ['failed','cancelled','interrupted'].includes(job.status)) {
      const resume = document.createElement('button'); resume.textContent = 'Resume'; resume.onclick = () => api('/api/job/' + job.id + '/resume',{}).then(refresh).catch(e => el('message').textContent=e.message); row.append(resume);
    }
    return row;
  }));
  }
  if (selected) el('output').textContent = (await api('/api/job/' + selected + '/log')).text || 'Waiting for worker…';
  } finally { refreshing = false; }
}
models().then(refresh).catch(e => el('message').textContent = e.message);
setInterval(() => refresh().catch(e => el('message').textContent = e.message), 2000);
