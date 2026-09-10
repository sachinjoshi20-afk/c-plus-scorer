const $ = s => document.querySelector(s);
let DATA = {rows: []}, sortKey = 'c_plus', sortDir = -1;
const CLS = {'BUY':'buy','HOLD/WAIT':'hold','SELL':'sell','NOT SCORED (BFSI)':'na'};

async function init(){
  const res = await fetch('data/c_plus_scores.json');
  DATA = await res.json();
  $('#updated').textContent = 'Last updated: ' +
    new Date(DATA.updated).toLocaleString('en-IN',{timeZone:'Asia/Kolkata'}) + ' IST';
  const sectors = [...new Set(DATA.rows.map(r=>r.sector).filter(Boolean))].sort();
  $('#sector').innerHTML = '<option value="">All sectors</option>' +
    sectors.map(s=>`<option>${s}</option>`).join('');
  $('#search').addEventListener('input', render);
  $('#signal').addEventListener('change', render);
  $('#sector').addEventListener('change', render);
  document.querySelectorAll('th[data-key]').forEach(th =>
    th.addEventListener('click', () => {
      const k = th.dataset.key;
      sortDir = (sortKey === k) ? -sortDir : (k === 'c_plus' ? -1 : 1);
      sortKey = k; render();
    }));
  render();
}

function filtered(){
  const q = $('#search').value.trim().toLowerCase();
  const sig = $('#signal').value, sec = $('#sector').value;
  const rows = DATA.rows.filter(r =>
    (!q || ((r.ticker||'')+' '+(r.name||'')).toLowerCase().includes(q)) &&
    (!sig || r.recommendation === sig) &&
    (!sec || r.sector === sec));
  rows.sort((a,b) => {
    const va = a[sortKey], vb = b[sortKey];
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va > vb ? 1 : va < vb ? -1 : 0) * sortDir;
  });
  return rows;
}

function render(){
  const c = {BUY:0,'HOLD/WAIT':0,SELL:0,'NOT SCORED (BFSI)':0};
  DATA.rows.forEach(r => { if (c[r.recommendation] !== undefined) c[r.recommendation]++; });
  $('#summary').textContent =
    `🟢 BUY ${c.BUY}  ·  🟡 HOLD/WAIT ${c['HOLD/WAIT']}  ·  🔴 SELL ${c.SELL}  ·  ⚪ BFSI not scored ${c['NOT SCORED (BFSI)']}`;
  $('#tbody').innerHTML = filtered().map((r,i) => `
    <tr>
      <td>${i+1}</td>
      <td><b>${(r.ticker||'').replace('.NS','')}</b> <span class="badge">${r.badge||''}</span></td>
      <td>${r.name||''}</td>
      <td>${r.sector||''}</td>
      <td class="score">${r.c_plus == null ? '—' :
        `<span class="bar" style="width:${Math.round(r.c_plus)}px"></span>${r.c_plus.toFixed(2)}`}</td>
      <td><span class="pill ${CLS[r.recommendation]||''}">${r.recommendation||''}</span></td>
      <td class="reason">${r.reason||''}</td>
    </tr>`).join('');
}
init();
