const API_BASE = window.PROJECT2_API_BASE || "/project2/api/";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
let state = {model:'tree', lambda:0.01, rule:'assignment'};

async function api(url, options={}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function pct(v){return `${(Number(v)*100).toFixed(1)}%`}
function esc(v){return String(v ?? '').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
const labels={bill_length_mm:'Bill length (mm)',bill_depth_mm:'Bill depth (mm)',flipper_length_mm:'Flipper length (mm)',body_mass_g:'Body mass (g)',island:'Island',sex:'Sex',year:'Year'};

async function loadTask1(){
  const d=await api(`${API_BASE}task1/`);
  $('#baselineAccuracy').textContent=pct(d.accuracy);
  $('#baselineLeaves').textContent=d.leaves;
  $('#baselineTree').src=`data:image/png;base64,${d.tree_image}`;
}

function readControls(){
  state.model=$('#modelType').value;
  state.lambda=Number($('#lambdaSlider').value);
  state.rule=$('#selectionRule').value;
  $('#lambdaValue').textContent=state.lambda.toFixed(3);
}

async function refreshModel(){
  readControls();
  $('#selectedName').textContent='Loading…';
  const q=new URLSearchParams({model:state.model,lambda:state.lambda,rule:state.rule});
  const d=await api(`${API_BASE}model/?${q}`);
  $('#selectedName').textContent=d.selected_name;
  $('#selectedAccuracy').textContent=pct(d.accuracy);
  $('#complexityLabel').textContent=d.complexity_label;
  $('#selectedComplexity').textContent=d.complexity;
  const tbody=$('#candidateTable tbody');
  tbody.innerHTML=d.candidate_rows.map(r=>`<tr class="${r.selected?'selected-row':''}"><td>${esc(r.name)}${r.selected?' ✓':''}</td><td>${esc(r.fit_parameter)}</td><td>${pct(r.accuracy)}</td><td>${r.complexity}</td><td>${r.score.toFixed(5)}</td></tr>`).join('');
  if(state.model==='tree'){
    $('#modelVisual').innerHTML=`<h3>Selected decision tree</h3><div class="image-scroller"><img alt="Selected decision tree" src="data:image/png;base64,${d.tree_image}"></div>`;
  } else {
    const groups={}; d.coefficients.forEach(r=>(groups[r.species]??=[]).push(r));
    $('#modelVisual').innerHTML=`<h3>Largest logistic-regression coefficients</h3><p class="beginner-help">A positive coefficient pushes the model more toward that species; a negative coefficient pushes away. Larger absolute values have stronger influence.</p><div class="coef-grid">${Object.entries(groups).map(([species,rows])=>`<div class="coef-card"><h4>${esc(species)}</h4><ol>${rows.map(r=>`<li>${esc(r.feature)}: <b>${r.coefficient}</b></li>`).join('')}</ol></div>`).join('')}</div>`;
  }
  $('#counterfactualResults').innerHTML=''; $('#effectResults').innerHTML='';
}

async function loadExamples(){
  const d=await api(`${API_BASE}examples/`);
  $('#exampleSelect').innerHTML=d.examples.map(e=>`<option value="${e.index}">${esc(e.label)}</option>`).join('');
  await showExample();
}

async function showExample(){
  const idx=$('#exampleSelect').value || 0;
  const d=await api(`${API_BASE}example/?index=${idx}`);
  $('#exampleCard').innerHTML=`<div class="cf-title"><strong>Selected penguin</strong><span>True species: <b>${esc(d.species)}</b></span></div><div class="value-grid">${Object.keys(labels).map(k=>`<div class="value-chip"><small>${labels[k]}</small><b>${esc(d[k])}</b></div>`).join('')}</div>`;
  const targets=['Adelie','Chinstrap','Gentoo'].filter(s=>s!==d.species);
  if(targets.length) $('#targetLabel').value=targets[0];
}

async function generateCounterfactuals(){
  readControls();
  const btn=$('#counterfactualButton'); btn.disabled=true;
  $('#counterfactualStatus').textContent='Searching locally for the smallest changes that reach the requested species…';
  $('#counterfactualResults').innerHTML='';
  try{
    const d=await api(`${API_BASE}counterfactual/`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:state.model,lambda:state.lambda,rule:state.rule,example_index:Number($('#exampleSelect').value),target_label:$('#targetLabel').value,k:5})});
    $('#counterfactualStatus').textContent=d.found?`Found ${d.found} counterfactual example${d.found===1?'':'s'} using ${d.model_name}. Original model prediction: ${d.original_prediction}.`:`No counterfactual was found after progressively increasing the sample size and local variance.`;
    if(d.found){
      $('#counterfactualResults').innerHTML=`<div class="notice"><b>Method:</b> ${esc(d.method_note)}</div>`+d.counterfactuals.map((c,i)=>`<article class="cf-card"><div class="cf-title"><h3>Counterfactual ${i+1}</h3><span>MAD-weighted distance: <b>${c.distance}</b></span></div><div class="change-list">${c.changes.length?c.changes.map(ch=>`<div class="change"><b>${esc(ch.label)}</b>${esc(ch.from)} → <strong>${esc(ch.to)}</strong></div>`).join(''):'<div class="change">No displayed feature change.</div>'}</div></article>`).join('');
    }
  }catch(e){$('#counterfactualStatus').textContent=`Could not generate counterfactuals: ${e.message}`}
  finally{btn.disabled=false}
}

async function calculateEffects(){
  readControls();
  const btn=$('#effectButton'); btn.disabled=true;
  $('#effectStatus').textContent='Calculating PDP and ALE from the selected model…'; $('#effectResults').innerHTML='';
  try{
    const q=new URLSearchParams({model:state.model,lambda:state.lambda,rule:state.rule,feature:$('#effectFeature').value});
    const d=await api(`${API_BASE}effects/?${q}`);
    $('#effectStatus').textContent=`Plots use ${d.model_name}. ${d.ale_method}`;
    $('#effectResults').innerHTML=`<div class="plot-card"><img alt="PDP plot" src="data:image/png;base64,${d.pdp_image}"><p class="beginner-help">${esc(d.pdp_note)}</p></div><div class="plot-card"><img alt="ALE plot" src="data:image/png;base64,${d.ale_image}"><p class="beginner-help">${esc(d.ale_method)}</p></div>`;
  }catch(e){$('#effectStatus').textContent=`Could not calculate effects: ${e.message}`}
  finally{btn.disabled=false}
}

function setupTooltip(){
  const tip=$('#tooltip');
  $$('.info-btn').forEach(btn=>{
    btn.addEventListener('mouseenter',e=>{tip.textContent=btn.dataset.tip;tip.style.display='block';move(e)});
    btn.addEventListener('mousemove',move);
    btn.addEventListener('mouseleave',()=>tip.style.display='none');
    btn.addEventListener('focus',()=>{tip.textContent=btn.dataset.tip;tip.style.display='block';const r=btn.getBoundingClientRect();tip.style.left=`${r.left}px`;tip.style.top=`${r.bottom+8}px`});
    btn.addEventListener('blur',()=>tip.style.display='none');
  });
  function move(e){tip.style.left=`${Math.min(e.clientX+14,window.innerWidth-300)}px`;tip.style.top=`${Math.min(e.clientY+14,window.innerHeight-120)}px`}
}

$('#beginnerToggle').addEventListener('change',e=>document.body.classList.toggle('hidden-help',!e.target.checked));
$('#modelType').addEventListener('change',refreshModel);
$('#lambdaSlider').addEventListener('input',()=>{$('#lambdaValue').textContent=Number($('#lambdaSlider').value).toFixed(3)});
$('#lambdaSlider').addEventListener('change',refreshModel);
$('#selectionRule').addEventListener('change',refreshModel);
$('#exampleSelect').addEventListener('change',showExample);
$('#counterfactualButton').addEventListener('click',generateCounterfactuals);
$('#effectButton').addEventListener('click',calculateEffects);

Promise.all([loadTask1(),refreshModel(),loadExamples()]).catch(e=>console.error(e));
setupTooltip();
