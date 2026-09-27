#!/usr/bin/env python3
"""Build a self-contained HTML catalog browser from catalog/delaurenti/all_products.csv."""
import csv, json, sys, re

CAT_DIR = sys.argv[1]
OUT = sys.argv[2]
KEEP = ["category", "subtype", "brand", "place_of_origin", "grape_variety", "flavor_profile", "milk",
        "raw_or_pasteurized", "name", "size", "price_usd", "price_per_100ml", "availability", "flags",
        "tagline", "url", "description", "product_id"]
# category order + department, from cats.txt (same order as the index README)
slugs = ["grocery-olive-oil-26"] + [l.strip().rstrip("/").split("/")[-1] for l in open(f"{CAT_DIR}/tools/cats.txt") if l.strip()]
cats, data = [], []
for slug in slugs:
    d = json.load(open(f"{CAT_DIR}/{slug}/products.json"))
    cats.append({"slug": slug, "dept": slug.split("-")[0], "name": d["category"], "url": d["url"], "count": d["count"]})
    for r in d["products"]:
        row = {k: r[k] for k in KEEP if r.get(k) not in ("", None)}
        row["ck"] = slug
        data.append(row)
DEPT_KO = {"grocery": "식료품 Grocery", "cheese": "치즈 Cheese", "meats": "육가공 Meats", "antipasti": "안티파스티 Antipasti",
           "entertaining": "엔터테이닝 Entertaining", "gifts": "기프트 Gifts", "wine": "와인 Wine", "cafe": "카페 Cafe"}

DATA_JSON = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
CATS_JSON = json.dumps(cats, ensure_ascii=False).replace("</", "<\\/")
DEPT_JSON = json.dumps(DEPT_KO, ensure_ascii=False)

html = r"""<title>DeLaurenti Pantry Index</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f3f2ec; --card:#fbfaf6; --ink:#1d2118; --muted:#6a6f5d; --line:#d9d8cd; --line-soft:#e8e7de;
  --olive:#4d5f23; --olive-soft:#e6ead6; --olive-ink:#33410f; --pill-ok:#e3eedc; --pill-ok-ink:#2d5a1e;
  --pill-out:#f1e3dc; --pill-out-ink:#8a3b25; --focus:#7f9a3a; --shadow:0 1px 2px rgba(29,33,24,.06);
  --display:"Fraunces",Georgia,"Times New Roman",serif; --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme:dark; --paper:#171a13; --card:#20241b; --ink:#ebe9df; --muted:#a2a692; --line:#3a3f31; --line-soft:#2b3025;
  --olive:#a7bf5c; --olive-soft:#2c3520; --olive-ink:#d3e39c; --pill-ok:#2a3a22; --pill-ok-ink:#b9dca4;
  --pill-out:#40281f; --pill-out-ink:#f0b8a2; --focus:#b9d06e; --shadow:0 1px 2px rgba(0,0,0,.3);}}
:root[data-theme="dark"]{
  color-scheme:dark; --paper:#171a13; --card:#20241b; --ink:#ebe9df; --muted:#a2a692; --line:#3a3f31; --line-soft:#2b3025;
  --olive:#a7bf5c; --olive-soft:#2c3520; --olive-ink:#d3e39c; --pill-ok:#2a3a22; --pill-ok-ink:#b9dca4;
  --pill-out:#40281f; --pill-out-ink:#f0b8a2; --focus:#b9d06e; --shadow:0 1px 2px rgba(0,0,0,.3);}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);font-size:14px;line-height:1.5}
a{color:inherit}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.wrap{max-width:1360px;margin:0 auto;padding-inline:16px;padding-block:0 32px}
header.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;padding-block:22px 14px;border-bottom:1px solid var(--line)}
header.top h1{font-family:var(--display);font-weight:500;font-size:28px;margin:0;letter-spacing:-.01em;text-wrap:balance}
header.top h1 em{font-style:italic;color:var(--olive)}
header.top .sub{color:var(--muted);font-size:13px}
header.top .sub a{color:var(--muted)}
.layout{display:grid;grid-template-columns:250px minmax(0,1fr);gap:24px;padding-top:18px}
nav.side{position:sticky;top:env(safe-area-inset-top,0px);align-self:start;max-height:calc(100vh - 8px);overflow:auto;padding-right:4px}
nav.side .dept{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px;padding-left:8px}
nav.side .dept:first-child{margin-top:0}
nav.side button{display:flex;justify-content:space-between;align-items:center;gap:8px;width:100%;text-align:left;background:none;border:0;border-radius:6px;padding:5px 8px;font:inherit;color:var(--ink);cursor:pointer}
nav.side button:hover{background:var(--line-soft)}
nav.side button.on{background:var(--olive-soft);color:var(--olive-ink);font-weight:600}
nav.side button .n{font-family:var(--mono);font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
nav.side button.on .n{color:var(--olive-ink)}
.mobile-nav{display:none;margin-bottom:12px}
select,input[type=search]{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 10px}
main{min-width:0}
.cat-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:6px 16px;margin-bottom:12px}
.cat-head h2{font-family:var(--display);font-weight:500;font-size:24px;margin:0;text-wrap:balance}
.cat-head .src{font-size:12px;color:var(--muted)}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;padding:10px 0 12px;border-top:1px solid var(--line-soft);border-bottom:1px solid var(--line-soft);margin-bottom:12px}
.stats div{display:flex;flex-direction:column}
.stats b{font-family:var(--mono);font-weight:500;font-size:18px;font-variant-numeric:tabular-nums}
.stats span{font-size:11px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.controls input[type=search]{flex:1 1 220px;min-width:0}
.controls select{max-width:100%}
.controls label.chk{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:13px;cursor:pointer}
.controls .reset{background:none;border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit;color:var(--muted);cursor:pointer}
.controls .reset:hover{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;color:var(--ink);font:inherit;font-size:12px}
.chip.on{background:var(--olive);border-color:var(--olive);color:#fff}
:root[data-theme="dark"] .chip.on{color:#15190f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .chip.on{color:#15190f}}
.chip .n{opacity:.7;margin-left:4px;font-family:var(--mono);font-size:11px}
.result-line{display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:12px;color:var(--muted);margin-bottom:8px}
.result-line .view button{background:none;border:1px solid var(--line);border-radius:6px;padding:3px 8px;font:inherit;font-size:12px;color:var(--muted);cursor:pointer}
.result-line .view button.on{color:var(--ink);border-color:var(--ink)}
.group h3{font-family:var(--display);font-weight:500;font-size:17px;margin:22px 0 8px;display:flex;align-items:baseline;gap:8px}
.group h3 .n{font-family:var(--mono);font-size:12px;color:var(--muted);font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line-soft);border-radius:8px;padding:12px 12px 10px;display:flex;flex-direction:column;gap:6px;box-shadow:var(--shadow)}
.card .brand{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between;gap:8px}
.card .brand .origin{text-transform:none;letter-spacing:0}
.card .name{font-weight:600;font-size:14px;line-height:1.35;text-decoration:none}
.card .name:hover{text-decoration:underline;text-decoration-color:var(--olive)}
.card .meta{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.card .price{color:var(--ink);font-size:16px;font-weight:500}
.card .attrs{display:flex;flex-wrap:wrap;gap:4px}
.tag{font-size:11px;border-radius:4px;padding:1px 6px;background:var(--line-soft);color:var(--muted)}
.pill{font-size:11px;border-radius:999px;padding:1px 8px;font-weight:500}
.pill.ok{background:var(--pill-ok);color:var(--pill-ok-ink)}
.pill.out{background:var(--pill-out);color:var(--pill-out-ink)}
.card .tagline{font-size:12px;color:var(--olive-ink);font-style:italic}
.card details{font-size:12px;color:var(--muted);margin-top:auto}
.card details summary{cursor:pointer;color:var(--muted);list-style:none;font-size:12px}
.card details summary::-webkit-details-marker{display:none}
.card details summary::before{content:"+ ";font-family:var(--mono)}
.card details[open] summary::before{content:"− "}
.card details p{margin:6px 0 0;white-space:pre-line;line-height:1.5}
.tablewrap{overflow-x:auto;border:1px solid var(--line-soft);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line-soft);vertical-align:top}
th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--card);cursor:pointer;white-space:nowrap}
th.on{color:var(--ink)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right}
th.num{text-align:right}
td a{text-decoration:none;font-weight:500}
td a:hover{text-decoration:underline}
tr:last-child td{border-bottom:0}
.empty{padding:40px 0;text-align:center;color:var(--muted)}
footer{margin-top:36px;padding-top:12px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);line-height:1.6}
@media (max-width:820px){
  .layout{grid-template-columns:1fr}
  nav.side{display:none}
  .mobile-nav{display:block}
  .mobile-nav select{width:100%}
}
@media (prefers-reduced-motion:no-preference){.card{transition:border-color .15s}.card:hover{border-color:var(--line)}}
</style>

<div class="wrap">
<header class="top">
  <h1>DeLaurenti <em>Pantry Index</em></h1>
  <div class="sub">시애틀 파이크 플레이스 마켓의 이탈리안 식료품점 <a href="https://delaurenti.com/" target="_blank" rel="noopener">delaurenti.com</a> 전 카테고리 제품 정리 · 2026-09-27 수집</div>
</header>

<div class="layout">
  <nav class="side" id="sidenav" aria-label="카테고리"></nav>
  <main>
    <div class="mobile-nav"><select id="catSelect" aria-label="카테고리 선택"></select></div>
    <div class="cat-head">
      <h2 id="catTitle"></h2>
      <a class="src" id="catSrc" href="#" target="_blank" rel="noopener">사이트에서 보기 ↗</a>
    </div>
    <div class="stats" id="stats"></div>
    <div class="controls">
      <input type="search" id="q" placeholder="제품명, 브랜드, 설명 검색…" aria-label="검색">
      <select id="fOrigin" aria-label="산지"></select>
      <select id="fBrand" aria-label="브랜드"></select>
      <select id="fExtra" aria-label="추가 속성" hidden></select>
      <label class="chk"><input type="checkbox" id="fStock"> 재고 있는 것만</label>
      <button class="reset" id="reset" type="button">필터 초기화</button>
    </div>
    <div class="chips" id="chips"></div>
    <div class="result-line">
      <span id="resultCount"></span>
      <span class="view">
        <button type="button" id="vCards" class="on">카드</button>
        <button type="button" id="vTable">표</button>
      </span>
    </div>
    <div id="results"></div>
  </main>
</div>

<footer>
  가격과 재고는 수집 시점 기준이며 사이트에서 달라질 수 있습니다. 브랜드·산지·품종·우유 종류는 사이트 필터가 제공하는 값만 반영되어 일부 제품은 비어 있습니다.
  유형(subtype)은 제품명 키워드로 나눈 분류라 경계 사례는 사이트에서 확인하세요. 가격은 미국 달러.
</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script id="cats" type="application/json">__CATS__</script>
<script>
const PRODUCTS = JSON.parse(document.getElementById('data').textContent);
const CATS = JSON.parse(document.getElementById('cats').textContent);
const DEPT = __DEPT__;
const byCat = {};
for (const p of PRODUCTS) (byCat[p.ck] ||= []).push(p);
const catBySlug = Object.fromEntries(CATS.map(c => [c.slug, c]));
const state = { cat: null, q: '', origin: '', brand: '', extra: '', stock: false, sub: '', view: 'cards', sort: 'price', dir: 1 };
try { const v = localStorage.getItem('dl.view'); if (v) state.view = v; } catch (e) {}

const $ = id => document.getElementById(id);
const fmt = v => v ? '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const isOut = p => /out/i.test(p.availability || '');
const EXTRA_LABEL = { grape_variety: '품종', flavor_profile: '맛 프로필', milk: '우유 종류', raw_or_pasteurized: '살균' };

function buildNav() {
  const nav = $('sidenav'), sel = $('catSelect');
  let html = '', opts = '', lastDept = '';
  for (const c of CATS) {
    if (c.dept !== lastDept) { html += `<div class="dept">${esc(DEPT[c.dept] || c.dept)}</div>`; opts += (lastDept ? '</optgroup>' : '') + `<optgroup label="${esc(DEPT[c.dept] || c.dept)}">`; lastDept = c.dept; }
    html += `<button type="button" data-cat="${esc(c.slug)}"><span>${esc(c.name)}</span><span class="n">${c.count}</span></button>`;
    opts += `<option value="${esc(c.slug)}">${esc(c.name)} (${c.count})</option>`;
  }
  nav.innerHTML = html; sel.innerHTML = opts + '</optgroup>';
  nav.addEventListener('click', e => { const b = e.target.closest('button'); if (b) selectCat(b.dataset.cat); });
  sel.addEventListener('change', () => selectCat(sel.value));
}

function selectCat(slug) {
  state.cat = slug; const name = (catBySlug[slug] || {}).name || slug; state.q = ''; state.origin = ''; state.brand = ''; state.extra = ''; state.sub = ''; state.stock = false;
  $('q').value = ''; $('fStock').checked = false;
  document.querySelectorAll('#sidenav button').forEach(b => b.classList.toggle('on', b.dataset.cat === slug));
  $('catSelect').value = slug;
  const c = catBySlug[slug];
  $('catTitle').textContent = name + (c ? ` · ${(DEPT[c.dept] || c.dept).split(' ')[1] || ''}` : '');
  $('catSrc').href = c ? c.url : '#';
  const rows = byCat[slug] || [];
  // stats
  const prices = rows.map(r => +r.price_usd).filter(Boolean).sort((a, b) => a - b);
  const inStock = rows.filter(r => !isOut(r)).length;
  const med = prices.length ? prices[Math.floor(prices.length / 2)] : 0;
  $('stats').innerHTML = `<div><b>${rows.length}</b><span>제품</span></div><div><b>${inStock}</b><span>재고 있음</span></div><div><b>${rows.length - inStock}</b><span>품절</span></div>` +
    (prices.length ? `<div><b>${fmt(prices[0])} – ${fmt(prices[prices.length - 1])}</b><span>가격 범위</span></div><div><b>${fmt(med)}</b><span>중간 가격</span></div>` : '') +
    `<div><b>${new Set(rows.map(r => r.brand).filter(Boolean)).size}</b><span>브랜드</span></div>`;
  // filters
  fillSelect($('fOrigin'), rows, 'place_of_origin', '산지 전체');
  fillSelect($('fBrand'), rows, 'brand', '브랜드 전체');
  const extraKey = Object.keys(EXTRA_LABEL).find(k => rows.some(r => r[k]));
  state.extraKey = extraKey;
  $('fExtra').hidden = !extraKey;
  if (extraKey) fillSelect($('fExtra'), rows, extraKey, EXTRA_LABEL[extraKey] + ' 전체');
  // subtype chips
  const subs = count(rows, 'subtype');
  $('chips').innerHTML = subs.length ? `<button type="button" class="chip on" data-sub="">전체<span class="n">${rows.length}</span></button>` +
    subs.map(([k, n]) => `<button type="button" class="chip" data-sub="${esc(k)}">${esc(k)}<span class="n">${n}</span></button>`).join('') : '';
  render();
  try { localStorage.setItem('dl.cat', slug); } catch (e) {}
}
function count(rows, key) {
  const m = new Map();
  for (const r of rows) if (r[key]) m.set(r[key], (m.get(r[key]) || 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}
function fillSelect(sel, rows, key, allLabel) {
  sel.innerHTML = `<option value="">${allLabel}</option>` + count(rows, key).map(([k, n]) => `<option value="${esc(k)}">${esc(k)} (${n})</option>`).join('');
  sel.value = '';
}

function filtered() {
  const rows = byCat[state.cat] || [];
  const q = state.q.trim().toLowerCase();
  return rows.filter(r =>
    (!state.origin || r.place_of_origin === state.origin) &&
    (!state.brand || r.brand === state.brand) &&
    (!state.extra || r[state.extraKey] === state.extra) &&
    (!state.sub || r.subtype === state.sub) &&
    (!state.stock || !isOut(r)) &&
    (!q || [r.name, r.brand, r.tagline, r.description, r.place_of_origin, r.subtype].join(' ').toLowerCase().includes(q)));
}
function sortRows(rows) {
  const k = state.sort, d = state.dir;
  return rows.slice().sort((a, b) => {
    if (k === 'price' || k === 'unit') { const ka = k === 'price' ? 'price_usd' : 'price_per_100ml'; const x = +a[ka] || Infinity, y = +b[ka] || Infinity; return (x - y) * d; }
    return String(a[k] || '').localeCompare(String(b[k] || '')) * d;
  });
}

function card(p) {
  const out = isOut(p);
  const attrs = ['grape_variety', 'flavor_profile', 'milk', 'raw_or_pasteurized'].filter(k => p[k]).map(k => `<span class="tag">${esc(p[k])}</span>`).join('');
  return `<article class="card">
    <div class="brand"><span>${esc(p.brand || '')}</span><span class="origin">${esc(p.place_of_origin || '')}</span></div>
    <a class="name" href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.name)}</a>
    <div class="meta"><span class="price">${fmt(p.price_usd)}</span>${p.size ? `<span>${esc(p.size)}</span>` : ''}${p.price_per_100ml ? `<span>${fmt(p.price_per_100ml)}/100ml</span>` : ''}</div>
    ${p.tagline ? `<div class="tagline">${esc(p.tagline)}</div>` : ''}
    <div class="attrs"><span class="pill ${out ? 'out' : 'ok'}">${out ? '품절' : '재고 있음'}</span>${p.flags ? `<span class="tag">${esc(p.flags)}</span>` : ''}${attrs}</div>
    ${p.description ? `<details><summary>설명</summary><p>${esc(p.description)}</p></details>` : ''}
  </article>`;
}
function table(rows) {
  const extra = state.extraKey;
  const cols = [['brand', '브랜드'], ['name', '제품명'], ['size', '용량'], ['price', '가격', 'num'], ['unit', '$/100ml', 'num'], ['place_of_origin', '산지']];
  if (extra) cols.push([extra, EXTRA_LABEL[extra]]);
  if (rows.some(r => r.subtype)) cols.push(['subtype', '유형']);
  cols.push(['availability', '재고']);
  const th = cols.map(([k, l, c]) => `<th class="${c || ''} ${state.sort === k ? 'on' : ''}" data-sort="${k}">${l}${state.sort === k ? (state.dir > 0 ? ' ↑' : ' ↓') : ''}</th>`).join('');
  const tr = rows.map(p => `<tr>` + cols.map(([k, , c]) => {
    let v;
    if (k === 'name') v = `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.name)}</a>${p.tagline ? `<div class="tagline" style="font-size:12px;color:var(--olive-ink);font-style:italic">${esc(p.tagline)}</div>` : ''}`;
    else if (k === 'price') v = fmt(p.price_usd);
    else if (k === 'unit') v = p.price_per_100ml ? fmt(p.price_per_100ml) : '';
    else if (k === 'availability') v = `<span class="pill ${isOut(p) ? 'out' : 'ok'}">${isOut(p) ? '품절' : '재고 있음'}</span>`;
    else v = esc(p[k] || '');
    return `<td class="${c || ''}">${v}</td>`;
  }).join('') + `</tr>`).join('');
  return `<div class="tablewrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

function render() {
  const rows = sortRows(filtered());
  const all = (byCat[state.cat] || []).length;
  $('resultCount').textContent = rows.length === all ? `${all}개 제품` : `${rows.length}개 / 전체 ${all}개`;
  document.querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c.dataset.sub === state.sub));
  $('vCards').classList.toggle('on', state.view === 'cards'); $('vTable').classList.toggle('on', state.view === 'table');
  const box = $('results');
  if (!rows.length) { box.innerHTML = `<div class="empty">조건에 맞는 제품이 없습니다.</div>`; return; }
  if (state.view === 'table') { box.innerHTML = table(rows); return; }
  // cards, grouped by subtype when the category has subtypes and no single subtype is chosen
  const hasSub = rows.some(r => r.subtype) && !state.sub;
  if (hasSub) {
    const groups = count(rows, 'subtype');
    box.innerHTML = groups.map(([k, n]) => `<section class="group"><h3>${esc(k)}<span class="n">${n}</span></h3><div class="grid">${rows.filter(r => r.subtype === k).map(card).join('')}</div></section>`).join('');
  } else {
    box.innerHTML = `<div class="grid">${rows.map(card).join('')}</div>`;
  }
}

$('q').addEventListener('input', e => { state.q = e.target.value; render(); });
$('fOrigin').addEventListener('change', e => { state.origin = e.target.value; render(); });
$('fBrand').addEventListener('change', e => { state.brand = e.target.value; render(); });
$('fExtra').addEventListener('change', e => { state.extra = e.target.value; render(); });
$('fStock').addEventListener('change', e => { state.stock = e.target.checked; render(); });
$('reset').addEventListener('click', () => selectCat(state.cat));
$('chips').addEventListener('click', e => { const c = e.target.closest('.chip'); if (c) { state.sub = c.dataset.sub; render(); } });
$('vCards').addEventListener('click', () => { state.view = 'cards'; try { localStorage.setItem('dl.view', 'cards'); } catch (e) {} render(); });
$('vTable').addEventListener('click', () => { state.view = 'table'; try { localStorage.setItem('dl.view', 'table'); } catch (e) {} render(); });
$('results').addEventListener('click', e => { const th = e.target.closest('th[data-sort]'); if (!th) return; const k = th.dataset.sort; if (state.sort === k) state.dir *= -1; else { state.sort = k; state.dir = 1; } render(); });

buildNav();
let start = 'grocery-olive-oil-26';
try { const saved = localStorage.getItem('dl.cat'); if (saved && byCat[saved]) start = saved; } catch (e) {}
const hash = location.hash.slice(1);
if (hash && byCat[hash]) start = hash;
selectCat(start);
</script>
"""
html = html.replace("__DATA__", DATA_JSON).replace("__CATS__", CATS_JSON).replace("__DEPT__", DEPT_JSON)
open(OUT, "w", encoding="utf-8").write(html)
print("wrote", OUT, len(html) // 1024, "KB")
