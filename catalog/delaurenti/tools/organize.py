#!/usr/bin/env python3
"""Turn scraped DeLaurenti category JSON into organized CSV + Markdown.

usage: organize.py <out_root> <slug.json> [<slug.json> ...]
Writes <out_root>/<category-slug>/products.csv, products.json, README.md and refreshes
<out_root>/README.md + <out_root>/all_products.csv.
"""
import csv, json, os, re, sys, collections

FIELDS = ["category", "subtype", "brand", "place_of_origin", "name", "size", "size_ml",
          "price_usd", "price_per_100ml", "availability", "flags", "tagline", "sku",
          "product_id", "url", "image", "description"]

SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(ml|l|liter|litre|oz|fl\.? ?oz|g|gr|kg|lb|lbs|ct|pk|pack|pc|pcs|piece|pieces|x\s*\d+\s*ml)\b', re.I)


def parse_size(name):
    """Return (label, ml_equivalent or None)."""
    m = re.search(r'(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*(ml)\b', name, re.I)
    if m:
        n, v = int(m.group(1)), float(m.group(2))
        return f"{n} x {v:g}ml", n * v
    m = SIZE_RE.search(name)
    if not m:
        return "", None
    v, u = float(m.group(1)), m.group(2).lower().replace(" ", "").replace(".", "")
    label = f"{v:g}{m.group(2)}"
    ml = None
    if u == "ml":
        ml = v
    elif u in ("l", "liter", "litre"):
        ml = v * 1000
    elif u in ("oz", "floz"):
        ml = v * 29.5735
    return label, ml


def olive_oil_subtype(name, desc="", tagline=""):
    n = name.lower()
    t = (name + " " + tagline).lower()
    if any(k in t for k in ["top 20", "tasting set", "subscription", "gift", "trio", "collection", "set,", " set ", "tubes"]):
        return "세트/기프트 (Sets & Gift Packs)"
    if any(k in n for k in ["sesame", "algae", "walnut", "hazelnut", "avocado", "grapeseed", "pumpkin", "sunflower",
                            "canola", "pistachio", "almond oil", "argan", "cooking oil", "peanut", "coconut"]):
        return "기타 식용유 (Other Oils: nut, seed, algae)"
    if any(k in n for k in ["lemon", "chili", "chilli", "garlic", "basil", "truffle", "agrumato", "orange", "rosemary",
                            "mandarin", "bergamot", "herb", "pepper", "ginger", "lime", "tangerine", "yuzu", "smoked",
                            "porcini", "infused", "flavored", "flavoured", "with "]):
        return "향 첨가/인퓨즈드 (Flavored & Infused Olive Oils)"
    if "novello" in n or "nuovo" in n or "new harvest" in t or "2025" in n:
        return "올리오 누오보/새 수확 (Olio Nuovo / New Harvest)"
    d = (name + " " + desc).lower()
    if re.search(r'\b(dop|igp|pdo|pgi)\b', n) or "100%" in n or any(k in d for k in ["monocultivar", "single-cultivar", "single cultivar", "single-estate", "single estate", "estate-grown", "estate grown", "monovarietal", "single varietal", "single-varietal"]):
        return "EVOO – 단일 산지/품종·DOP/IGP (Estate, Single-Cultivar, DOP/IGP)"
    return "EVOO – 일반 (Extra Virgin Olive Oil)"


def generic_subtype(name, desc="", tagline=""):
    return ""


SUBTYPE_RULES = {"grocery-olive-oil-26": olive_oil_subtype}


def price_tier(p):
    if p is None:
        return "가격 미표시"
    if p < 15:
        return "$0–15"
    if p < 30:
        return "$15–30"
    if p < 50:
        return "$30–50"
    if p < 80:
        return "$50–80"
    return "$80+"


def load(path):
    d = json.load(open(path))
    side_path = path.replace(".json", ".facets.json")
    side = json.load(open(side_path)) if os.path.exists(side_path) else {}
    slug = os.path.basename(path)[:-5]
    rule = SUBTYPE_RULES.get(slug, generic_subtype)
    rows = []
    for p in d["products"]:
        extra = side.get(p["product_id"], {})
        size, ml = parse_size(p["name"])
        price = p.get("price_usd") if p.get("price_usd") is not None else p.get("detail_price_usd")
        avail = p.get("availability") or ("Out of Stock" if p.get("out_of_stock") else "In Stock")
        row = {
            "category": d["category"],
            "subtype": rule(p["name"], p.get("description", ""), p.get("tagline", "")),
            "brand": p.get("brand") or extra.get("brand", ""),
            "place_of_origin": (p.get("place_of_origin") or extra.get("place_of_origin", "")).replace("Morrocco", "Morocco"),
            "name": p["name"],
            "size": size,
            "size_ml": round(ml) if ml else "",
            "price_usd": price,
            "price_per_100ml": round(price / ml * 100, 2) if (price and ml) else "",
            "availability": avail,
            "flags": p.get("flags", ""),
            "tagline": p.get("tagline", ""),
            "sku": p["sku"],
            "product_id": p["product_id"],
            "url": p["url"],
            "image": p.get("image", ""),
            "description": p.get("description", ""),
        }
        rows.append(row)
    return d, slug, rows


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt_price(v):
    return f"${v:,.2f}" if isinstance(v, (int, float)) else ""


def product_table(rows, cols=("brand", "name", "size", "price_usd", "price_per_100ml", "place_of_origin", "availability")):
    head = {"brand": "브랜드", "name": "제품명", "size": "용량", "price_usd": "가격", "price_per_100ml": "$/100ml",
            "place_of_origin": "산지", "availability": "재고", "subtype": "유형", "tagline": "비고"}
    out = ["| " + " | ".join(head[c] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if c == "name":
                v = f"[{md_escape(v)}]({r['url']})"
            elif c in ("price_usd",):
                v = fmt_price(v)
            elif c == "price_per_100ml":
                v = f"${v:.2f}" if v != "" else ""
            elif c == "availability":
                v = "품절" if "out" in str(v).lower() else "재고 있음"
            else:
                v = md_escape(v)
            cells.append(v)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_category(out_root, d, slug, rows):
    cat_dir = os.path.join(out_root, slug)
    os.makedirs(cat_dir, exist_ok=True)
    with open(os.path.join(cat_dir, "products.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    json.dump({"category": d["category"], "url": d["url"], "count": len(rows), "facets": d.get("facets", {}),
               "products": rows}, open(os.path.join(cat_dir, "products.json"), "w"), indent=2, ensure_ascii=False)

    in_stock = sum(1 for r in rows if "out" not in r["availability"].lower())
    prices = [r["price_usd"] for r in rows if isinstance(r["price_usd"], (int, float))]
    md = [f"# {d['category']} — DeLaurenti 제품 정리", "",
          f"- 출처: <{d['url']}>", f"- 제품 수: **{len(rows)}** (재고 있음 {in_stock}, 품절 {len(rows) - in_stock})",
          f"- 가격 범위: {fmt_price(min(prices))} – {fmt_price(max(prices))} (중간값 {fmt_price(sorted(prices)[len(prices)//2])})" if prices else "",
          f"- 데이터 파일: `products.csv`, `products.json`", ""]

    has_sub = any(r["subtype"] for r in rows)
    has_origin = any(r["place_of_origin"] for r in rows)
    has_brand = any(r["brand"] for r in rows)

    # 1. Summary by subtype / origin / brand
    md += ["## 요약", ""]
    if has_sub:
        c = collections.Counter(r["subtype"] for r in rows)
        md += ["### 유형별", "", "| 유형 | 제품 수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c.most_common()] + [""]
    if has_origin:
        c = collections.Counter(r["place_of_origin"] or "(미표시)" for r in rows)
        md += ["### 산지별", "", "| 산지 | 제품 수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c.most_common()] + [""]
    c = collections.Counter(price_tier(r["price_usd"]) for r in rows)
    order = ["$0–15", "$15–30", "$30–50", "$50–80", "$80+", "가격 미표시"]
    md += ["### 가격대별", "", "| 가격대 | 제품 수 |", "|---|---|"] + [f"| {k} | {c[k]} |" for k in order if c.get(k)] + [""]
    if has_brand:
        c = collections.Counter(r["brand"] or "(미표시)" for r in rows)
        md += ["### 브랜드별 (제품 수 상위)", "", "| 브랜드 | 제품 수 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c.most_common(25)] + [""]

    # 2. Full listing grouped
    if has_sub:
        md += ["## 유형별 제품 목록", ""]
        for sub, _ in collections.Counter(r["subtype"] for r in rows).most_common():
            grp = sorted([r for r in rows if r["subtype"] == sub], key=lambda r: (r["place_of_origin"], r["brand"], r["price_usd"] or 0))
            md += [f"### {sub} ({len(grp)})", "", product_table(grp), ""]
    if has_origin:
        md += ["## 산지별 → 브랜드별 제품 목록", ""]
        for origin, _ in collections.Counter(r["place_of_origin"] or "(미표시)" for r in rows).most_common():
            grp = [r for r in rows if (r["place_of_origin"] or "(미표시)") == origin]
            md += [f"### {origin} ({len(grp)})", ""]
            for brand, _ in collections.Counter(r["brand"] or "(브랜드 미표시)" for r in grp).most_common():
                bg = sorted([r for r in grp if (r["brand"] or "(브랜드 미표시)") == brand], key=lambda r: r["price_usd"] or 0)
                md += [f"**{brand}** ({len(bg)})", "", product_table(bg, cols=("name", "size", "price_usd", "price_per_100ml", "availability")), ""]
    elif has_brand:
        md += ["## 브랜드별 제품 목록", ""]
        for brand, _ in collections.Counter(r["brand"] or "(브랜드 미표시)" for r in rows).most_common():
            bg = sorted([r for r in rows if (r["brand"] or "(브랜드 미표시)") == brand], key=lambda r: r["price_usd"] or 0)
            md += [f"### {brand} ({len(bg)})", "", product_table(bg, cols=("name", "size", "price_usd", "place_of_origin", "availability")), ""]
    else:
        md += ["## 전체 제품 목록 (가격순)", "", product_table(sorted(rows, key=lambda r: r["price_usd"] or 0)), ""]

    # 3. Price-sorted full list
    md += ["## 전체 제품 (가격 오름차순)", "", product_table(sorted(rows, key=lambda r: (r["price_usd"] is None, r["price_usd"] or 0)),
                                                       cols=("brand", "name", "size", "price_usd", "price_per_100ml", "place_of_origin", "availability")), ""]
    # 4. Notes / taglines
    tagged = [r for r in rows if r["tagline"]]
    if tagged:
        md += ["## 스토어 코멘트가 있는 제품", "", product_table(tagged, cols=("name", "tagline", "price_usd")), ""]

    open(os.path.join(cat_dir, "README.md"), "w", encoding="utf-8").write("\n".join(md))
    return {"slug": slug, "category": d["category"], "url": d["url"], "count": len(rows), "in_stock": in_stock,
            "min": min(prices) if prices else None, "max": max(prices) if prices else None}


def write_index(out_root, summaries, all_rows):
    with open(os.path.join(out_root, "all_products.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    uniq = {r["product_id"] for r in all_rows}
    dept = collections.OrderedDict()
    for s in summaries:
        dept.setdefault(s["slug"].split("-")[0], []).append(s)
    md = ["# DeLaurenti (delaurenti.com) 제품 카탈로그 정리", "",
          f"- 수집 카테고리: {len(summaries)}개", f"- 제품 행 수: {len(all_rows)} (카테고리 중복 제외 고유 제품 {len(uniq)}개)",
          "- 전체 CSV: `all_products.csv` / 카테고리별: 각 폴더의 `README.md`, `products.csv`, `products.json`", "",
          "| 부서 | 카테고리 | 제품 수 | 재고 있음 | 가격 범위 | 정리본 |", "|---|---|---|---|---|---|"]
    for dep, items in dept.items():
        for s in items:
            rng = f"{fmt_price(s['min'])} – {fmt_price(s['max'])}" if s["min"] is not None else ""
            md.append(f"| {dep.title()} | [{s['category']}]({s['url']}) | {s['count']} | {s['in_stock']} | {rng} | [{s['slug']}]({s['slug']}/README.md) |")
    open(os.path.join(out_root, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    out_root = sys.argv[1]
    os.makedirs(out_root, exist_ok=True)
    summaries, all_rows = [], []
    for path in sys.argv[2:]:
        d, slug, rows = load(path)
        summaries.append(write_category(out_root, d, slug, rows))
        all_rows += rows
        print(f"{slug}: {len(rows)}")
    write_index(out_root, summaries, all_rows)
