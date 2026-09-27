#!/usr/bin/env python3
"""Fetch product detail pages for a category JSON and enrich with description/tagline/availability."""
import json, re, sys, html, time
from scrape import fetch, clean

def parse_detail(h):
    d = {}
    m = re.search(r'<h2 class="product-detail-line">(.*?)</h2>', h, re.S)
    d["tagline"] = clean(re.sub(r"<[^>]+>", " ", m.group(1))) if m else ""
    m = re.search(r'<div class="text-product-desc">(.*?)</div>\s*</section>', h, re.S)
    desc = m.group(1) if m else ""
    desc = re.sub(r"<br\s*/?>", "\n", desc)
    desc = re.sub(r"</p>", "\n", desc)
    desc = re.sub(r"<[^>]+>", " ", desc)
    desc = html.unescape(desc)
    desc = "\n".join(l.strip() for l in desc.splitlines())
    d["description"] = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", desc)).strip()
    m = re.search(r'<div class="product-availability[^"]*">\s*<label>(.*?)</label>', h, re.S)
    d["availability"] = clean(m.group(1)) if m else ""
    m = re.search(r'<strong class="priceCurrent">\s*\$?([\d,]+\.\d{2})', h)
    d["detail_price_usd"] = float(m.group(1).replace(",", "")) if m else None
    m = re.search(r'<s class="text-pricestrike">\s*\$?([\d,]+\.\d{2})', h)
    d["strike_price_usd"] = float(m.group(1).replace(",", "")) if m else None
    m = re.search(r'name="category" value="(\d+)"', h)
    d["category_id"] = m.group(1) if m else None
    return d

if __name__ == "__main__":
    path = sys.argv[1]
    data = json.load(open(path))
    for i, p in enumerate(data["products"]):
        try:
            h = fetch(p["url"])
            p.update(parse_detail(h))
        except Exception as e:
            p["detail_error"] = str(e)
        if i % 20 == 0:
            print(f"{i}/{len(data['products'])}", flush=True)
        time.sleep(0.3)
    json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)
    print("done", path)
