#!/usr/bin/env python3
"""Map products -> facet values (brand, place_of_origin, ...) using the store's filter search URLs."""
import json, re, sys, html, time
from scrape import fetch, parse_cards, clean, search_page_count

FACET_LINK = re.compile(r'href="(https://delaurenti\.com/store/search\.asp\?[^"]*?&amp;([a-z_]+)=[^"&]*)"[^>]*>(.*?)</a>\s*<span class="pfs-count">\((\d+)\)', re.S)

def facet_products(url):
    h = fetch(url)
    items = parse_cards(h)
    for p in range(1, search_page_count(h)):
        sep = "&" if "?" in url else "?"
        items.extend(parse_cards(fetch(f"{url}{sep}page={p}")))
        time.sleep(0.3)
    return {it["product_id"] for it in items}

if __name__ == "__main__":
    path = sys.argv[1]
    data = json.load(open(path))
    first = fetch(data["url"])
    mapping = {}  # product_id -> {facet: [values]}
    links = FACET_LINK.findall(first)
    print(f"{len(links)} facet links")
    for href, facet, label, count in links:
        href = html.unescape(href)
        label = clean(label)
        ids = facet_products(href)
        for pid in ids:
            mapping.setdefault(pid, {}).setdefault(facet, []).append(label)
        time.sleep(0.3)
    side = {pid: {k: "; ".join(sorted(set(v))) for k, v in m.items()} for pid, m in mapping.items()}
    out = path.replace(".json", ".facets.json")
    json.dump(side, open(out, "w"), indent=2, ensure_ascii=False)
    n = sum(1 for p in data["products"] if side.get(p["product_id"], {}).get("brand"))
    print(f"done {out}: brand mapped for {n}/{len(data['products'])}")
