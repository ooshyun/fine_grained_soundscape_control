#!/usr/bin/env python3
"""Scrape DeLaurenti (NitroSell) category listing pages into JSON/CSV."""
import csv, html, json, re, sys, time, os
import urllib.request

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
BASE = "https://delaurenti.com"
OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)


def fetch(url, retries=4):
    key = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:200]
    p = os.path.join(CACHE, key + ".html")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            open(p, "w", encoding="utf-8").write(body)
            return body
        except Exception as e:
            print(f"  retry {i+1} {url}: {e}", file=sys.stderr)
            time.sleep(2 ** i)
    raise RuntimeError("failed " + url)


CARD = re.compile(r'<article class="product-card" data-sku="([^"]*)">(.*?)</article>', re.S)


def clean(s):
    return html.unescape(re.sub(r"\s+", " ", s)).strip()


def parse_cards(page_html):
    items = []
    for sku, body in CARD.findall(page_html):
        m = re.search(r'<a href="([^"]+)" class="product-link productnameTitle">(.*?)</a>', body, re.S)
        if not m:
            continue
        url, name = m.group(1), clean(m.group(2))
        pid = re.search(r"-(\d+)/$", url)
        prices = re.findall(r'<span class="text-price(?:special)?">\s*\$?([\d,]+\.\d{2})', body)
        price = float(prices[0].replace(",", "")) if prices else None
        was = re.search(r'<span class="text-pricewas">\s*\$?([\d,]+\.\d{2})', body)
        out_of_stock = bool(re.search(r"out of stock|outofstock|sold out", body, re.I))
        flags = clean(re.sub(r"<[^>]+>", " ", (re.search(r'<span class="flags[^"]*">(.*?)</span>', body, re.S) or [None, ""])[1] if re.search(r'<span class="flags[^"]*">', body) else ""))
        img = re.search(r'data-src="([^"]+)"', body)
        items.append({
            "sku": sku,
            "product_id": pid.group(1) if pid else None,
            "name": name,
            "price_usd": price,
            "was_price_usd": float(was.group(1).replace(",", "")) if was else None,
            "out_of_stock": out_of_stock,
            "flags": flags,
            "url": BASE + url,
            "image": img.group(1) if img else None,
        })
    return items


def parse_facets(page_html):
    """Return {facet_name: {value: count}} from the filter sidebar."""
    facets = {}
    for m in re.finditer(r'href="https://delaurenti\.com/store/search\.asp\?[^"]*?&amp;(brand|placeoforigin|[a-z_]+)=([^"&]*)"[^>]*>(.*?)</a>\s*<span class="pfs-count">\((\d+)\)', page_html, re.S):
        facets.setdefault(m.group(1), {})[clean(m.group(3))] = int(m.group(4))
    return facets


def category_page_count(page_html):
    """1-indexed page count from the real pagination block (ignores the commented-out next-card)."""
    m = re.search(r'<div class="pagination">(.*?)</div>', page_html, re.S)
    if not m:
        return 1
    nums = [int(x) for x in re.findall(r'\?page=(\d+)', m.group(1))]
    return max(nums) if nums else 1


def search_page_count(page_html):
    """Number of 0-indexed result pages in a search.asp result (gotoResultPage(i))."""
    m = re.search(r'<div class="pagination">(.*?)</div>', page_html, re.S)
    if not m:
        return 1
    idx = [int(x) for x in re.findall(r'gotoResultPage\((\d+)\)', m.group(1))]
    return (max(idx) + 1) if idx else 1


def scrape_category(url):
    first = fetch(url)
    title = clean((re.search(r'<h2 class="department-header">(.*?)</h2>', first, re.S) or [None, ""])[1])
    last = category_page_count(first)
    items = parse_cards(first)
    for p in range(2, last + 1):
        h = fetch(f"{url}?page={p}")
        items.extend(parse_cards(h))
        time.sleep(0.5)
    # dedupe by product_id
    seen, uniq = set(), []
    for it in items:
        if it["product_id"] in seen:
            continue
        seen.add(it["product_id"])
        uniq.append(it)
    return {"category": title, "url": url, "pages": last, "count": len(uniq), "facets": parse_facets(first), "products": uniq}


if __name__ == "__main__":
    url = sys.argv[1]
    slug = url.rstrip("/").split("/")[-1]
    data = scrape_category(url)
    json.dump(data, open(os.path.join(OUT, slug + ".json"), "w"), indent=2, ensure_ascii=False)
    print(f"{data['category']}: {data['count']} products across {data['pages']} pages")
