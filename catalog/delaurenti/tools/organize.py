#!/usr/bin/env python3
"""Turn scraped DeLaurenti category JSON into organized CSV + Markdown.

usage: organize.py <out_root> <slug.json> [<slug.json> ...]
Writes <out_root>/<category-slug>/products.csv, products.json, README.md and refreshes
<out_root>/README.md + <out_root>/all_products.csv.
"""
import csv, json, os, re, sys, collections

EXTRA_FACETS = [("grape_variety", "품종"), ("flavor_profile", "맛 프로필"), ("milk", "우유 종류"), ("raw_or_pasteurized", "살균 여부")]
FIELDS = ["category", "subtype", "brand", "place_of_origin", "grape_variety", "flavor_profile", "milk", "raw_or_pasteurized", "name", "size", "size_ml",
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



def keyword_rules(rules, default):
    """Build a subtype classifier from an ordered list of (label, [keywords]) matched on name+tagline."""
    def f(name, desc="", tagline=""):
        t = (name + " " + tagline).lower()
        for label, kws in rules:
            if any(k in t for k in kws):
                return label
        return default
    return f


chocolate_subtype = keyword_rules([
    ("세트/기프트 (Gift & Sets)", ["gift", "set", "collection", "assort", "box of", "give "]),
    ("핫초코/코코아 (Hot Chocolate & Cocoa)", ["hot chocolate", "cocoa", "cacao powder", "drinking"]),
    ("트러플/봉봉/프랄린 (Truffles, Bonbons & Pralines)", ["truffle", "bonbon", "praline", "gianduja", "gianduia", "cremino", "caramel"]),
    ("초콜릿 코팅 견과/과일 (Chocolate-Covered Nuts & Fruit)", ["covered", "coated", "enrobed", "almond", "hazelnut", "orange peel", "cherries", "cherry", "raisin", "espresso bean", "dragee"]),
    ("스프레드 (Spreads)", ["spread", "crema", "nutella", "cream"]),
    ("초콜릿 바 (Bars)", ["bar", "tablet", "%", "dark", "milk", "white choc", "compartes", "kessho", "3oz", "tony's", "ritter", "lindt"]),
    ("사탕/캔디 (Candy & Confections)", ["candy", "toffee", "nougat", "torrone", "mint", "licorice", "gummy", "marzipan", "fudge"]),
], "기타 초콜릿 (Other)")

fish_subtype = keyword_rules([
    ("세트/컬렉션 (Sets & Collections)", ["tour", "set", "collection", "gift", "sampler"]),
    ("참치 (Tuna)", ["tuna", "ventresca", "tonno", "bonito", "atún", "atun", "albacore"]),
    ("앤초비 (Anchovies)", ["anchov", "anchoa", "alici", "boquerones", "colatura"]),
    ("정어리 (Sardines)", ["sardine", "sardinha", "sardinilla", "brisling"]),
    ("고등어 (Mackerel)", ["mackerel", "sgombro", "cavala"]),
    ("연어/송어 (Salmon & Trout)", ["salmon", "trout", "sockeye", "lox", "gravlax"]),
    ("문어/오징어/갑오징어 (Octopus, Squid & Cuttlefish)", ["octopus", "squid", "calamar", "cuttlefish", "pulpo", "polvo"]),
    ("조개/홍합/굴/가리비 (Clams, Mussels, Oysters & Scallops)", ["clam", "mussel", "oyster", "scallop", "cockle", "berberecho", "razor", "navaja", "vongole", "mejillon"]),
    ("새우/게/랍스터/성게 (Shrimp, Crab, Lobster & Urchin)", ["shrimp", "prawn", "crab", "lobster", "urchin", "langoustine", "gambas"]),
    ("어란/캐비아/보타르가 (Roe, Caviar & Bottarga)", ["caviar", "roe", "bottarga", "ikura", "tobiko", "botarga"]),
    ("대구/기타 흰살 생선 (Cod & Other Whitefish)", ["cod", "bacalao", "baccalà", "baccala", "hake", "sea bass", "sea bream", "eel", "herring", "whitefish", "smoked fish", "sprat", "seabass", "branzino", "halibut", "garfish", "rockfish", "mackeral", "smoked"]),
    ("소스/양념 (Fish Sauces & Seasonings)", ["sauce", "paste", "garum", "dashi", "furikake", "broth"]),
], "기타 해산물 (Other Seafood)")

pasta_subtype = keyword_rules([
    ("뇨끼 (Gnocchi)", ["gnocchi", "gnocchetti"]),
    ("속 채운 파스타 (Filled Pasta: Ravioli, Tortellini)", ["ravioli", "tortellini", "tortelloni", "agnolotti", "cappelletti", "mezzelune"]),
    ("쿠스쿠스/프레골라/오르조 (Couscous, Fregola & Orzo)", ["couscous", "fregola", "fregula", "orzo", "risoni", "ptitim", "israeli"]),
    ("글루텐프리/통곡물/콩 파스타 (Gluten-Free, Whole Grain & Legume)", ["gluten", "gf ", "whole wheat", "integrale", "chickpea", "lentil", "brown rice", "buckwheat", "farro", "spelt", "einkorn", "kamut"]),
    ("에그 파스타/탈리아텔레/파파르델레 (Egg Pasta & Ribbons)", ["egg", "uovo", "tagliatelle", "pappardelle", "fettuccine", "tagliolini", "tajarin", "lasagn", "nest"]),
    ("긴 파스타 (Long: Spaghetti, Linguine, Bucatini)", ["spaghett", "linguin", "bucatin", "vermicell", "capellini", "angel hair", "chitarra", "trenette", "mafald", "reginette", "pici", "troccoli", "tonnarelli", "fusilli lunghi", "candele", "bigoli", "fettucin", "sheets", "sfoglia"]),
    ("짧은 파스타 (Short: Rigatoni, Penne, Orecchiette)", ["rigaton", "penne", "orecchiette", "paccheri", "casarecce", "fusill", "conchiglie", "cavatelli", "strozzapreti", "trofie", "gemelli", "mezze", "maccheron", "calamarata", "tubetti", "ditali", "farfalle", "campanelle", "gigli", "radiatori", "lumac", "garganelli", "malloreddus", "busiate", "cavatappi", "ziti", "sedani", "creste", "torchio", "schiaffoni", "pasta mista", "cannelloni", "manicotti", "shells", "elbow", "trottole", "vesuvio", "foglie", "gramigna", "pipe", "anelli", "caserecce", "casareccia", "trecce", "cascatelli", "cannolicchi", "radiatore", "riccia", "buchi", "rotolini", "rotini", "rotelle", "ruote"]),
    ("수프용/작은 파스타 (Soup Pasta & Small Shapes)", ["stelline", "acini", "pastina", "ditalini", "quadrucci", "soup", "alphabet", "fideos", "fideo", "grattini", "passatelli"]),
    ("아시아/기타 면 (Noodles: Asian & Other)", ["noodle", "ramen", "udon", "soba", "rice stick", "spätzle", "spaetzle", "pierogi", "dumpling"]),
    ("파스타 소스/키트 (Pasta Sauces & Kits)", ["sauce", "sugo", "pesto", "ragu", "ragù", "kit", "carbonara", "amatriciana", "course", "class"]),
], "기타 파스타 (Other Pasta)")

vinegar_subtype = keyword_rules([
    ("발사믹 트래디지오날레 DOP (Traditional Balsamic DOP)", ["tradizionale", "traditional balsamic", "dop", "affinato", "extravecchio", "extra vecchio", "25 year", "12 year"]),
    ("발사믹/콘디멘토 (Balsamic & Condimento)", ["balsam", "condimento", "saba", "sapa", "mosto", "vincotto", "aceto"]),
    ("셰리/와인 식초 (Sherry & Wine Vinegar)", ["sherry", "jerez", "red wine", "white wine", "wine vinegar", "champagne", "banyuls", "chardonnay", "cabernet", "moscatel", "vin", "barolo", "chianti"]),
    ("사과/과일 식초 (Cider & Fruit Vinegar)", ["cider", "apple", "fig", "raspberry", "pear", "quince", "pomegranate", "cherry", "peach", "fruit", "honey", "date", "plum", "mango", "blueberry", "citrus", "lemon", "orange", "yuzu"]),
    ("쌀/맥아/기타 식초 (Rice, Malt & Other Vinegar)", ["rice", "malt", "black vinegar", "chinkiang", "coconut", "ume", "tarragon", "herb", "garlic", "chili", "verjus", "verjuice", "shrub", "drinking vinegar", "vinaigrette", "dressing"]),
], "기타 식초 (Other Vinegar)")

salt_spice_subtype = keyword_rules([
    ("소금 (Salt)", ["salt", "sale", "sel ", "fleur de sel", "flake", "fiore", "salz", "flor de sal", "salt blend"]),
    ("고추/칠리/파프리카 (Chili, Pepper Flakes & Paprika)", ["chili", "chile", "chilli", "calabrian", "espelette", "aleppo", "urfa", "paprika", "pimenton", "pimentón", "gochugaru", "cayenne", "peperoncino", "crushed red", "harissa", "berbere"]),
    ("후추 (Peppercorns)", ["pepper", "peppercorn", "tellicherry", "kampot", "sichuan", "szechuan", "sansho", "cubeb"]),
    ("시즈닝 블렌드/러브 (Seasoning Blends & Rubs)", ["seasoning", "blend", "rub", "za'atar", "zaatar", "dukkah", "everything", "furikake", "shichimi", "togarashi", "curry", "garam masala", "ras el hanout", "baharat", "herbes de provence", "italian seasoning", "bbq", "spice mix", "mix", "spice bomb", "grind", "shawarma", "dipping powder", "bruschetta", "spaghettata", "sixer", "sauce"]),
    ("사프란/바닐라/고급 향신료 (Saffron, Vanilla & Premium Spices)", ["saffron", "vanilla", "cardamom", "truffle", "sumac", "mahlab", "mastic", "juniper", "star anise"]),
    ("허브 (Dried Herbs)", ["oregano", "origano", "thyme", "rosemary", "basil", "bay", "sage", "marjoram", "herb", "fennel", "dill", "lavender", "tarragon", "mint"]),
    ("단일 향신료 (Single Spices)", ["cinnamon", "cumin", "coriander", "nutmeg", "clove", "turmeric", "ginger", "mustard", "anise", "caraway", "fenugreek", "allspice", "garlic", "onion", "sesame", "poppy", "nigella", "mace", "asafoetida", "annatto", "celery", "mustard seed"]),
], "기타 소금·향신료 (Other)")

olives_capers_subtype = keyword_rules([
    ("케이퍼/케이퍼베리 (Capers & Caperberries)", ["caper", "cucunci", "caperberr"]),
    ("올리브 페이스트/타프나드 (Olive Paste & Tapenade)", ["tapenade", "paste", "pate", "spread", "crema", "pesto"]),
    ("속 채운 올리브 (Stuffed Olives)", ["stuffed", "farcite", "ripiene", "filled"]),
    ("칵테일/마티니 올리브 (Cocktail & Martini Olives)", ["cocktail", "martini", "vermouth", "dirty"]),
    ("그린 올리브 (Green Olives)", ["castelvetrano", "nocellara", "cerignola", "green", "verdi", "verde", "lucques", "picholine", "manzanilla", "gordal", "halkidiki", "chalkidiki", "sevillano", "arbequina"]),
    ("블랙/다크 올리브 (Black & Dark Olives)", ["kalamata", "black", "nere", "taggiasca", "taggiasche", "gaeta", "niçoise", "nicoise", "leccino", "thassos", "throuba", "beldi", "oil-cured", "oil cured", "dry-cured", "dry cured", "nyons"]),
    ("믹스/기타 올리브 (Mixed & Other Olives)", ["mix", "mixed", "assort", "medley", "olive", "olives"]),
], "기타 (Other)")

condiments_subtype = keyword_rules([
    ("머스타드 (Mustard)", ["mustard", "moutarde", "senape", "dijon", "mostarda"]),
    ("마요네즈/아이올리 (Mayonnaise & Aioli)", ["mayo", "mayonnaise", "aioli", "alioli", "kewpie"]),
    ("핫소스/칠리 소스 (Hot Sauce & Chili Crisp)", ["hot sauce", "chili crisp", "chilli crisp", "chili oil", "chile crisp", "sriracha", "harissa", "sambal", "gochujang", "calabrian chili", "bomba", "crunch", "salsa macha", "yuzu kosho", "pepper sauce", "hot honey", "chili sauce", "chile sauce", "crushed hot chili", "bonache", "habanero", "chipotle", "hatch", "piri"]),
    ("케첩/바비큐/스테이크 소스 (Ketchup, BBQ & Steak Sauce)", ["ketchup", "bbq", "barbecue", "steak sauce", "worcestershire", "worcstershire", "hp sauce", "a1", "brown sauce"]),
    ("페스토/타프나드/스프레드 (Pesto, Tapenade & Spreads)", ["pesto", "tapenade", "spread", "crema", "pate", "paté", "pâté", "dip", "hummus", "muhammara", "romesco", "bagna cauda", "ajvar", "bomba calabrese", "garlic paste", "culinary paste", "bruschetta", "muffuletta"]),
    ("피클/렐리시/처트니 (Pickles, Relish & Chutney)", ["pickle", "relish", "chutney", "giardiniera", "cornichon", "kimchi", "sauerkraut", "pickled", "piccalilli", "mostarda", "preserved lemon", "escabeche"]),
    ("아시아 소스/간장/미소 (Soy, Miso, Fish Sauce & Asian Sauces)", ["soy", "shoyu", "tamari", "miso", "fish sauce", "ponzu", "mirin", "hoisin", "oyster sauce", "teriyaki", "tonkatsu", "dashi", "yuzu", "sesame sauce", "tahini", "tahina"]),
    ("드레싱/비네그레트 (Dressings & Vinaigrettes)", ["dressing", "vinaigrette", "caesar", "ranch"]),
    ("글레이즈/시럽/감미 조미료 (Glazes, Syrups & Sweet Condiments)", ["glaze", "syrup", "saba", "vincotto", "honey", "jam", "marmalade", "agrodolce", "caramel", "jelly", "molasses"]),
    ("소스/그레이비 (Sauces & Gravies: pan, wine, demi)", ["sauce", "gravy", "demi", "jus", "salsa", "chimichurri", "mojo", "béarnaise", "bearnaise", "hollandaise", "tzatziki", "mole", "pipian", "xilli"]),
], "기타 조미료 (Other Condiments)")

cookies_subtype = keyword_rules([
    ("세트/기프트 (Gift & Sets)", ["gift", "set", "collection", "tin", "assort", "box"]),
    ("비스코티/칸투치/이탈리아 쿠키 (Biscotti, Cantucci & Italian Cookies)", ["biscotti", "cantucci", "cantuccini", "amaretti", "baci", "krumiri", "savoiardi", "ladyfinger", "pizzelle", "brutti", "ricciarelli", "cavallucci", "ossi", "frollini", "canestrelli", "baiocchi", "pan di stelle", "gocciole", "mulino", "butterhorn", "lady finger", "brigidini", "sbrisolona", "shorties"]),
    ("웨이퍼/버터 쿠키/쇼트브레드 (Wafers, Butter Cookies & Shortbread)", ["wafer", "butter cookie", "shortbread", "sable", "sablé", "galette", "palmier", "speculoos", "biscoff", "gavotte", "crepe dentelle", "petit beurre", "langue de chat", "walkers", "loacker", "quadratini", "digestive", "hobnob", "mcvitie", "bahlsen", "leibniz", "pim's", "pims", "lu "]),
    ("초콜릿 쿠키/브라우니 (Chocolate Cookies & Brownies)", ["chocolate chip", "brownie", "chocolate cookie", "cookie dough", "tate's", "tates"]),
    ("케이크/파네토네/판포르테 (Cakes, Panettone & Panforte)", ["panettone", "pandoro", "panforte", "cake", "colomba", "torta", "madeleine", "financier", "loaf", "babka", "stollen", "kouign", "canelé", "cannele", "baba"]),
    ("토로네/누가/캔디 (Torrone, Nougat & Candy)", ["torrone", "nougat", "candy", "caramel", "toffee", "licorice", "liquorice", "marzipan", "pastiglie", "leone", "drops", "lollipop", "gummy", "mint", "brittle", "croccante", "jelly", "pâte de fruit", "pate de fruit", "calisson", "dragee", "dragées", "confetti", "marshmallow", "sugar", "swedish fish", "gold coins", "licquorice", "candied", "bonbon", "chococherr", "chocohigo", "orange delights", "prunes"]),
    ("메렝게/마카롱/페이스트리 (Meringue, Macarons & Pastries)", ["meringue", "macaron", "pastr", "croissant", "sfogliatine", "cannoli", "puff", "sfoglia", "strudel", "tart", "crostata", "pie", "baklava", "cream puff"]),
    ("잼/스프레드/토핑 (Sweet Spreads & Toppings)", ["spread", "crema", "gianduja", "hazelnut", "honey", "jam", "curd", "dulce", "sprinkle", "topping", "syrup", "fudge sauce", "peanut butter sauce", "nutella", "pistachio cream", "hot chocolate"]),
    ("쿠키 일반 (Cookies & Biscuits)", ["cookie", "biscuit", "biscotto", "cracker", "snap", "thin", "crisp", "bar"]),
], "기타 과자 (Other Sweets)")

SUBTYPE_RULES = {
    "grocery-olive-oil-26": olive_oil_subtype,
    "grocery-chocolate-20": chocolate_subtype,
    "grocery-fish-19": fish_subtype,
    "grocery-pasta-27": pasta_subtype,
    "grocery-vinegar-90": vinegar_subtype,
    "grocery-salt-spices-29": salt_spice_subtype,
    "grocery-olives-capers-18": olives_capers_subtype,
    "grocery-condiments-22": condiments_subtype,
    "grocery-cookies-sweets-23": cookies_subtype,
}


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
            "grape_variety": extra.get("grape_variety", ""),
            "flavor_profile": extra.get("flavor_profile", ""),
            "milk": extra.get("milk", ""),
            "raw_or_pasteurized": extra.get("raw_or_pasteurized", ""),
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
            "place_of_origin": "산지", "availability": "재고", "subtype": "유형", "tagline": "비고",
            "grape_variety": "품종", "flavor_profile": "맛 프로필", "milk": "우유", "raw_or_pasteurized": "살균"}
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
    extras = [(k, lab) for k, lab in EXTRA_FACETS if any(r[k] for r in rows)]
    extra_cols = tuple(k for k, _ in extras)
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
    for k, lab in extras:
        c = collections.Counter(r[k] or "(미표시)" for r in rows)
        md += [f"### {lab}별", "", f"| {lab} | 제품 수 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in c.most_common()] + [""]
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
                md += [f"**{brand}** ({len(bg)})", "", product_table(bg, cols=("name", "size", "price_usd", "price_per_100ml") + extra_cols + ("availability",)), ""]
    elif has_brand:
        md += ["## 브랜드별 제품 목록", ""]
        for brand, _ in collections.Counter(r["brand"] or "(브랜드 미표시)" for r in rows).most_common():
            bg = sorted([r for r in rows if (r["brand"] or "(브랜드 미표시)") == brand], key=lambda r: r["price_usd"] or 0)
            md += [f"### {brand} ({len(bg)})", "", product_table(bg, cols=("name", "size", "price_usd", "place_of_origin") + extra_cols + ("availability",)), ""]
    else:
        md += ["## 전체 제품 목록 (가격순)", "", product_table(sorted(rows, key=lambda r: r["price_usd"] or 0)), ""]

    # 3. Price-sorted full list
    md += ["## 전체 제품 (가격 오름차순)", "", product_table(sorted(rows, key=lambda r: (r["price_usd"] is None, r["price_usd"] or 0)),
                                                       cols=("brand", "name", "size", "price_usd", "price_per_100ml", "place_of_origin") + extra_cols + ("availability",)), ""]
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
          "- 전체 CSV: `all_products.csv` / 카테고리별: 각 폴더의 `README.md`, `products.csv`, `products.json`",
          "- 수집 방법: 카테고리 목록 페이지 전부 + 각 제품 상세 페이지(설명, 스토어 코멘트, 재고) + 스토어 필터(브랜드, 산지, 품종, 우유 종류 등). 도구는 `tools/` 참고.",
          "- 열 설명: `subtype`(키워드 기반 유형 분류, 올리브 오일·초콜릿·생선·파스타·식초·소금/향신료·올리브/케이퍼·조미료·쿠키에만 적용), `size`/`size_ml`(제품명에서 추출), `price_per_100ml`(용량이 ml/L/oz로 파싱된 경우만).", "",
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
