"""Suu tam anh cay than go don le tu Wikimedia Commons (giay phep mo).

Quy trinh:
  1. Duyet cac category "Solitary trees ..." (cay moc don le, chup ngang, ca cay nam giua anh).
  2. Lay thong tin file (kich thuoc, mime, tac gia, giay phep, category) qua MediaWiki API.
  3. Loc so bo theo quy tac (ti le khung hinh, kich thuoc, tu khoa loai tru).
  4. Tai ban thu nho 1024px ve data/raw/ va ghi data/raw_manifest.csv.
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config

API = "https://commons.wikimedia.org/w/api.php"

# Category goc: (ten category, do sau duyet subcat)
SEED_CATEGORIES = [
    ("Category:Solitary Quercus robur", 0),
    ("Category:Solitary Betula pendula", 0),
    ("Category:Solitary Pinus sylvestris", 0),
    ("Category:Solitary Juglans regia", 0),
    ("Category:Solitary Alnus glutinosa", 0),
    ("Category:Solitary Populus alba", 0),
    ("Category:Solitary Populus × canadensis", 0),
    ("Category:Quality images of solitary trees", 0),
    ("Category:Solitary trees", 0),
    ("Category:Solitary trees by country", 2),
    ("Category:Named individual trees by species", 1),
    ("Category:Quality images of trees by species", 1),
]

# Chi (genus) cay than go -> nhom hinh thai
CONIFER = {"Pinus", "Picea", "Abies", "Larix", "Cedrus", "Cupressus", "Taxus", "Juniperus", "Sequoia",
           "Sequoiadendron", "Araucaria", "Thuja", "Metasequoia", "Taxodium", "Pseudotsuga", "Tsuga", "Cryptomeria"}
BROADLEAF = {"Quercus", "Betula", "Fagus", "Acer", "Tilia", "Salix", "Populus", "Juglans", "Alnus", "Fraxinus",
             "Platanus", "Ulmus", "Adansonia", "Ficus", "Eucalyptus", "Olea", "Castanea", "Aesculus", "Prunus",
             "Malus", "Pyrus", "Sorbus", "Carpinus", "Robinia", "Acacia", "Vachellia", "Magnolia", "Liriodendron",
             "Ginkgo", "Ceiba", "Delonix", "Tamarindus", "Mangifera", "Terminalia", "Morus", "Celtis", "Catalpa",
             "Paulownia", "Cercis", "Gleditsia", "Nothofagus", "Erythrina", "Jacaranda", "Schinus", "Ailanthus",
             "Zelkova", "Carya", "Liquidambar", "Cornus", "Crataegus", "Dracaena", "Bombax", "Samanea", "Albizia",
             "Faidherbia", "Balanites", "Ziziphus", "Ilex", "Arbutus", "Cinnamomum", "Camphora", "Lagerstroemia"}
PALM = {"Cocos", "Phoenix", "Washingtonia", "Roystonea", "Borassus", "Elaeis", "Arecaceae", "Hyphaene", "Syagrus"}
GENUS_RE = re.compile(r"\b(" + "|".join(sorted(CONIFER | BROADLEAF | PALM)) + r")\b")

BAD_WORDS = ["map", "diagram", "leaf", "leaves", "bark", "trunk", "detail", "flower", "blossom", "fruit", "seed",
             "cone", "close", "macro", "stump", "plantation", "panorama", "forest", "avenue", "alley", "allee",
             "bonsai", "herbarium", "illustration", "drawing", "painting", "sketch", "nursery", "sapling",
             "seedling", "root", "timber", "log ", "interior", "aerial", "drone", "palm", "plaque", "sign",
             "sculpture", "statue", "graffiti", "engraving", "poster", "stamp", "coin", "book"]
BAD_CATS = ["in art", "drawing", "painting", "diagram", "aerial", "palm", "art of", "illustration", "plaque",
            "sculpture", "postcard", "stamps", "engraving", "bark", "leaves", "trunk", "detail", "flowers", "fruit",
            "cones", "book", "postage", "photomontage", "collage"]

MAX_FILES = 1000          # so anh tho toi da tai ve (sau loc / khu trung lap con >= 500)


def source_priority(cats: list[str]) -> int:
    """0 = category cay don le theo loai; 1 = anh chat luong cay don le; 2 = cay don le (chung / theo nuoc);
    3 = cay noi tieng theo loai; 4 = anh chat luong theo loai (kem thuan hon)."""
    joined = " | ".join(cats)
    if "Category:Solitary " in joined and "Solitary trees" not in joined.replace("Solitary trees by", ""):
        return 0
    if "Quality images of solitary trees" in joined:
        return 1
    if "Solitary trees" in joined:
        return 2
    if "named individual trees" in joined or "named specimens" in joined:
        return 3
    return 4


MIN_SIDE = 640
ASPECT_RANGE = (0.65, 1.6)  # rong/cao cua anh goc


def _api(params: dict) -> dict:
    params = dict(params, format="json")
    data = urllib.parse.urlencode(params).encode()  # POST: tranh loi 414 khi titles qua dai
    for attempt in range(4):
        try:
            req = urllib.request.Request(API, data=data, headers={"User-Agent": config.USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def list_members(cat: str, cmtype: str) -> list[str]:
    out, cont = [], {}
    while True:
        d = _api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": cmtype,
                  "cmlimit": "500", **cont})
        out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
        if "continue" in d:
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
        else:
            return out


def crawl(seed_categories=SEED_CATEGORIES) -> dict[str, list[str]]:
    """Tra ve {file_title: [category, ...]} cho moi file tim duoc."""
    files: dict[str, list[str]] = {}
    seen_cats: set[str] = set()

    def visit(cat: str, depth: int):
        if cat in seen_cats:
            return
        seen_cats.add(cat)
        low = cat.lower()
        if any(b in low for b in BAD_CATS):
            return
        for t in list_members(cat, "file"):
            files.setdefault(t, []).append(cat)
        if depth > 0:
            for sub in list_members(cat, "subcat"):
                visit(sub, depth - 1)

    for cat, depth in seed_categories:
        visit(cat, depth)
        print(f"[crawl] {cat}: tong file toi nay = {len(files)}", flush=True)
    return files


def fetch_info(titles: list[str]) -> dict[str, dict]:
    """Lay imageinfo + categories cho danh sach file (theo lo 50)."""
    info: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        params = {"action": "query", "titles": "|".join(batch), "prop": "imageinfo|categories",
                  "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1024",
                  "iiextmetadatafilter": "Artist|LicenseShortName|ObjectName|Credit", "cllimit": "500"}
        cont: dict = {}
        while True:
            d = _api({**params, **cont})
            for p in d.get("query", {}).get("pages", {}).values():
                rec = info.setdefault(p["title"], {"pageid": p.get("pageid"), "categories": []})
                if "imageinfo" in p:
                    rec["imageinfo"] = p["imageinfo"][0]
                rec["categories"] += [c["title"] for c in p.get("categories", [])]
            if "continue" in d:
                cont = {k: v for k, v in d["continue"].items() if k != "continue"}
            else:
                break
        if (i // 50) % 5 == 0:
            print(f"[info] {min(i + 50, len(titles))}/{len(titles)}", flush=True)
    return info


def detect_genus(title: str, cats: list[str]) -> tuple[str, str]:
    text = " ".join(cats) + " " + title
    m = GENUS_RE.findall(text)
    if not m:
        return "", "unknown"
    genus = max(set(m), key=m.count)  # genus xuat hien nhieu nhat
    group = "conifer" if genus in CONIFER else "palm" if genus in PALM else "broadleaf"
    return genus, group


def accept(title: str, rec: dict) -> tuple[bool, str]:
    ii = rec.get("imageinfo")
    if not ii:
        return False, "no-imageinfo"
    if ii.get("mime") not in ("image/jpeg", "image/png"):
        return False, "mime"
    w, h = ii.get("width", 0), ii.get("height", 0)
    if min(w, h) < MIN_SIDE:
        return False, "small"
    if not (ASPECT_RANGE[0] <= w / h <= ASPECT_RANGE[1]):
        return False, "aspect"
    low = title.lower()
    if any(b in low for b in BAD_WORDS):
        return False, "badword"
    catlow = " ".join(rec.get("categories", [])).lower()
    if any(b in catlow for b in BAD_CATS):
        return False, "badcat"
    _genus, group = detect_genus(title, rec.get("categories", []))
    if group == "palm":
        return False, "palm"
    return True, "ok"


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return False


def run(max_files: int | None = None, workers: int = 6) -> Path:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.CACHE_DIR / "commons_info.json"
    if cache.exists():
        info = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[cache] dung lai thong tin {len(info)} file", flush=True)
    else:
        crawl_cache = config.CACHE_DIR / "commons_crawl.json"
        if crawl_cache.exists():
            files = json.loads(crawl_cache.read_text(encoding="utf-8"))
        else:
            files = crawl()
            crawl_cache.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
        info = fetch_info(sorted(files))
        for t, cats in files.items():
            info.setdefault(t, {}).setdefault("categories", [])
            info[t]["categories"] = sorted(set(info[t]["categories"] + cats))
        cache.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    rejected: dict[str, int] = {}
    accepted = []
    for title, rec in sorted(info.items()):
        ok, why = accept(title, rec)
        if not ok:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        accepted.append((title, rec))
    print(f"[filter] chap nhan {len(accepted)} / {len(info)}; loai: {rejected}", flush=True)
    # Uu tien nguon "thuan" (cay don le) truoc, category tong quat sau; gioi han so luong tai ve
    accepted.sort(key=lambda tr: (source_priority(tr[1].get("categories", [])), tr[0]))
    max_files = max_files or MAX_FILES
    accepted = accepted[:max_files]
    prio = {}
    for _t, rec in accepted:
        pr = source_priority(rec.get("categories", []))
        prio[pr] = prio.get(pr, 0) + 1
    print(f"[filter] giu {len(accepted)} anh; theo muc uu tien nguon: {dict(sorted(prio.items()))}", flush=True)

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {}
        for idx, (title, rec) in enumerate(accepted, start=1):
            ii = rec["imageinfo"]
            url = ii.get("thumburl") or ii["url"]
            dest = config.RAW_DIR / f"{idx:05d}.jpg"
            futs[ex.submit(download, url, dest)] = (idx, title, rec, url, dest)
        done = 0
        for fut in as_completed(futs):
            idx, title, rec, url, dest = futs[fut]
            done += 1
            if done % 50 == 0:
                print(f"[download] {done}/{len(futs)}", flush=True)
            if not fut.result():
                continue
            ii = rec["imageinfo"]
            md = ii.get("extmetadata", {})
            genus, group = detect_genus(title, rec.get("categories", []))
            rows.append({
                "raw_id": idx, "raw_file": dest.name, "commons_title": title,
                "page_url": ii.get("descriptionurl", ""), "download_url": url,
                "orig_width": ii.get("width"), "orig_height": ii.get("height"),
                "author": _strip_html(md.get("Artist", {}).get("value", ""))[:120],
                "license": md.get("LicenseShortName", {}).get("value", ""),
                "genus": genus, "group": group,
                "categories": "|".join(rec.get("categories", [])),
            })
    rows.sort(key=lambda r: r["raw_id"])
    with open(config.RAW_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] da tai {len(rows)} anh -> {config.RAW_DIR}; manifest: {config.RAW_MANIFEST}", flush=True)
    return config.RAW_MANIFEST


if __name__ == "__main__":
    run()
