#!/usr/bin/env python3
"""Paper crawler: DBLP + OpenAlex only. Every record traceable to an API response."""
import json, re, time, sys, urllib.parse, urllib.request, urllib.error, os

MAILTO = "anonymous@example.com"
UA = f"paper-crawler/1.0 (mailto:{MAILTO})"
OUT = "/path/to/repo/tex_wacv/crawl"
DBLP_HOSTS = ["dblp.uni-trier.de", "dblp.org", "dblp.dagstuhl.de"]

TOPICS = {
 "T1_video_rodent_seizure": [
   "automated seizure detection rodent video",
   "Racine scale automated behavioral seizure scoring",
   "video-based seizure detection mice deep learning",
   "automated rodent behavior monitoring epilepsy video",
   "convulsive seizure detection video neural network",
   "preclinical epilepsy video monitoring automated analysis",
   "generalized tonic-clonic seizure video detection",
 ],
 "T2_ecg_cardiac_seizure": [
   "ictal tachycardia seizure detection heart rate",
   "ECG based seizure detection wearable",
   "heart rate variability epilepsy seizure onset",
   "cardiac autonomic seizure detection algorithm",
   "seizure detection electrocardiogram deep learning",
   "ictal heart rate change epilepsy patients",
   "photoplethysmography seizure detection",
 ],
 "T3_animal_behavior_dl": [
   "DeepLabCut markerless pose estimation deep learning",
   "SLEAP multi-animal pose tracking",
   "deep learning animal behavior analysis",
   "behavioral phenotyping mice machine learning",
   "unsupervised behavior segmentation rodent motion",
   "markerless tracking animals neural network",
   "computational ethology deep learning",
 ],
 "T4_imbalance_rare_event": [
   "class imbalance deep learning medical time series",
   "focal loss dense object detection",
   "rare event detection physiological time series deep learning",
   "imbalanced data classification medical",
   "anomaly detection medical time series deep learning",
   "class imbalance video action recognition long-tailed",
   "systematic study class imbalance neural networks",
 ],
 "T5_cross_subject": [
   "cross-subject generalization EEG deep learning",
   "subject-independent evaluation biomedical machine learning",
   "inter-subject variability physiological signals model",
   "patient-independent seizure detection generalization",
   "leave-one-subject-out cross validation bias",
   "domain shift clinical machine learning external validation",
   "subject-wise cross validation neuroimaging pitfalls",
 ],
 "T6_false_alarm_deployment": [
   "false alarm rate seizure detection algorithm",
   "false positives per hour continuous EEG monitoring",
   "alarm fatigue physiological monitoring intensive care",
   "seizure detection performance metrics evaluation standards",
   "wearable seizure detection device clinical validation",
   "long-term continuous monitoring detection false detection rate",
 ],
}

def fetch(url, tries=5, sleep=2.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            code = getattr(e, "code", None)
            sys.stderr.write(f"    retry {i+1}/{tries} ({code or type(e).__name__}) {url[:90]}\n")
            time.sleep(sleep * (i + 1))
    return None

def dblp(q, h=60):
    for host in DBLP_HOSTS:
        u = f"https://{host}/search/publ/api?q={urllib.parse.quote(q)}&format=json&h={h}"
        d = fetch(u, tries=2, sleep=2.0)
        if d and "result" in d:
            return d
        time.sleep(1.0)
    return None

def openalex(q, per_page=50):
    u = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q) +
         f"&per_page={per_page}&mailto={MAILTO}"
         "&select=id,doi,title,display_name,publication_year,cited_by_count,type,"
         "authorships,primary_location,locations,biblio,ids,abstract_inverted_index,language")
    return fetch(u, tries=4, sleep=1.5)

def norm(t):
    if not t: return ""
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())

def inv2abs(inv):
    if not inv: return ""
    pairs = []
    for w, ps in inv.items():
        for p in ps: pairs.append((p, w))
    pairs.sort()
    s = " ".join(w for _, w in pairs)
    return s[:1400]

papers = {}   # norm_title -> record

def add(rec):
    k = norm(rec["title"])
    if not k or len(k) < 12: return False
    if k in papers:
        ex = papers[k]
        # merge: prefer having DOI, keep both source ids, keep max citations
        for f in ("doi", "abstract", "url", "arxiv", "volume", "pages", "venue_full"):
            if not ex.get(f) and rec.get(f): ex[f] = rec[f]
        if rec.get("cited_by_count", 0) > ex.get("cited_by_count", 0):
            ex["cited_by_count"] = rec["cited_by_count"]
        if rec["source_api"] not in ex["source_api"]:
            ex["source_api"] += "+" + rec["source_api"]
            ex["record_id"] += " | " + rec["record_id"]
        ex.setdefault("also_queries", []).append(rec["source_query"])
        return False
    papers[k] = rec
    return True

def arxiv_from(w):
    for loc in (w.get("locations") or []):
        src = (loc.get("source") or {})
        nm = (src.get("display_name") or "")
        lp = loc.get("landing_page_url") or ""
        pdf = loc.get("pdf_url") or ""
        for s in (lp, pdf):
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", s or "")
            if m: return m.group(1)
        if "arxiv" in nm.lower():
            m = re.search(r"([0-9]{4}\.[0-9]{4,5})", lp + " " + pdf)
            if m: return m.group(1)
    return ""

total_calls = 0
qi = 0
allq = [(t, q) for t, qs in TOPICS.items() for q in qs]
for topic, q in allq:
    qi += 1
    before = len(papers)
    # ---- OpenAlex ----
    d = openalex(q); total_calls += 1
    n_oa = 0
    if d and d.get("results"):
        for w in d["results"]:
            title = w.get("display_name") or w.get("title") or ""
            pl = (w.get("primary_location") or {})
            src = (pl.get("source") or {}) if pl else {}
            b = w.get("biblio") or {}
            pages = ""
            if b.get("first_page"):
                pages = b["first_page"] + ("-" + b["last_page"] if b.get("last_page") else "")
            rec = {
              "title": title,
              "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])
                          if a.get("author", {}).get("display_name")],
              "venue_full": src.get("display_name") or "",
              "year": w.get("publication_year"),
              "abstract": inv2abs(w.get("abstract_inverted_index")),
              "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
              "arxiv": arxiv_from(w),
              "volume": b.get("volume") or "",
              "pages": pages,
              "type": w.get("type") or "",
              "cited_by_count": w.get("cited_by_count", 0),
              "url": w.get("doi") or w.get("id"),
              "source_api": "openalex",
              "record_id": w.get("id", ""),
              "source_query": q,
              "topic": topic,
            }
            if add(rec): n_oa += 1
    time.sleep(0.5)
    # ---- DBLP ----
    d2 = dblp(q); total_calls += 1
    n_db = 0
    if d2:
        hits = ((d2.get("result") or {}).get("hits") or {}).get("hit") or []
        if isinstance(hits, dict): hits = [hits]
        for hh in hits:
            info = hh.get("info") or {}
            au = info.get("authors", {}).get("author", [])
            if isinstance(au, dict): au = [au]
            names = [a.get("text", "") if isinstance(a, dict) else str(a) for a in au]
            rec = {
              "title": (info.get("title") or "").rstrip("."),
              "authors": names,
              "venue_full": info.get("venue") if isinstance(info.get("venue"), str) else
                            (info.get("venue", [""])[0] if info.get("venue") else ""),
              "year": int(info["year"]) if info.get("year", "").isdigit() else None,
              "abstract": "",
              "doi": (info.get("doi") or ""),
              "arxiv": "",
              "volume": info.get("volume") or "",
              "pages": info.get("pages") or "",
              "type": info.get("type") or "",
              "cited_by_count": 0,
              "url": info.get("ee") or info.get("url") or "",
              "source_api": "dblp",
              "record_id": info.get("key", ""),
              "source_query": q,
              "topic": topic,
            }
            if add(rec): n_db += 1
    new = len(papers) - before
    print(f"[{qi}/{len(allq)}] {topic}: '{q}' -> +{new} new (oa {n_oa}, dblp {n_db}); total {len(papers)}", flush=True)
    time.sleep(1.0)

out = list(papers.values())
with open(os.path.join(OUT, "papers.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"\nTOTAL unique: {len(out)}  API calls: {total_calls}")
from collections import Counter
print("by topic:", dict(Counter(p["topic"] for p in out)))
print("by api:", dict(Counter(p["source_api"] for p in out)))
print("with DOI:", sum(1 for p in out if p.get("doi")))
