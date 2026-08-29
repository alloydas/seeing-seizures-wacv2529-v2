#!/usr/bin/env python3
import json, re, time, sys, urllib.parse, urllib.request, os
UA = "paper-crawler/1.0 (mailto:anonymous@example.com)"
HOSTS = ["dblp.org", "dblp.uni-trier.de", "dblp.dagstuhl.de"]
QUERIES = [
 "seizure detection video","epileptic seizure detection deep learning","rodent behavior video analysis",
 "Racine seizure severity","mouse behavior recognition video","animal pose estimation deep learning",
 "DeepLabCut","SLEAP animal pose","markerless pose tracking","behavior classification rodent",
 "ECG seizure detection","heart rate seizure","electrocardiogram deep learning arrhythmia",
 "photoplethysmography seizure","wearable seizure detection",
 "focal loss object detection","class imbalance convolutional neural network","long-tailed recognition",
 "imbalanced classification deep learning survey","rare event detection time series",
 "cross-subject EEG classification","subject-independent EEG","domain adaptation EEG",
 "leave-one-subject-out evaluation","external validation clinical prediction model",
 "false alarm reduction ICU","alarm fatigue monitoring","EEG monitoring false positive",
 "action recognition video 3D convolution","video understanding transformer",
 "epilepsy machine learning","seizure prediction EEG","neonatal seizure detection",
 "animal behavior deep learning","behavioral phenotyping automated",
]
def fetch(url, tries=6):
    for i in range(tries):
        for host in HOSTS:
            u = url.replace("HOSTPLACEHOLDER", host)
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=40) as r:
                    return json.loads(r.read().decode("utf-8","replace"))
            except Exception as e:
                sys.stderr.write(f"  fail {host} ({getattr(e,'code',type(e).__name__)})\n")
                time.sleep(1.5)
        time.sleep(3*(i+1))
    return None
def norm(t):
    return re.sub(r"[^a-z0-9]","",re.sub(r"<[^>]+>"," ",t or "").lower())

P = "/path/to/repo/tex_wacv/crawl/papers.json"
papers = json.load(open(P))
idx = {norm(p["title"]): p for p in papers}
added = 0
for qi,q in enumerate(QUERIES,1):
    url = f"https://HOSTPLACEHOLDER/search/publ/api?q={urllib.parse.quote(q)}&format=json&h=80"
    d = fetch(url)
    n=0
    if d:
        hits = ((d.get("result") or {}).get("hits") or {}).get("hit") or []
        if isinstance(hits, dict): hits=[hits]
        for hh in hits:
            info = hh.get("info") or {}
            t = (info.get("title") or "").rstrip(".")
            k = norm(t)
            if not k or len(k)<12: continue
            au = info.get("authors",{}).get("author",[])
            if isinstance(au,dict): au=[au]
            names=[a.get("text","") if isinstance(a,dict) else str(a) for a in au]
            ven = info.get("venue")
            if isinstance(ven,list): ven = ven[0] if ven else ""
            if k in idx:
                ex = idx[k]
                if "dblp" not in ex["source_api"]:
                    ex["source_api"] += "+dblp"
                    ex["record_id"] += " | " + info.get("key","")
                if not ex.get("dblp_key"): ex["dblp_key"] = info.get("key","")
                if not ex.get("doi") and info.get("doi"): ex["doi"]=info["doi"]
                if not ex.get("venue_dblp"): ex["venue_dblp"]=ven or ""
                continue
            rec = {"title":t,"authors":names,"venue_full":ven or "","venue_dblp":ven or "",
                   "year": int(info["year"]) if str(info.get("year","")).isdigit() else None,
                   "abstract":"","doi":info.get("doi",""),"arxiv":"","volume":info.get("volume",""),
                   "pages":info.get("pages",""),"type":info.get("type",""),"cited_by_count":0,
                   "url":info.get("ee") or info.get("url",""),"source_api":"dblp",
                   "record_id":info.get("key",""),"dblp_key":info.get("key",""),
                   "source_query":q,"topic":"dblp_pass"}
            idx[k]=rec; papers.append(rec); n+=1; added+=1
    print(f"[{qi}/{len(QUERIES)}] '{q}' -> +{n} (total {len(papers)})", flush=True)
    time.sleep(1.2)
json.dump(papers, open(P,"w"), indent=1)
print(f"\nDBLP pass added {added}. Total now {len(papers)}")
from collections import Counter
print("by api:", dict(Counter(p["source_api"] for p in papers)))
