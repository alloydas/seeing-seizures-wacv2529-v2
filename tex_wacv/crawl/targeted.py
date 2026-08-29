#!/usr/bin/env python3
import json, re, time, sys, urllib.parse, urllib.request
MAILTO="anonymous@example.com"; UA=f"paper-crawler/1.0 (mailto:{MAILTO})"
P="/path/to/repo/tex_wacv/crawl/papers.json"

# (topic, openalex filter expression, sort)
Q=[
# ---- T1 video / rodent seizure ----
("T1",'title_and_abstract.search:seizure AND video AND (mice OR rodent OR rat OR mouse)'),
("T1",'title_and_abstract.search:seizure AND ("deep learning" OR "machine learning" OR "neural network") AND (mice OR rodent OR rat OR mouse) AND (video OR behavioral OR behaviour)'),
("T1",'title_and_abstract.search:racine AND (scale OR score OR scoring OR stage)'),
("T1",'title_and_abstract.search:"seizure severity" AND (automated OR automatic OR "machine learning" OR "deep learning" OR grading)'),
("T1",'title.search:seizure AND (video OR visual)'),
("T1",'title_and_abstract.search:"behavioral seizure" AND (detection OR scoring OR quantification)'),
("T1",'title_and_abstract.search:convulsive AND detection AND (video OR camera OR automated)'),
# ---- T2 ECG / cardiac ----
("T2",'title_and_abstract.search:seizure AND (ECG OR electrocardiogram OR electrocardiography)'),
("T2",'title_and_abstract.search:"ictal tachycardia"'),
("T2",'title_and_abstract.search:seizure AND "heart rate variability"'),
("T2",'title_and_abstract.search:"seizure detection" AND "heart rate"'),
("T2",'title_and_abstract.search:seizure AND (photoplethysmography OR PPG OR wristband) AND detection'),
("T2",'title_and_abstract.search:epilepsy AND cardiac AND (autonomic OR "heart rate") AND seizure'),
("T2",'title_and_abstract.search:rat AND (ECG OR "heart rate") AND seizure'),
# ---- T3 animal behavior DL ----
("T3",'title_and_abstract.search:markerless AND "pose estimation"'),
("T3",'title_and_abstract.search:(DeepLabCut OR SLEAP OR "B-SOiD" OR DeepEthogram OR MoSeq OR "SimBA")'),
("T3",'title_and_abstract.search:"animal behavior" AND ("deep learning" OR "machine learning" OR "neural network")'),
("T3",'title_and_abstract.search:"pose estimation" AND (animal OR rodent OR mice OR mouse)'),
("T3",'title_and_abstract.search:behavioral AND phenotyping AND (automated OR "machine learning") AND (mice OR rodent)'),
("T3",'title_and_abstract.search:"computational ethology"'),
("T3",'title_and_abstract.search:unsupervised AND behavior AND segmentation AND (mouse OR rodent OR animal)'),
# ---- T4 imbalance / rare event ----
("T4",'title_and_abstract.search:"focal loss"'),
("T4",'title_and_abstract.search:"class imbalance" AND (deep OR neural OR convolutional)'),
("T4",'title_and_abstract.search:("long-tailed" OR "long tailed") AND recognition'),
("T4",'title_and_abstract.search:"class imbalance" AND (medical OR clinical OR physiological OR "time series")'),
("T4",'title_and_abstract.search:"rare event" AND (detection OR classification) AND ("machine learning" OR "deep learning")'),
("T4",'title_and_abstract.search:imbalanced AND (SMOTE OR resampling OR "cost-sensitive")'),
("T4",'title_and_abstract.search:"precision-recall" AND (imbalanced OR "class distribution" OR ROC)'),
# ---- T5 cross-subject ----
("T5",'title_and_abstract.search:"cross-subject" AND (EEG OR physiological OR biosignal)'),
("T5",'title_and_abstract.search:("subject-independent" OR "patient-independent") AND (EEG OR seizure OR classification)'),
("T5",'title_and_abstract.search:"inter-subject variability" AND (EEG OR physiological OR model)'),
("T5",'title_and_abstract.search:"leave-one-subject-out"'),
("T5",'title_and_abstract.search:(subject-wise OR record-wise) AND "cross-validation"'),
("T5",'title_and_abstract.search:"external validation" AND ("machine learning" OR "prediction model") AND clinical'),
("T5",'title_and_abstract.search:"data leakage" AND ("machine learning" OR neuroimaging OR clinical)'),
("T5",'title_and_abstract.search:generalization AND ("deep learning" OR model) AND (EEG OR clinical OR medical) AND (site OR subject OR patient OR hospital)'),
# ---- T6 false alarm / deployment ----
("T6",'title_and_abstract.search:"false alarm" AND (seizure OR EEG OR monitoring)'),
("T6",'title_and_abstract.search:("false positives per hour" OR "false detections per hour" OR "false alarm rate") AND (seizure OR detection)'),
("T6",'title_and_abstract.search:"alarm fatigue"'),
("T6",'title_and_abstract.search:seizure AND detection AND (sensitivity AND specificity) AND (wearable OR ambulatory OR "long-term")'),
("T6",'title_and_abstract.search:"seizure detection" AND (validation OR "clinical trial" OR prospective) AND device'),
("T6",'title_and_abstract.search:neonatal AND seizure AND detection AND algorithm'),
]
def fetch(u,tries=5):
    for i in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept":"application/json"})
            with urllib.request.urlopen(r,timeout=45) as x: return json.loads(x.read().decode("utf-8","replace"))
        except Exception as e:
            sys.stderr.write(f"   retry{i+1} {getattr(e,'code',type(e).__name__)}\n"); time.sleep(2*(i+1))
    return None
def norm(t): return re.sub(r"[^a-z0-9]","",re.sub(r"<[^>]+>"," ",t or "").lower())
def inv2abs(inv):
    if not inv: return ""
    pr=[(p,w) for w,ps in inv.items() for p in ps]; pr.sort()
    return " ".join(w for _,w in pr)[:1500]
def arx(w):
    for loc in (w.get("locations") or []):
        for s in (loc.get("landing_page_url") or "", loc.get("pdf_url") or ""):
            m=re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",s)
            if m: return m.group(1)
    return ""
papers=json.load(open(P)); idx={norm(p["title"]):p for p in papers}
SEL=("id,doi,title,display_name,publication_year,cited_by_count,type,authorships,"
     "primary_location,locations,biblio,ids,abstract_inverted_index")
added=0
for i,(tp,filt) in enumerate(Q,1):
    n=0
    for srt in ("cited_by_count:desc","publication_year:desc"):
        u=("https://api.openalex.org/works?filter="+urllib.parse.quote(filt,safe=':,|()"')+
           f"&per_page=50&sort={srt}&select={SEL}&mailto={MAILTO}")
        d=fetch(u)
        if not d or not d.get("results"): continue
        for w in d["results"]:
            t=w.get("display_name") or ""; k=norm(t)
            if not k or len(k)<12: continue
            if k in idx:
                idx[k].setdefault("targeted_topics",[]);
                if tp not in idx[k]["targeted_topics"]: idx[k]["targeted_topics"].append(tp)
                continue
            pl=w.get("primary_location") or {}; src=(pl.get("source") or {}) if pl else {}
            b=w.get("biblio") or {}
            pg=(b.get("first_page") or "")+("-"+b["last_page"] if b.get("last_page") else "")
            rec={"title":t,"authors":[a["author"]["display_name"] for a in (w.get("authorships") or []) if a.get("author",{}).get("display_name")],
                 "venue_full":src.get("display_name") or "","year":w.get("publication_year"),
                 "abstract":inv2abs(w.get("abstract_inverted_index")),
                 "doi":(w.get("doi") or "").replace("https://doi.org/",""),"arxiv":arx(w),
                 "volume":b.get("volume") or "","pages":pg,"type":w.get("type") or "",
                 "cited_by_count":w.get("cited_by_count",0),"url":w.get("doi") or w.get("id"),
                 "source_api":"openalex","record_id":w.get("id",""),"source_query":filt,
                 "topic":"targeted_"+tp,"targeted_topics":[tp]}
            idx[k]=rec; papers.append(rec); n+=1; added+=1
        time.sleep(0.4)
    print(f"[{i}/{len(Q)}] {tp} +{n} | {filt[:74]}",flush=True)
json.dump(papers,open(P,"w"),indent=1)
print(f"\nTargeted pass added {added}; total {len(papers)}")
