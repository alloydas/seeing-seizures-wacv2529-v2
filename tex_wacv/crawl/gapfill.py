import json,re,time,urllib.parse,urllib.request,sys
MAILTO="anonymous@example.com";UA=f"paper-crawler/1.0 (mailto:{MAILTO})"
P="/path/to/repo/tex_wacv/crawl/papers.json"
GAPS=[
 'title.search:"systematic study of the class imbalance problem"',
 'title.search:"class-balanced loss"',
 'title_and_abstract.search:"standards for testing and clinical validation of seizure detection devices"',
 'title_and_abstract.search:Beniczky AND seizure detection AND (standards OR validation OR guideline)',
 'title.search:"subject-wise" OR "record-wise"',
 'title_and_abstract.search:"approximate the use-case" OR ("cross-validation" AND "clinical machine learning" AND leakage)',
 'title.search:DeepEthogram',
 'title_and_abstract.search:"motion sequencing" OR "MoSeq" AND behavior',
 'title.search:"Mapping Sub-Second Structure in Mouse Behavior"',
 'title.search:racine AND (motor OR kindling OR seizure)',
 'title_and_abstract.search:"modification of seizure activity by electrical stimulation"',
 'title.search:"A Closer Look at Memorization" OR "Rethinking the Value"',
 'title_and_abstract.search:"tonic-clonic" AND video AND detection AND (deep OR machine OR automated)',
 'title_and_abstract.search:"seizure detection" AND (rodent OR mice OR rat) AND (EEG OR video) AND (deep learning OR machine learning)',
 'title_and_abstract.search:"sudden unexpected death in epilepsy" AND (monitoring OR detection OR wearable)',
 'title_and_abstract.search:"action recognition" AND (medical OR clinical OR patient) AND video AND deep',
 'title_and_abstract.search:"data leakage" AND ("machine learning" OR neuroimaging)',
 'title_and_abstract.search:"pig" OR "swine" OR "canine" AND seizure detection video',
 'title.search:"Kinetics" AND "Human Action Video Dataset"',
 'title_and_abstract.search:"gait" OR "behavior" AND rodent AND "deep learning" AND classification AND video',
]
def fetch(u,tries=4):
    for i in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept":"application/json"})
            with urllib.request.urlopen(r,timeout=45) as x: return json.loads(x.read().decode("utf-8","replace"))
        except Exception as e:
            sys.stderr.write(f"  retry{i+1} {getattr(e,'code',type(e).__name__)}\n");time.sleep(2*(i+1))
    return None
def norm(t): return re.sub(r"[^a-z0-9]","",re.sub(r"<[^>]+>"," ",t or "").lower())
def inv2abs(inv):
    if not inv: return ""
    pr=[(p,w) for w,ps in inv.items() for p in ps];pr.sort()
    return " ".join(w for _,w in pr)[:1500]
def arx(w):
    for loc in (w.get("locations") or []):
        for s in (loc.get("landing_page_url") or "",loc.get("pdf_url") or ""):
            m=re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",s)
            if m:return m.group(1)
    return ""
SEL=("id,doi,title,display_name,publication_year,cited_by_count,type,authorships,"
     "primary_location,locations,biblio,ids,abstract_inverted_index")
papers=json.load(open(P));idx={norm(p["title"]):p for p in papers};added=0
for i,f in enumerate(GAPS,1):
    u=("https://api.openalex.org/works?filter="+urllib.parse.quote(f,safe=':,|()"')+
       f"&per_page=40&sort=cited_by_count:desc&select={SEL}&mailto={MAILTO}")
    d=fetch(u);n=0
    if d and d.get("results"):
        for w in d["results"]:
            t=w.get("display_name") or "";k=norm(t)
            if not k or len(k)<10 or k in idx: continue
            pl=w.get("primary_location") or {};src=(pl.get("source") or {}) if pl else {}
            b=w.get("biblio") or {}
            pg=(b.get("first_page") or "")+("-"+b["last_page"] if b.get("last_page") else "")
            rec={"title":t,"authors":[a["author"]["display_name"] for a in (w.get("authorships") or []) if a.get("author",{}).get("display_name")],
                 "venue_full":src.get("display_name") or "","year":w.get("publication_year"),
                 "abstract":inv2abs(w.get("abstract_inverted_index")),
                 "doi":(w.get("doi") or "").replace("https://doi.org/",""),"arxiv":arx(w),
                 "volume":b.get("volume") or "","pages":pg,"type":w.get("type") or "",
                 "cited_by_count":w.get("cited_by_count",0),"url":w.get("doi") or w.get("id"),
                 "source_api":"openalex","record_id":w.get("id",""),"source_query":f,"topic":"gapfill"}
            idx[k]=rec;papers.append(rec);n+=1;added+=1
    print(f"[{i}/{len(GAPS)}] +{n} | {f[:70]}",flush=True)
    time.sleep(0.5)
json.dump(papers,open(P,"w"),indent=1)
print(f"\ngapfill added {added}; total {len(papers)}")
