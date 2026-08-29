import json,re
papers=json.load(open("/path/to/repo/tex_wacv/crawl/papers.json"))
def T(p): return re.sub(r"<[^>]+>"," ",(p.get("title") or "")).lower()
def AB(p): return (p.get("abstract") or "").lower()
SEIZ=[r"seizure",r"epilep",r"convuls",r"ictal",r"racine",r"spasm",r"status epilepticus"]
TECH=[r"\bvideo",r"camera",r"automat",r"deep learn",r"machine learn",r"neural net",r"\bcnn\b",
      r"computer vision",r"pose",r"tracking",r"detect",r"classif",r"scoring",r"quantif",
      r"grading",r"sever",r"monitor",r"algorithm",r"markerless",r"behavio"]
WET=[r"crispr",r"receptor",r"protein",r"gene ",r"knockout",r"mrna",r"astro",r"neuroinflam",r"rapamycin",
     r"myelin",r"interneuron",r"antibod",r"expression",r"pathway",r"inhibitor",r"agonist",r"extract",
     r"treatment of",r"attenuat",r"ameliorat",r"neuroprotect",r"kinase",r"channel",r"cortical neuron",
     r"blood.brain",r"transporter",r"synap",r"hippocamp",r"dose",r"drug",r"pharmac",r"anticonvuls",
     r"therap",r"stimulation",r"optogenetic",r"metabol",r"microbio",r"probiotic"]
def okay(p):
    t=T(p)
    if not any(re.search(r,t) for r in SEIZ): return False
    if not any(re.search(r,t) for r in TECH): return False
    if sum(1 for r in WET if re.search(r,t))>=1: return False
    return True
def anim(p):
    s=T(p)+" "+AB(p)
    return any(re.search(r,s) for r in [r"\bmice\b",r"\bmouse\b",r"\brats?\b",r"rodent",r"murine",r"zebrafish",r"preclinical",r"animal model",r"\bdogs?\b",r"canine"])
def vid(p):
    s=T(p)+" "+AB(p)
    return any(re.search(r,s) for r in [r"\bvideo",r"camera",r"visual",r"pose",r"motion",r"image",r"behavio",r"kinematic",r"accelerom"])
hits=[p for p in papers if okay(p) and anim(p) and vid(p)]
def sc(p):
    q=min(p.get("cited_by_count",0),5000)**0.4
    y=p.get("year") or 0
    if y>=2015:q+=4
    if y>=2020:q+=3
    t=T(p)
    q+=3*sum(1 for r in [r"\bvideo",r"deep learn",r"machine learn",r"automat",r"racine",r"markerless",r"computer vision",r"neural net"] if re.search(r,t))
    return q
hits.sort(key=lambda p:-sc(p))
print(f"T1 STRICT: {len(hits)}\n"+"="*104)
for p in hits[:45]:
    au=", ".join(p["authors"][:3])+(" et al." if len(p["authors"])>3 else "")
    print(f"{sc(p):5.1f}|{p.get('year')}|c={p.get('cited_by_count',0):<5}| {p['title'][:100]}")
    print(f"        {au[:88]} || {str(p.get('venue_full') or p.get('venue_dblp'))[:52]} || doi:{p.get('doi','')[:44]}")
