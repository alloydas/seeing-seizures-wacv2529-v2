import json, re, sys
P="/path/to/repo/tex_wacv/crawl/papers.json"
papers=json.load(open(P))
def txt(p): return ((p.get("title") or "")+" "+(p.get("abstract") or "")).lower()

# (must-have-any, bonus, penalty)
TOPIC={
"1_video_rodent":(
  [r"\brodent",r"\bmice\b",r"\bmouse\b",r"\brat\b",r"\brats\b",r"racine",r"preclinical",r"animal model"],
  [r"video",r"seizure",r"convuls",r"behavio",r"epilep",r"camera",r"automat"],
  [r"\bhuman patients only\b"]),
"2_ecg_cardiac":(
  [r"\becg\b",r"electrocardiogra",r"heart rate",r"cardiac",r"tachycardia",r"\bhrv\b",r"photoplethysmo",r"\bppg\b",r"pulse"],
  [r"seizure",r"epilep",r"ictal",r"detect"],
  []),
"3_animal_behavior_dl":(
  [r"deeplabcut",r"sleap",r"pose estimation",r"markerless",r"behavio",r"ethology",r"tracking"],
  [r"animal",r"rodent",r"mice",r"mouse",r"deep learning",r"neural network",r"phenotyp"],
  []),
"4_imbalance":(
  [r"imbalanc",r"long-tail",r"long tail",r"rare event",r"focal loss",r"class-imbalance",r"resampl",r"minority class",r"anomaly detect"],
  [r"deep",r"neural",r"medical",r"time series",r"video",r"classif",r"detect"],
  []),
"5_cross_subject":(
  [r"cross-subject",r"cross subject",r"subject-independent",r"subject independent",r"inter-subject",
   r"leave-one-subject",r"patient-independent",r"patient independent",r"subject-wise",r"external validation",
   r"generaliz",r"domain shift",r"domain adapt"],
  [r"eeg",r"physiolog",r"biomedical",r"clinical",r"seizure",r"subject",r"patient",r"variabil"],
  []),
"6_false_alarm":(
  [r"false alarm",r"false positive",r"false detection",r"alarm fatigue",r"per hour",r"\bfp/h\b",r"specificity"],
  [r"seizure",r"monitor",r"eeg",r"icu",r"detect",r"wearable",r"clinical",r"continuous"],
  []),
}
def score(p, must, bonus, pen):
    t=txt(p)
    m=sum(1 for r in must if re.search(r,t))
    if m==0: return -1
    b=sum(1 for r in bonus if re.search(r,t))
    q=m*3+b
    if p.get("cited_by_count",0)>0: q+= min(p["cited_by_count"],3000)**0.35
    if p.get("year") and p["year"]>=2015: q+=1.5
    if p.get("abstract"): q+=0.5
    if (p.get("type") or "") in ("article","inproceedings","conference","proceedings-article"): q+=0.5
    if re.search(r"retract|erratum|correction to|editorial|comment on",t): q-=15
    return q
which=sys.argv[1] if len(sys.argv)>1 else None
for name,(must,bonus,pen) in TOPIC.items():
    if which and which not in name: continue
    sc=[(score(p,must,bonus,pen),p) for p in papers]
    sc=[x for x in sc if x[0]>0]
    sc.sort(key=lambda x:-x[0])
    print(f"\n{'='*100}\nTOPIC {name}  ({len(sc)} candidates)\n{'='*100}")
    for s,p in sc[:int(sys.argv[2]) if len(sys.argv)>2 else 28]:
        au=", ".join(p["authors"][:3])+(" et al." if len(p["authors"])>3 else "")
        print(f"[{s:5.1f}] {p.get('year')} | cit={p.get('cited_by_count',0)} | {p['title'][:115]}")
        print(f"        {au[:95]}")
        print(f"        VEN={str(p.get('venue_full') or p.get('venue_dblp'))[:78]} | DOI={p.get('doi','')[:60]} | {p['source_api']}")
