import json,re,sys
P="/path/to/repo/tex_wacv/crawl/papers.json"
papers=json.load(open(P))
def T(p): return re.sub(r"<[^>]+>"," ",(p.get("title") or "")).lower()
def AB(p): return (p.get("abstract") or "").lower()
def h(s,pats): return sum(1 for r in pats if re.search(r,s))
JUNK=r"retract|erratum|correction to|^comment on|editorial|author correction|supplementary|abstracts? from|conference abstract"
def base(p):
    q=min(p.get("cited_by_count",0),8000)**0.4
    y=p.get("year") or 0
    if y>=2015:q+=3
    if y>=2019:q+=2
    if p.get("abstract"):q+=1.5
    if p.get("doi"):q+=1
    if re.search(JUNK,T(p)):q-=60
    if (p.get("type") or "")=="paratext":q-=60
    return q
RULES={
"T1":(lambda p:(h(T(p),[r"seizure",r"epilep",r"convuls",r"ictal",r"racine",r"spasm"])>0 and
        h(T(p)+AB(p),[r"\bvideo",r"camera",r"visual",r"behavio",r"pose",r"motion",r"image"])>0 and
        h(T(p)+AB(p),[r"\bmice\b",r"\bmouse\b",r"\brats?\b",r"rodent",r"murine",r"zebrafish",r"preclinical",r"animal"])>0),
     [r"\bvideo",r"deep learn",r"machine learn",r"neural net",r"automat",r"racine",r"sever",r"grad",r"scor"]),
"T2":(lambda p:h(T(p)+AB(p),[r"\becg\b",r"electrocardiogra",r"heart rate",r"cardiac",r"tachycard",r"\bhrv\b",r"photoplethysmo",r"\bppg\b"])>0 and
        h(T(p),[r"seizure",r"epilep",r"ictal"])>0,
     [r"detect",r"\becg\b",r"heart rate",r"tachycard",r"automat",r"algorithm",r"wearable",r"variab"]),
"T3":(lambda p:h(T(p)+AB(p),[r"pose estimation",r"markerless",r"deeplabcut",r"\bsleap\b",r"b-soid",r"deepethogram",r"moseq",r"etholog",r"behavio.*(track|classif|segment|phenotyp)",r"keypoint"])>0 and
        h(T(p)+AB(p),[r"animal",r"mice",r"mouse",r"rodent",r"\brats?\b",r"fly",r"primate",r"behavio"])>0,
     [r"deep learn",r"neural net",r"markerless",r"pose estimation",r"deeplabcut",r"sleap",r"unsupervised",r"toolbox",r"open.source"]),
"T4":(lambda p:h(T(p),[r"imbalanc",r"long.tail",r"rare event",r"focal loss",r"minority",r"oversampl",r"undersampl",r"resampl",r"cost.sensitive",r"skew",r"smote",r"class.weight"])>0,
     [r"deep",r"neural",r"medical",r"clinical",r"time series",r"video",r"survey",r"systematic",r"detect"]),
"T5":(lambda p:h(T(p),[r"cross.subject",r"subject.independent",r"inter.subject",r"leave.one.subject",r"patient.independent",r"subject.wise",r"record.wise",r"cross.patient",r"external validation",r"data leakage",r"generaliz",r"cross.site",r"cross.dataset"])>0 and
        h(T(p)+AB(p),[r"\beeg\b",r"physiolog",r"biomedical",r"clinical",r"seizure",r"medical",r"patient",r"brain",r"\becg\b",r"neuroimag",r"biosignal"])>0,
     [r"\beeg\b",r"seizure",r"cross.subject",r"leave.one.subject",r"validation",r"leakage",r"variabil",r"pitfall",r"bias"]),
"T6":(lambda p:h(T(p),[r"false alarm",r"false positive",r"false detection",r"alarm fatigue",r"per hour",r"specificity"])>0 or
       (h(T(p),[r"seizure detection",r"seizure.detect"])>0 and h(AB(p),[r"false alarm",r"per hour",r"false.positive rate",r"false detection"])>0),
     [r"seizure",r"false alarm",r"per hour",r"monitor",r"wearable",r"clinical",r"prospective",r"validation",r"icu",r"neonat"]),
}
which=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 30
rule,bonus=RULES[which]
hits=[p for p in papers if rule(p)]
def sc(p): return base(p)+2.0*h(T(p)+AB(p),bonus)
hits.sort(key=lambda p:-sc(p))
print(f"TOPIC {which}: {len(hits)} candidates\n"+"="*106)
for p in hits[:N]:
    au=", ".join(p["authors"][:3])+(" et al." if len(p["authors"])>3 else "")
    print(f"{sc(p):5.1f} | {p.get('year')} | c={p.get('cited_by_count',0):<5} | {p['title'][:104]}")
    print(f"        {au[:90]}")
    print(f"        {str(p.get('venue_full') or p.get('venue_dblp'))[:66]} | doi:{p.get('doi','')[:46]} | {p['source_api']}")
