import json, re, sys
P="/path/to/repo/tex_wacv/crawl/papers.json"
papers=json.load(open(P))
def T(p): return re.sub(r"<[^>]+>"," ",(p.get("title") or "")).lower()
def A(p): return (p.get("abstract") or "").lower()
def has(s,pats): return any(re.search(r,s) for r in pats)

ANIMAL=[r"\brodent",r"\bmice\b",r"\bmouse\b",r"\brats?\b",r"\bmurine\b",r"preclinical",r"animal model",r"\bzebrafish",r"in vivo",r"\bfreely.moving"]
VIDEO=[r"\bvideo",r"\bvisual\b",r"camera",r"image",r"\bvision\b",r"optical flow",r"motion",r"pose",r"tracking",r"depth sens"]
SEIZ=[r"seizure",r"epilep",r"convuls",r"ictal",r"racine",r"status epilepticus"]
DL=[r"deep learn",r"neural network",r"machine learn",r"\bcnn\b",r"transformer",r"automat",r"classif",r"detect",r"artificial intelligen",r"supervised"]
CARD=[r"\becg\b",r"electrocardiogra",r"heart rate",r"cardiac",r"tachycard",r"\bhrv\b",r"photoplethysmo",r"\bppg\b",r"heart.rate variab",r"pulse rate",r"\brr interval"]
POSE=[r"pose estimation",r"markerless",r"deeplabcut",r"sleap",r"tracking",r"behavio",r"etholog",r"phenotyp",r"keypoint"]
IMB=[r"imbalanc",r"long.tail",r"rare event",r"focal loss",r"minority class",r"resampl",r"oversampl",r"undersampl",r"skewed",r"class.weight",r"anomaly detect",r"novelty detect"]
XSUB=[r"cross.subject",r"subject.independent",r"inter.subject",r"leave.one.subject",r"patient.independent",r"subject.wise",r"cross.patient",r"external validation",r"inter.individual",r"subject.specific",r"generaliz",r"domain adapt",r"domain shift",r"transfer learn"]
FA=[r"false alarm",r"false positive",r"false detection",r"alarm fatigue",r"per hour",r"specificity",r"precision.recall"]
MON=[r"monitor",r"continuous",r"long.term",r"ambulatory",r"wearable",r"\bicu\b",r"intensive care",r"deploy",r"real.world",r"clinical"]

RULES={
"1_video_rodent_seizure": lambda p: (has(T(p),SEIZ) and has(T(p),ANIMAL+[r""] if False else ANIMAL) and has(T(p)+" "+A(p),VIDEO+DL)) or
                                    (has(T(p),SEIZ) and has(T(p),VIDEO) and has(T(p)+" "+A(p),ANIMAL)) or
                                    (has(T(p),[r"racine"])),
"2_ecg_cardiac_seizure": lambda p: has(T(p),CARD) and has(T(p)+" "+A(p),SEIZ),
"3_animal_behavior_dl":  lambda p: has(T(p),POSE) and has(T(p)+" "+A(p),ANIMAL+[r"\banimal"]) and has(T(p)+" "+A(p),DL),
"4_imbalance_rare":      lambda p: has(T(p),IMB),
"5_cross_subject":       lambda p: has(T(p),XSUB) and has(T(p)+" "+A(p),[r"\beeg\b",r"physiolog",r"biomedical",r"clinical",r"seizure",r"\becg\b",r"patient",r"subject",r"medical",r"brain"]),
"6_false_alarm":         lambda p: has(T(p),FA) and has(T(p)+" "+A(p),MON+SEIZ),
}
def sc(p):
    q=0.0
    c=p.get("cited_by_count",0)
    q += min(c,5000)**0.4
    y=p.get("year") or 0
    if y>=2015: q+=4
    if y>=2020: q+=2
    if p.get("abstract"): q+=1
    if p.get("doi"): q+=1
    if re.search(r"retract|erratum|correction to|editorial|^comment",T(p)): q-=50
    return q
which=sys.argv[1] if len(sys.argv)>1 else ""
N=int(sys.argv[2]) if len(sys.argv)>2 else 25
for name,rule in RULES.items():
    if which and which not in name: continue
    hits=[p for p in papers if rule(p)]
    hits.sort(key=lambda p:-sc(p))
    print(f"\n{'='*104}\nTOPIC {name} -- {len(hits)} matches\n{'='*104}")
    for p in hits[:N]:
        au=", ".join(p["authors"][:3])+(" et al." if len(p["authors"])>3 else "")
        print(f"{p.get('year')} cit={p.get('cited_by_count',0):<6} {p['title'][:112]}")
        print(f"      {au[:92]}")
        print(f"      VEN={str(p.get('venue_full') or p.get('venue_dblp'))[:74]} DOI={p.get('doi','')[:52]} [{p['source_api']}]")
