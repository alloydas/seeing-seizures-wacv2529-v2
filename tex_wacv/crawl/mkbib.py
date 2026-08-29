import json,re,unicodedata
V=json.load(open("/path/to/repo/tex_wacv/crawl/verified.json"))
def ascii_(s):
    return unicodedata.normalize("NFKD",s or "").encode("ascii","ignore").decode()
def clean(t):
    t=re.sub(r"<[^>]+>","",t or "")
    return t.replace("&","\\&").replace("_","\\_").replace("%","\\%").strip()
def key(r):
    a=ascii_(r["authors"][0].split()[-1]).lower() if r["authors"] else "anon"
    a=re.sub(r"[^a-z]","",a)
    w=[x for x in re.sub(r"[^a-zA-Z ]"," ",re.sub(r"<[^>]+>","",r["title"])).split()
       if len(x)>3 and x.lower() not in ("with","from","using","that","this","were","have","been","their","study")]
    return f"{a}{r['year']}{w[0].lower() if w else 'x'}"
seen={}
lines=[]
JOURNAL_ABBR={
 "IEEE Journal of Biomedical and Health Informatics":"IEEE J. Biomed. Health Inform. (JBHI)",
 "IEEE Transactions on Neural Networks and Learning Systems":"IEEE Trans. Neural Netw. Learn. Syst. (TNNLS)",
 "IEEE Transactions on Pattern Analysis and Machine Intelligence":"IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI)",
 "International Journal of Neural Systems":"Int. J. Neural Syst. (IJNS)",
 "Journal of Artificial Intelligence Research":"J. Artif. Intell. Res. (JAIR)",
 "Nature Neuroscience":"Nat. Neurosci.","Nature Methods":"Nat. Methods",
 "Nature Protocols":"Nat. Protoc.","Nature Communications":"Nat. Commun.",
 "Nature Machine Intelligence":"Nat. Mach. Intell.","Communications Biology":"Commun. Biol.",
 "Scientific Reports":"Sci. Rep.","Clinical Neurophysiology":"Clin. Neurophysiol.",
 "Epilepsy & Behavior":"Epilepsy Behav.","Epilepsy Research":"Epilepsy Res.",
 "Journal Of Big Data":"J. Big Data","Expert Systems with Applications":"Expert Syst. Appl.",
 "Frontiers in Neurology":"Front. Neurol.","Frontiers in Computational Neuroscience":"Front. Comput. Neurosci.",
 "Frontiers in Human Neuroscience":"Front. Hum. Neurosci.","Clinical Autonomic Research":"Clin. Auton. Res.",
 "Neuropsychopharmacology":"Neuropsychopharmacology","IEEE Sensors Journal":"IEEE Sens. J.",
 "JAMA Internal Medicine":"JAMA Intern. Med.","PLoS Medicine":"PLoS Med.","PLoS ONE":"PLoS ONE",
 "Clinical Kidney Journal":"Clin. Kidney J.","Biomedical Instrumentation & Technology":"Biomed. Instrum. Technol.",
 "International Journal of Environmental Research and Public Health":"Int. J. Environ. Res. Public Health",
 "Journal of Neural Engineering":"J. Neural Eng.","Epilepsia":"Epilepsia","Seizure":"Seizure",
 "EBioMedicine":"EBioMedicine","Neuron":"Neuron","Brain":"Brain","eNeuro":"eNeuro","eLife":"eLife",
 "Sensors":"Sensors","Epileptic Disorders":"Epileptic Disord.",
}
for r in sorted(V,key=lambda x:(x["topic"],-(x["cited_by_count"] or 0))):
    k=key(r); n=2
    while k in seen: k=key(r)+chr(ord('a')+n-2); n+=1
    seen[k]=1
    ven=r["venue"] or ""
    typ="inproceedings" if r["type"]=="conference-paper" else ("misc" if r["type"]=="preprint" else "article")
    f=[]
    f.append(f"  author  = {{{ ' and '.join(clean(a) for a in r['authors']) }}}")
    f.append(f"  title   = {{{clean(r['title'])}}}")
    if typ=="inproceedings": f.append(f"  booktitle = {{{clean(ven) or 'Proceedings'}}}")
    elif typ=="misc": f.append(f"  howpublished = {{{clean(ven)}}}")
    else: f.append(f"  journal = {{{clean(ven)}}}")
    if r["volume"]: f.append(f"  volume  = {{{r['volume']}}}")
    if r["issue"]:  f.append(f"  number  = {{{r['issue']}}}")
    if r["pages"]:  f.append(f"  pages   = {{{r['pages']}}}")
    f.append(f"  year    = {{{r['year']}}}")
    if r["doi"]:    f.append(f"  doi     = {{{r['doi']}}}")
    ab=JOURNAL_ABBR.get(ven)
    note=[]
    if ab and ab!=ven: note.append(f"abbrev: {ab}")
    if r["pmid"]: note.append(f"PMID {r['pmid']}")
    lines.append(f"% [{r['topic']}] OpenAlex {r['openalex_id']} | cited_by={r['cited_by_count']}"+(f" | {'; '.join(note)}" if note else ""))
    lines.append(f"@{typ}{{{k},\n"+",\n".join(f)+"\n}\n")
open("/path/to/repo/tex_wacv/crawl/verified_refs.bib","w").write("\n".join(lines))
print(f"wrote {len(V)} entries to verified_refs.bib")
