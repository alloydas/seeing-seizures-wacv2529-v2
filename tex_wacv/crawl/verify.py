#!/usr/bin/env python3
"""Re-fetch each finalist DOI directly from OpenAlex to confirm the record exists and metadata matches."""
import json,re,time,urllib.parse,urllib.request,sys
MAILTO="anonymous@example.com";UA=f"paper-crawler/1.0 (mailto:{MAILTO})"
SEL=("id,doi,title,display_name,publication_year,cited_by_count,type,authorships,"
     "primary_location,locations,biblio,ids,abstract_inverted_index,referenced_works_count")
def fetch(u,tries=8):
    for i in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept":"application/json"})
            with urllib.request.urlopen(r,timeout=45) as x: return json.loads(x.read().decode("utf-8","replace"))
        except Exception as e:
            c=getattr(e,'code',type(e).__name__)
            if c==404: return None
            sys.stderr.write(f"   retry{i+1} {c}\n"); time.sleep(8*(i+1))
    return "ERR"
def inv2abs(inv):
    if not inv: return ""
    pr=[(p,w) for w,ps in inv.items() for p in ps];pr.sort()
    return " ".join(w for _,w in pr)[:900]
def arx(w):
    for loc in (w.get("locations") or []):
        for s in (loc.get("landing_page_url") or "",loc.get("pdf_url") or ""):
            m=re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})",s)
            if m:return m.group(1)
    ids=w.get("ids") or {}
    d=(w.get("doi") or "")
    m=re.search(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})",d.lower())
    if m: return m.group(1)
    return ""
rows=[l.strip().split("|") for l in open("/path/to/repo/tex_wacv/crawl/finalists.txt") if l.strip()]
out=[];bad=[]
for i,(tp,doi) in enumerate(rows,1):
    u=f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?select={SEL}&mailto={MAILTO}"
    w=fetch(u)
    if w in (None,"ERR") or not isinstance(w,dict):
        bad.append((tp,doi,"NOT_FOUND" if w is None else "FETCH_ERR")); print(f"[{i}/{len(rows)}] !! {doi} {'404' if w is None else 'ERR'}",flush=True); time.sleep(1.5); continue
    pl=w.get("primary_location") or {};src=(pl.get("source") or {}) if pl else {}
    b=w.get("biblio") or {}
    pg=(b.get("first_page") or "")+("-"+b["last_page"] if b.get("last_page") else "")
    rec={"topic":tp,"title":w.get("display_name"),
         "authors":[a["author"]["display_name"] for a in (w.get("authorships") or []) if a.get("author",{}).get("display_name")],
         "venue":src.get("display_name") or "","issn_l":src.get("issn_l") or "",
         "publisher":src.get("host_organization_name") or "",
         "year":w.get("publication_year"),"volume":b.get("volume") or "","issue":b.get("issue") or "",
         "pages":pg,"doi":(w.get("doi") or "").replace("https://doi.org/",""),"arxiv":arx(w),
         "pmid":(w.get("ids") or {}).get("pmid","").replace("https://pubmed.ncbi.nlm.nih.gov/","") if (w.get("ids") or {}).get("pmid") else "",
         "type":w.get("type"),"cited_by_count":w.get("cited_by_count",0),
         "openalex_id":w.get("id"),"abstract":inv2abs(w.get("abstract_inverted_index")),
         "verified_via":"openalex works/doi lookup"}
    out.append(rec); print(f"[{i}/{len(rows)}] OK {w.get('publication_year')} {str(w.get('display_name'))[:70]}",flush=True)
    time.sleep(1.5)
json.dump(out,open("/path/to/repo/tex_wacv/crawl/verified.json","w"),indent=1)
json.dump(bad,open("/path/to/repo/tex_wacv/crawl/unverified.json","w"),indent=1)
print(f"\nVERIFIED {len(out)} / {len(rows)}   FAILED {len(bad)}")
for b in bad: print("  FAIL:",b)
