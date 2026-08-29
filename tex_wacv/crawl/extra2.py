import json,urllib.parse,urllib.request,time
MAILTO="anonymous@example.com";UA=f"paper-crawler/1.0 (mailto:{MAILTO})"
SEL="id,doi,display_name,publication_year,cited_by_count,type,authorships,primary_location,biblio,ids"
def fetch(u,tries=5):
    for i in range(tries):
        try:
            r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept":"application/json"})
            with urllib.request.urlopen(r,timeout=45) as x: return json.loads(x.read().decode("utf-8","replace"))
        except Exception as e:
            c=getattr(e,'code',type(e).__name__)
            if c==404: return None
            time.sleep(12*(i+1))
    return None
Q=["Modification of seizure activity by electrical stimulation motor seizure",
   "Standards for testing and clinical validation of seizure detection devices",
   "A systematic study of the class imbalance problem in convolutional neural networks",
   "Class-Balanced Loss Based on Effective Number of Samples"]
out=[]
for q in Q:
    u=("https://api.openalex.org/works?filter=title.search:"+urllib.parse.quote(q,safe='')+
       f"&per_page=4&select={SEL}&mailto={MAILTO}")
    d=fetch(u); print("###",q[:74],flush=True)
    if not d or not d.get("results"): print("    <no result>",flush=True); time.sleep(6); continue
    for w in d["results"][:3]:
        pl=w.get("primary_location") or {};src=(pl.get("source") or {}) if pl else {}
        b=w.get("biblio") or {}
        pg=(b.get("first_page") or "")+("-"+b["last_page"] if b.get("last_page") else "")
        au="; ".join([a["author"]["display_name"] for a in (w.get("authorships") or [])][:6])
        print(f"    {w.get('publication_year')} c={w.get('cited_by_count')} type={w.get('type')} | {src.get('display_name')} vol={b.get('volume')} iss={b.get('issue')} pp={pg}",flush=True)
        print(f"      {w.get('display_name')[:92]}",flush=True)
        print(f"      {au[:84]}",flush=True)
        print(f"      doi={(w.get('doi') or '').replace('https://doi.org/','')} | {w.get('id')}",flush=True)
        out.append(w)
    time.sleep(6)
json.dump(out,open("/path/to/repo/tex_wacv/crawl/extra2.json","w"),indent=1)
print("done",flush=True)
