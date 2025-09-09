import requests
import json
import getpass
import csv
import re
from get_edges import get_edge_info
from petco_store_ips import StoreIPs


user = input('Enter email address here: \n')
password = getpass.getpass()

enterpriseId = 3

login = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/login/enterpriseLogin', json={'username': user, 'password': password})
# print(login.headers)
auth = login.cookies

count = 0

# Call function to get list of edges
result = get_edge_info(enterpriseId,auth)

for edge in result['branches']:
    if count < 1500:
    #if count < 1:
        if 'PETCO' in edge['edgeName']:
            location = re.findall('[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]', edge["edgeName"])
        #if 'LAB82' in edge['edgeName']:
        #    location = re.findall('[0-9][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]', edge["edgeName"])
            store = int(location[0])
            count = count + 1
            store_id = edge['edgeId']
            store_name = edge['edgeName']
            get_ips = StoreIPs(store)
            vlan90ip = get_ips.vlan90_gateway

            response = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/edge/getEdgeConfigurationStack', json={'enterpriseId': 3, 'edgeId': int(store_id)}, cookies=auth)
            config_data = json.loads(response.content)

            try:
                edgeSpecificProfile = config_data[0]
                edgeSpecificProfileDeviceSettings = [m for m in edgeSpecificProfile['modules'] if m['name'] == 'deviceSettings'][0]
                edgeSpecificProfileDeviceSettingsData = edgeSpecificProfileDeviceSettings['data']
                moduleId = edgeSpecificProfileDeviceSettings['id']
            
                for i in edgeSpecificProfileDeviceSettingsData['lan']['networks']:
                    if i['vlanId'] == 90 and i['cidrIp'] != vlan90ip:
                        i['cidrIp'] = vlan90ip
                        
                        response_data = { 'enterpriseId': 3,
                        'id': moduleId,
                        '_update': { 'data':  edgeSpecificProfileDeviceSettingsData }}
                        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                        update = requests.post('https://vco56-usvi1.velocloud.net/portal/rest/configuration/updateConfigurationModule', data=json.dumps(response_data), cookies=auth, headers=headers)
                        if update.status_code == 200:
                            print(f'{store_name} has been updated with correct VLAN 90 configuration')
                        else:
                            print(f'{update.reason,update.status_code},{update.content},{store_name} needs VLAN 90 validation!')                            

            except:
                pass

------------------------------------------------------
# common_facts.py
import json, re, unicodedata
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import requests

FACTS_FILE = Path("editor_knowledge.json")
AUDIT_FILE = Path("editor_audit.json")

# Ollama config
OLLAMA = "http://localhost:11434"
PARSER_MODEL = "llama3.1:8b"  # can be "phi3" or "llama3.2:3b-instruct" if you prefer

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_facts():
    if not FACTS_FILE.exists():
        return []
    return json.loads(FACTS_FILE.read_text(encoding="utf-8"))

def save_facts(facts):
    FACTS_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")

def append_audit(entry):
    log = []
    if AUDIT_FILE.exists():
        try:
            log = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append(entry)
    AUDIT_FILE.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_fact(key: str, value: str, tags=None, editor="unknown"):
    facts = load_facts()
    for f in facts:
        if f["key"] == key and f.get("status","active") == "active":
            f["status"] = "retired"
            f["updated_at"] = _now_iso()
            f["replaced_by"] = value
    newf = {
        "key": key,
        "value": value,
        "status": "active",
        "tags": tags or [],
        "updated_at": _now_iso(),
        "editor": editor
    }
    facts.append(newf)
    save_facts(facts)
    append_audit({"action":"upsert", "key":key, "value":value, "editor":editor, "ts":_now_iso()})
    return newf

def retire_fact(key: str, editor="unknown"):
    facts = load_facts()
    changed = False
    for f in facts:
        if f["key"] == key and f.get("status","active") == "active":
            f["status"] = "retired"
            f["updated_at"] = _now_iso()
            changed = True
    if changed:
        save_facts(facts)
        append_audit({"action":"retire", "key":key, "editor":editor, "ts":_now_iso()})
    return changed

def get_active_facts():
    return [f for f in load_facts() if f.get("status","active") == "active"]

def find_relevant_facts(question: str, top_n=5):
    q = question.lower()
    facts = get_active_facts()
    scored = []
    for f in facts:
        hay = (f["key"] + " " + f.get("value","")).lower()
        score = sum(1 for token in set(q.split()) if token in hay)
        if score > 0:
            scored.append((score, f))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:top_n]]

# ---------- generic NL parser ----------
_WORD_RE = re.compile(r"[a-z0-9]+", re.I)

def _slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    tokens = _WORD_RE.findall(t)
    return "_".join(tokens)

def _make_key(subject: str, attribute: str) -> str:
    s = _slugify(subject)
    a = _slugify(attribute)
    if not s and not a:
        return "general.value"
    if not s:
        return a or "misc.value"
    if not a:
        return f"{s.split('_')[0]}.value"
    return f"{s.split('_')[0]}.{a}"

def _extract_json_block(text: str) -> Optional[str]:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def _normalize_changes(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm = []
    for ch in changes:
        if not isinstance(ch, dict): continue
        action = (ch.get("action") or "").strip().lower()
        key = (ch.get("key") or "").strip()
        value = ch.get("value")
        subject = (ch.get("subject") or "").strip()
        attribute = (ch.get("attribute") or "").strip()
        if not key:
            key = _make_key(subject or "general", attribute or "value")
        if action not in {"upsert","retire","replace"}:
            action = "upsert" if value not in (None, "") else "retire"
        norm.append({"action": action, "key": key, "value": value})
    # dedupe
    uniq, seen = [], set()
    for ch in norm:
        sig = (ch["action"], ch["key"], str(ch.get("value")))
        if sig not in seen:
            uniq.append(ch); seen.add(sig)
    return uniq

def parse_editor_command_nl(text: str) -> Dict[str, Any]:
    """
    Parse ANY instruction or paragraph into:
    {"changes":[{"action":"upsert|retire|replace","key":"x.y","value":"..."}]}
    """
    t = text.strip()

    # quick generic retirees
    quick_changes = []
    for phrase in re.findall(r"(?i)\b(?:is no longer|retire|deprecated|legacy|superseded)\b.*?(?:the\s+)?(.+?)\b", t):
        key_guess = _make_key(phrase, "status")
        quick_changes.append({"action": "retire", "key": key_guess})

    # LLM-driven multi-extraction (domain-agnostic)
    prompt = f"""
Extract FACT UPDATES from the instruction below.
Return ONLY minified JSON with this schema:

{{
  "changes": [
    {{
      "action": "upsert" | "retire" | "replace",
      "key": "dot_or_snake.key",
      "value": "string (optional for retire)",
      "subject": "who/what (optional)",
      "attribute": "which property (e.g., owner, deadline, url) (optional)"
    }}
  ]
}}

Examples:
INPUT: "Project Alpha deadline is August 12. The old status page is legacy. Replace vendor ABC with vendor XYZ for payments."
OUTPUT: {{"changes":[
  {{"action":"upsert","key":"project.alpha.deadline","value":"August 12","subject":"Project Alpha","attribute":"deadline"}},
  {{"action":"retire","key":"status.page.url","subject":"status page","attribute":"url"}},
  {{"action":"upsert","key":"payments.vendor","value":"vendor XYZ","subject":"payments","attribute":"vendor"}}
]}}

Now parse this instruction:

INSTRUCTION:
{t}

JSON:
""".strip()

    llm_changes = []
    try:
        r = requests.post(
            f"{OLLAMA}/api/generate",
            json={"model": PARSER_MODEL, "prompt": prompt, "stream": False,
                  "options":{"num_predict":256,"temperature":0.1}},
            timeout=90
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        block = _extract_json_block(raw) or raw
        data = json.loads(block)
        if isinstance(data, dict) and isinstance(data.get("changes"), list):
            llm_changes = data["changes"]
    except Exception:
        pass

    merged = quick_changes + llm_changes
    if not merged:
        # fallback: key=value lines
        for k, v in re.findall(r"(?i)\b([a-z0-9_.]+)\s*=\s*([^\n;]+)", t):
            merged.append({"action":"upsert","key":k.strip(),"value":v.strip()})

    return {"changes": _normalize_changes(merged)}

-----------------------------------------------------
# build_index.py
import json, time
from pathlib import Path
import numpy as np
import requests
import faiss

DOCS_DIR = Path("test_docs")
STORE_DIR = Path("faiss_store")
EMBED_MODEL = "bge-m3"                 # exact name
CHUNK_WORDS = 180
CHUNK_OVERLAP = 60
OLLAMA = "http://localhost:11434"

def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_docx(p: Path) -> str:
    from docx import Document
    doc = Document(str(p))
    return "\n".join(para.text for para in doc.paragraphs)

def load_docs() -> list[dict]:
    docs = []
    for p in DOCS_DIR.glob("**/*"):
        if not p.is_file(): continue
        if p.suffix.lower() == ".txt":
            txt = read_txt(p)
        elif p.suffix.lower() == ".docx":
            try: txt = read_docx(p)
            except Exception as e:
                print(f"Skipping {p}: {e}"); continue
        else:
            continue
        if txt.strip():
            docs.append({"path": str(p), "text": txt})
    return docs

def chunk(text: str, max_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + max_words, len(words))
        c = " ".join(words[i:j]).strip()
        if c: chunks.append(c)
        if j == len(words): break
        i = max(0, j - overlap)
    return chunks

def embed_ollama(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t, "options": {"truncate": 8192}},
            timeout=120
        )
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama /api/embeddings 404. Run `ollama pull {EMBED_MODEL}` "
                "or restart/update Ollama."
            )
        r.raise_for_status()
        vecs.append(np.array(r.json()["embedding"], dtype=np.float32))
        time.sleep(0.01)
    return np.vstack(vecs) if vecs else np.zeros((0, 1024), dtype=np.float32)

def main():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    if not docs:
        print(f"No .txt/.docx found in {DOCS_DIR.resolve()}"); return

    chunks, meta = [], []
    for d in docs:
        cs = chunk(d["text"])
        for i, c in enumerate(cs):
            chunks.append(c)
            meta.append({"path": d["path"], "chunk_idx": i, "text": c})
    print(f"Total chunks: {len(chunks)}")

    print("Embedding chunks with Ollama…")
    X = embed_ollama(chunks)  # (N, D)

    # normalize for cosine
    norms = (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Xn = X / norms

    d = Xn.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(Xn)

    faiss.write_index(index, str(STORE_DIR / "index.faiss"))
    np.save(STORE_DIR / "embeddings.npy", X)  # optional
    with open(STORE_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Index built at {STORE_DIR.resolve()}")

if __name__ == "__main__":
    main()
----------------------------------------------
# ask.py
import sys, json, math
from pathlib import Path
import numpy as np
import requests
import faiss
from common_facts import find_relevant_facts

STORE_DIR = Path("faiss_store")
EMBED_MODEL = "bge-m3"             # exact name
CHAT_MODEL  = "llama3.1:8b"        # exact name (use "llama3.2:3b-instruct" for a lighter model)
TOP_K = 6
OLLAMA = "http://localhost:11434"

def embed(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text, "options": {"truncate": 8192}},
        timeout=120
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)

def load_index():
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    meta = json.loads((STORE_DIR / "meta.json").read_text(encoding="utf-8"))
    return index, meta

def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + " ..."

def cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(a @ b)

def mmr_select(candidates, q_vec, cand_vecs, top_n=4, lambda_=0.7):
    selected = []
    rem = set(candidates)
    q = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    sims_q = (cand_vecs @ q).tolist()
    while rem and len(selected) < top_n:
        if not selected:
            i = max(rem, key=lambda k: sims_q[k])
            selected.append(i); rem.remove(i)
        else:
            def score(k):
                diversity = max(cosine_vec(cand_vecs[k], cand_vecs[j]) for j in selected) if selected else 0.0
                return lambda_ * sims_q[k] - (1 - lambda_) * diversity
            i = max(rem, key=score)
            selected.append(i); rem.remove(i)
    return selected

def multi_query_expansion(question: str, n=3) -> list[str]:
    prompt = (
        "Rewrite the user question into distinct search queries that cover different angles. "
        f"Return exactly {n} lines, no numbering, no extra text.\n\nQUESTION: {question}\n"
    )
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False,
              "options":{"num_predict":128, "temperature":0.3}},
        timeout=120
    )
    if r.status_code == 404:
        raise RuntimeError("Ollama /api/generate is not available. Restart/update Ollama and re-pull the chat model.")
    r.raise_for_status()
    lines = [ln.strip() for ln in r.json().get("response","").split("\n")]
    return [ln for ln in lines if ln][:n]

def build_prompt(question: str, doc_ctxs: list[dict], fact_ctxs: list[dict]) -> str:
    trimmed, total = [], 0
    for c in doc_ctxs:
        t = _truncate(c["text"], 1100)
        total += len(t)
        if total > 5000: break
        trimmed.append({**c, "text": t})

    sources = [f"[{i}] {c['path']} (chunk {c['chunk_idx']})" for i, c in enumerate(trimmed, 1)]
    source_map = "\n".join(sources) if sources else "(none)"

    facts_str = "\n".join([f"- {f['key']}: {f['value']}" for f in fact_ctxs]) if fact_ctxs else "(none)"
    docs_str = "\n\n---\n\n".join([f"[{i}] {c['text']}" for i, c in enumerate(trimmed, 1)])

    return (
        "You are an onboarding assistant. Use the FACTS and DOCUMENT CONTEXT only. "
        "Prefer editor-curated FACTS over conflicting document text. "
        "If the answer is unknown, say you don't have that info.\n\n"
        f"FACTS (editor-curated, newest wins):\n{facts_str}\n\n"
        f"DOCUMENT CONTEXT (numbered for citation):\n{docs_str}\n\n"
        f"QUESTION: {question}\n\n"
        "RESPONSE STYLE:\n"
        "- Start with a concise answer.\n"
        "- Then provide key steps/bullets if helpful.\n"
        "- Add citations using [1], [2] that refer to sources below.\n"
        "- If something is legacy, note the replacement if present.\n\n"
        f"SOURCES:\n{source_map}\n\n"
        "FINAL ANSWER (with citations where relevant):"
    )

def chat(prompt: str) -> str:
    with requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": True,
              "options": {"num_ctx": 4096, "num_predict": 512, "temperature": 0.2}},
        timeout=600, stream=True
    ) as r:
        if r.status_code == 404:
            raise RuntimeError("Ollama /api/generate is not available. Restart/update Ollama and re-pull the chat model.")
        r.raise_for_status()
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                piece = json.loads(line)
                tok = piece.get("response", "")
                if tok:
                    print(tok, end="", flush=True)
                    out.append(tok)
            except Exception:
                pass
        print()
        return "".join(out).strip()

def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"'); sys.exit(1)
    question = " ".join(sys.argv[1:])

    index, meta = load_index()

    # Original query
    qv0 = embed(question); qv0 = qv0 / (np.linalg.norm(qv0) + 1e-8)
    D0, I0 = index.search(qv0.reshape(1, -1), TOP_K)
    cand_idxs = set(I0[0].tolist())

    # Multi-query expansion → union of candidates
    for q in multi_query_expansion(question, n=3):
        v = embed(q); v = v / (np.linalg.norm(v) + 1e-8)
        D, I = index.search(v.reshape(1, -1), max(3, TOP_K-1))
        cand_idxs.update(I[0].tolist())

    cands = sorted(cand_idxs)
    cand_texts = [meta[i]["text"] for i in cands]
    cand_vecs = np.vstack([embed(t) for t in cand_texts]).astype(np.float32)

    # MMR to pick diverse, relevant set
    def mmr_idx(cands_local, qvec, vecs): return mmr_select(list(range(len(cands_local))), qvec, vecs, top_n=4, lambda_=0.7)
    selected_local_idx = mmr_idx(cands, qv0, cand_vecs)
    chosen = [cands[j] for j in selected_local_idx]
    doc_ctxs = [meta[i] for i in chosen]

    fact_ctxs = find_relevant_facts(question, top_n=5)
    prompt = build_prompt(question, doc_ctxs, fact_ctxs)
    ans = chat(prompt)

    print("\n" + "="*64)
    print("ANSWER ABOVE")
    print("="*64)
    print("Top sources:")
    for i, idx in enumerate(chosen, 1):
        m = meta[idx]
        print(f"[{i}] {m['path']} (chunk {m['chunk_idx']})")

if __name__ == "__main__":
    main()

--------------------------------------------------
# editor.py
import sys, json
from common_facts import upsert_fact, retire_fact, parse_editor_command_nl

HELP = """\
Editor mode (CLI)
  python editor.py upsert <key> "<value>"
  python editor.py retire <key>
  python editor.py parse "<natural language instruction>"

Examples:
  python editor.py upsert onboarding.deadline_days "3 business days"
  python editor.py retire portal.vpn.url
  python editor.py parse "Project Alpha deadline is Aug 12. Old VPN link is legacy. New VPN link is https://vpn.example.gov"
"""

def main():
    if len(sys.argv) < 2:
        print(HELP); return
    cmd = sys.argv[1].lower()

    if cmd == "upsert" and len(sys.argv) >= 4:
        key = sys.argv[2]
        value = " ".join(sys.argv[3:]).strip().strip('"')
        rec = upsert_fact(key, value, editor="cli")
        print("✅ upserted:", rec); return

    if cmd == "retire" and len(sys.argv) >= 3:
        key = sys.argv[2]
        ok = retire_fact(key, editor="cli")
        print("✅ retired" if ok else "Nothing to retire."); return

    if cmd == "parse" and len(sys.argv) >= 3:
        instr = " ".join(sys.argv[2:]).strip().strip('"')
        parsed = parse_editor_command_nl(instr)
        print("Parsed:", json.dumps(parsed, ensure_ascii=False))
        applied = []
        for ch in parsed.get("changes", []):
            action, key, value = ch.get("action"), ch.get("key"), ch.get("value")
            if not key: continue
            if action in ("upsert","replace"):
                rec = upsert_fact(key, value or "", editor="nl")
                applied.append({"action":"upsert","key":key,"value":value})
            elif action == "retire":
                ok = retire_fact(key, editor="nl")
                if ok: applied.append({"action":"retire","key":key})
        print("Applied:", json.dumps(applied, ensure_ascii=False)); return

    print(HELP)

if __name__ == "__main__":
    main()





