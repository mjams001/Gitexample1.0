This script will be used to do the following: 
    1. Reconfigure VLAN 90 for any branch edge device that is currently misconfigured. This current address will be validated against the correct IP (pulled from the petco_store_ip function) and will be updated for the module and pushed to the edge. 

Imports/Dependencies:
    1. get_edges function, using velocloud script to classify edges into dictionaries for branches and hubs
    2. petco_store_ips function, using python class logic to return ip information for specific store devices

This Change was peer-reviewed by Haley prior to CAB, 10/27. 


ollama run phi3


local_rag_editor/
│
├── test_docs/               # put your .txt and .docx here
├── faiss_store/             # created by index build
├── editor_knowledge.json    # fact store (auto-created)
│
├── requirements.txt
├── common_facts.py
├── build_index.py
├── ask.py
└── editor.py


pip install -r requirements.txt

python build_index.py

python ask.py "Give a brief overview of the setup"
python ask.py "Which links are for security users and what happens if they can't access them?"


# Natural language (multiple changes allowed in one paragraph)
python editor.py parse "Project Alpha deadline is August 12. Replace vendor ABC with vendor XYZ. The DR plan 2021 is legacy."

# Structured upsert / retire
python editor.py upsert onboarding.deadline_days "3 business days"
python editor.py retire portal.vpn.url



faiss-cpu
numpy
python-docx
requests

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
PARSER_MODEL = "phi3"  # change to "llama3.1:8b-instruct" if you prefer

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

# ---------- generic NL parser (domain-agnostic) ----------
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
        if not isinstance(ch, dict):
            continue
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

    # quick generic retirees (legacy/no longer/etc.)
    quick_changes = []
    for phrase in re.findall(r"(?i)\b(?:is no longer|retire|deprecated|legacy|superseded)\b.*?(?:the\s+)?(.+?)\b", t):
        key_guess = _make_key(phrase, "status")
        quick_changes.append({"action": "retire", "key": key_guess})

    # LLM-driven multi-extraction (domain-agnostic examples)
    prompt = f"""
Extract FACT UPDATES from the instruction below.
Return ONLY minified JSON with this schema:

{{
  "changes": [
    {{
      "action": "upsert" | "retire" | "replace",
      "key": "dot_or_snake.key",         // if missing, infer from subject+attribute
      "value": "string (optional for retire)",
      "subject": "who/what (optional)",
      "attribute": "which property (e.g., owner, deadline, url) (optional)"
    }}
  ]
}}

Examples (domain-agnostic):
INPUT: "Project Alpha deadline is August 12. The old status page is legacy. Replace vendor ABC with vendor XYZ for payments."
OUTPUT: {{"changes":[
  {{"action":"upsert","key":"project.alpha.deadline","value":"August 12","subject":"Project Alpha","attribute":"deadline"}},
  {{"action":"retire","key":"status.page.url","subject":"status page","attribute":"url"}},
  {{"action":"upsert","key":"payments.vendor","value":"vendor XYZ","subject":"payments","attribute":"vendor"}}
]}}

INPUT: "Support email is help@example.com. API base url is https://api.example.com/v2."
OUTPUT: {{"changes":[
  {{"action":"upsert","key":"support.email","value":"help@example.com","subject":"support","attribute":"email"}},
  {{"action":"upsert","key":"api.base_url","value":"https://api.example.com/v2","subject":"api","attribute":"base_url"}}
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
            json={"model": PARSER_MODEL, "prompt": prompt, "stream": False},
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


--------------------------------------------------------------
# build_index.py
import json, time
from pathlib import Path
import numpy as np
import requests
import faiss

DOCS_DIR = Path("test_docs")
STORE_DIR = Path("faiss_store")
EMBED_MODEL = "nomic-embed-text"      # ollama pull nomic-embed-text
CHUNK_WORDS = 250
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
        if not p.is_file():
            continue
        if p.suffix.lower() == ".txt":
            txt = read_txt(p)
        elif p.suffix.lower() == ".docx":
            try:
                txt = read_docx(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue
        else:
            continue
        if txt.strip():
            docs.append({"path": str(p), "text": txt})
    return docs

def chunk(text: str, max_words=CHUNK_WORDS) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i+max_words]) for i in range(0, len(w), max_words) if w[i:i+max_words]]

def embed_ollama(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        r = requests.post(f"{OLLAMA}/api/embeddings",
                          json={"model": EMBED_MODEL, "prompt": t, "options": {"truncate": 8192}},
                          timeout=120)
        r.raise_for_status()
        vecs.append(np.array(r.json()["embedding"], dtype=np.float32))
        time.sleep(0.01)
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

def main():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    if not docs:
        print(f"No .txt/.docx found in {DOCS_DIR.resolve()}")
        return

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
---------------------------------------------------------------

# ask.py
import sys, json
from pathlib import Path
import numpy as np
import requests
import faiss
from common_facts import find_relevant_facts

STORE_DIR = Path("faiss_store")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "phi3"   # change to "llama3.1:8b-instruct" if you prefer and your machine can handle it
TOP_K = 5
OLLAMA = "http://localhost:11434"

def embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text, "options": {"truncate": 8192}},
                      timeout=120)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype=np.float32)
    return v

def load_index():
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    meta = json.loads((STORE_DIR / "meta.json").read_text(encoding="utf-8"))
    return index, meta

def build_prompt(question: str, doc_ctxs: list[dict], fact_ctxs: list[dict]) -> str:
    facts_str = "\n".join([f"- {f['key']}: {f['value']}" for f in fact_ctxs]) if fact_ctxs else "(none)"
    docs_str = "\n\n---\n\n".join(
        [f"[Source: {c['path']} | chunk {c['chunk_idx']}]\n{c['text']}" for c in doc_ctxs]
    )
    return (
        "You are an onboarding assistant. Answer strictly from the FACTS and DOCUMENT CONTEXT.\n"
        "If the answer is not present, say you don't have that info.\n\n"
        f"FACTS (editor-curated, newest wins):\n{facts_str}\n\n"
        f"DOCUMENT CONTEXT:\n{docs_str}\n\n"
        f"QUESTION: {question}\n\n"
        "FINAL ANSWER:"
    )

def chat(prompt: str) -> str:
    r = requests.post(f"{OLLAMA}/api/generate",
                      json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
                      timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    index, meta = load_index()

    qv = embed(question)
    qv = qv / (np.linalg.norm(qv) + 1e-8)
    D, I = index.search(qv.reshape(1, -1), TOP_K)

    doc_ctxs = [meta[i] for i in I[0]]
    fact_ctxs = find_relevant_facts(question, top_n=5)

    prompt = build_prompt(question, doc_ctxs, fact_ctxs)
    ans = chat(prompt)

    print("\n" + "="*64)
    print("ANSWER:\n" + ans)
    print("="*64)
    print("Top document sources:")
    for idx, score in zip(I[0], D[0]):
        m = meta[idx]
        print(f"- {m['path']} (chunk {m['chunk_idx']})  score={float(score):.3f}")
    if fact_ctxs:
        print("\nFacts used:")
        for f in fact_ctxs:
            print(f"- {f['key']}: {f['value']}")

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
  python editor.py upsert network.manager "MJ"
  python editor.py retire network.manager
  python editor.py parse "Replace MJ with Alex as Network Team Manager"
"""

def main():
    if len(sys.argv) < 2:
        print(HELP)
        return
    cmd = sys.argv[1].lower()

    if cmd == "upsert" and len(sys.argv) >= 4:
        key = sys.argv[2]
        value = " ".join(sys.argv[3:]).strip().strip('"')
        rec = upsert_fact(key, value, editor="cli")
        print("✅ upserted:", rec)
        return

    if cmd == "retire" and len(sys.argv) >= 3:
        key = sys.argv[2]
        ok = retire_fact(key, editor="cli")
        print("✅ retired" if ok else "Nothing to retire.")
        return

    if cmd == "parse" and len(sys.argv) >= 3:
        instr = " ".join(sys.argv[2:]).strip().strip('"')
        parsed = parse_editor_command_nl(instr)
        print("Parsed:", json.dumps(parsed, ensure_ascii=False))
        applied = []
        for ch in parsed.get("changes", []):
            action = ch.get("action")
            key = ch.get("key")
            value = ch.get("value")
            if not key:
                continue
            if action in ("upsert","replace"):
                rec = upsert_fact(key, value or "", editor="nl")
                applied.append({"action":"upsert","key":key,"value":value})
            elif action == "retire":
                ok = retire_fact(key, editor="nl")
                if ok:
                    applied.append({"action":"retire","key":key})
        print("Applied:", json.dumps(applied, ensure_ascii=False))
        return

    print(HELP)

if __name__ == "__main__":

----------------------------------------------------------------------------


    main()
