r"""Capstone Checkpoint 5.1 — Agent-Based RAG over the FULL research-paper corpus (IMPROVED).

Improvements over the previous version (for the "improve your system" step):
    1. Cross-encoder RERANKER  — every search retrieves RERANK_CANDIDATES chunks with the
       hybrid retriever, then a cross-encoder rescores them against the query and keeps
       TOP_K. The planner and the answer step also rerank ALL accumulated passages against
       the ORIGINAL question, which fixes the old problem that per-query min-max scores
       were not comparable across queries.
    2. New tool LIST_PAPERS    — returns the paper catalog (file, year, title, authors,
       abstract snippet), optionally filtered by year or ranked by a topic keyword. Lets
       the agent answer "whole collection" questions that chunk search handles badly.
    3. New tool READ_SECTION   — reads a named section (abstract, limitations,
       conclusion, ...) of up to 3 specific papers, instead of hoping that scattered
       chunks happen to contain it. Falls back to the closest passages in that paper when
       the heading can't be found.
    4. LLM-JUDGE (1-5)         — scores every agent and baseline answer on relevance,
       faithfulness, completeness and overall quality, prints a scoreboard, and appends
       every result to eval_results.jsonl with a RUN_LABEL so you can compare runs
       (e.g. "before" with USE_RERANKER=False / ENABLE_PAPER_TOOLS=False vs "after").
    (+ planner decisions are now written to the log file, so the write-up can show them.)

Scenario: Research Paper Navigator (~150 research-paper PDFs).
Tools the agent can choose from:
    semantic_search  -> hybrid (embedding + lexical) search over paper chunks, reranked
    find_by_author   -> retrieve chunks filtered by author and/or year metadata
    list_papers      -> catalog of papers (optionally by year / ranked by keyword)
    read_section     -> read a named section of specific papers
    follow_citation  -> from what it already found, pull in cited / closely related work
    clarify          -> ask the user ONE question when the request is too vague to search
    answer           -> stop and answer once the collected passages cover the question

Jupytext-style cell markers (# %% / # %% [markdown]) preserved — runnable as a plain
script AND openable as cells in VS Code / PyCharm / Jupyter.

--------------------------------------------------------------------------------
Setup
    python 3.11 or 3.12
    pip install langchain-openai langchain-core python-dotenv langgraph \
                pypdf sentence-transformers rank-bm25 numpy
    # .env next to this file:
    #   OPENROUTER_API_KEY_RELANT=sk-or-v1-...
FIRST RUN WITH THE RERANKER: set ALLOW_HF_DOWNLOAD = True (just below the imports) so
sentence-transformers can download the cross-encoder once (~90 MB); then set it back to
False. If the reranker can't load, the script prints a note and keeps working without it.
The corpus cache is rebuilt once (new format: it now also stores per-paper text).
--------------------------------------------------------------------------------
"""

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import difflib
import hashlib
import importlib
import json
import os

# Set True ONCE so sentence-transformers may download models it doesn't have yet
# (the cross-encoder reranker), then set it back to False for fast offline runs.
ALLOW_HF_DOWNLOAD = True
if not ALLOW_HF_DOWNLOAD:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# %% [markdown]
# ## Configuration

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2

# The FULL corpus — not the toy sample. Change if you move the folder.
PDF_DIR = r"C:\Users\Francisco\Documents\FIG\MIT RAG and Context Engineering\PDF Validación 3.1"

SCENARIO = "research_papers"          # "research_papers" or "wikipedia"

MAX_STEPS = 3                         # max tool rounds before the agent is forced to answer
TOP_K = 4                             # chunks returned per search (after reranking)
CHUNK_CHARS = 1100                    # ~ a paragraph or two
CHUNK_OVERLAP = 150
HYBRID_ALPHA = 0.5                    # 0 = pure lexical, 1 = pure semantic
EMBED_MODEL = "all-MiniLM-L6-v2"      # local sentence-transformers model (no extra API)
DECIDE_CONTEXT_CHUNKS = 15            # passages shown to the PLANNER (kept cheap)
ANSWER_CONTEXT_CHUNKS = 30            # passages shown to the ANSWER prompt (fuller)
DEBUG = True

# --- NEW: reranker ---------------------------------------------------------------
USE_RERANKER = True                   # False = old behaviour (for a "before" run)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20                # hybrid retrieves this many, reranker keeps TOP_K

# --- NEW: paper-level tools --------------------------------------------------------
ENABLE_PAPER_TOOLS = True             # list_papers + read_section (False for a "before" run)
LIST_PAPERS_MAX = 25                  # papers returned per list_papers call
SECTION_MAX_CHARS = 4000              # max characters kept from one section
MAX_SECTION_DOCS = 3                  # papers read per read_section call
CATALOG_PLAN_CHARS = 6000             # catalog characters shown to the planner

# --- NEW: LLM-judge evaluation ------------------------------------------------------
RUN_JUDGE = True
JUDGE_MODEL = LLM_MODEL               # ideally a different / stronger model than the agent
JUDGE_CONTEXT_CHARS = 24000           # passages shown to the judge for faithfulness checks
RUN_LABEL = "v3: reranker + list_papers + read_section"  # change per run you compare

LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"
CACHE_PATH = Path.cwd() / "corpus_cache.pkl"
EVAL_RESULTS_PATH = Path.cwd() / "eval_results.jsonl"
CACHE_VERSION = "v2"                  # bump to force a cache rebuild

# %%
_TOOL_DOCS = {
    "semantic_search": (
        "search the paper collection by meaning. Give 1-2 'queries'. Use for topics, "
        "concepts, methods, results."),
    "find_by_author": (
        "retrieve papers filtered by 'author' and/or 'year' (a 4-digit year). Use when the "
        "question names an author."),
    "list_papers": (
        "get the paper catalog: file name, year, title, authors and a short abstract per "
        "paper. Optional 'year' filters by year; optional 'keyword' ranks papers by topic. "
        "Use for questions about the collection as a whole ('which papers...', 'summarize "
        "the papers from 2023', 'across the collection'), or to find the file names of the "
        "papers you want to read."),
    "read_section": (
        "read one named section of up to 3 specific papers. Give 'doc_ids' (file names "
        "exactly as shown in the passages or the catalog) and 'section' (one of: abstract, "
        "introduction, related_work, method, experiments, discussion, limitations, "
        "conclusion). Use when you know which papers matter and need a whole section "
        "(e.g. their limitations or conclusions) rather than scattered chunks."),
    "follow_citation": (
        "from the papers already found, pull in cited or closely related work. Use AFTER "
        "an initial search when the question asks about follow-up, prior, related, or "
        "subsequent work."),
    "clarify": (
        "ask the user ONE short question in 'clarification'. Use ONLY when the request is "
        "too ambiguous to search at all (missing which topic/paper/metric). Never clarify "
        "more than once."),
    "answer": "stop and answer, once the collected passages cover every part of the question.",
}


def build_plan_system(enable_paper_tools: bool) -> tuple[str, list[str]]:
    actions = ["semantic_search", "find_by_author"]
    if enable_paper_tools:
        actions += ["list_papers", "read_section"]
    actions += ["follow_citation", "clarify", "answer"]
    tool_lines = "\n".join(f"- {a}: {_TOOL_DOCS[a]}" for a in actions)
    prompt = (
        "You are the planner for an agentic research-paper assistant. You take exactly ONE "
        "action per step. The available actions are:\n"
        f"{tool_lines}\n\n"
        "You are given the question, the actions already run, the paper catalog (if it has "
        "been listed) and the passages found so far. Choose the next action. Respond with "
        "ONLY a JSON object:\n"
        '{"action": "' + "|".join(actions) + '", "queries": ["..."], "author": "", '
        '"year": "", "keyword": "", "doc_ids": [], "section": "", "clarification": "", '
        '"reasoning": "one sentence"}\n'
        "Do NOT repeat an action already run with the same arguments. Prefer to answer as "
        "soon as the passages are enough — extra rounds cost real money."
    )
    return prompt, actions


PLAN_SYSTEM, AVAILABLE_ACTIONS = build_plan_system(ENABLE_PAPER_TOOLS)

ANSWER_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided context, "
    "citing the source file names in [brackets] where you rely on them. The context may "
    "contain a PAPER CATALOG (file, year, title, abstract snippet) and full SECTIONS of "
    "papers as well as shorter passages. If the context does not contain the answer, say "
    "so plainly instead of guessing."
)

JUDGE_KEYS = ["relevance", "faithfulness", "completeness", "overall"]
JUDGE_SYSTEM = (
    "You are a strict evaluator of a research assistant. You get the QUESTION, the "
    "PASSAGES the assistant had available, and its RESPONSE. Score each criterion from 1 "
    "to 5:\n"
    "- relevance: does the response address the question that was asked?\n"
    "- faithfulness: is every claim supported by the PASSAGES (not outside knowledge)? "
    "Unsupported claims or wrong [citations] lower this score.\n"
    "- completeness: does it cover every part of the question, as far as the passages "
    "allow? Honestly stating that something is not in the passages is acceptable.\n"
    "- overall: your holistic judgment of the response's quality.\n"
    "Scale: 5 = excellent, 4 = good with minor issues, 3 = acceptable but with notable "
    "gaps, 2 = poor, 1 = fails.\n"
    "If the RESPONSE is a clarifying question instead of an answer, rate whether asking "
    "was the right move (5 = the question really was too vague to answer and the "
    "clarifying question is precise; 1 = the question was answerable and asking was "
    "unnecessary) and use that score for all four criteria.\n"
    'Respond with ONLY a JSON object: {"relevance": n, "faithfulness": n, '
    '"completeness": n, "overall": n, "rationale": "one or two sentences"}'
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY_RELANT") or os.getenv("OPENROUTER_API_KEY_RELANT")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY_RELANT not set. Grab a key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm(temperature: float = TEMPERATURE, model: str = LLM_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max a score vector into [0, 1] so semantic and lexical scores can be fused."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return raw.strip()


# %% [markdown]
# ## Section extraction — used by the read_section tool and for catalog abstracts
#
# PDF text keeps its line breaks, so section headings usually sit on their own line,
# e.g. "5 Conclusion", "LIMITATIONS", "II. RELATED WORK". We find heading lines, take the
# requested one, and read until the next heading.

# %%
SECTION_ALIASES: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related_work": ["related work", "related works", "background", "prior work",
                     "literature review", "background and related work"],
    "method": ["method", "methods", "methodology", "approach", "proposed method",
               "proposed approach", "our approach", "materials and methods"],
    "experiments": ["experiments", "experiment", "experimental setup", "experimental results",
                    "evaluation", "results", "results and discussion", "empirical evaluation"],
    "discussion": ["discussion"],
    "limitations": ["limitations", "limitation", "limitations and future work",
                    "threats to validity", "limitations and future directions"],
    "conclusion": ["conclusion", "conclusions", "conclusion and future work",
                   "conclusions and future work", "concluding remarks", "future work"],
}
_TERMINATORS = ["references", "bibliography", "acknowledgments", "acknowledgements",
                "acknowledgment", "acknowledgement", "appendix", "appendices",
                "supplementary material"]
_NAME_TO_CANON = {a: canon for canon, aliases in SECTION_ALIASES.items() for a in aliases}
for _t in _TERMINATORS:
    _NAME_TO_CANON.setdefault(_t, "_end")

_HEAD_RE = re.compile(
    r"^\s*((?:\d{1,2}|[IVX]{1,4})(?:\.\d{1,2})*\.?\s+)?([A-Za-z][A-Za-z ,&:\-]{2,70}?)\s*[:.]?\s*$"
)


def canonical_section(name: str) -> str:
    n = re.sub(r"[_\-]+", " ", (name or "").lower()).strip()
    if not n:
        return "abstract"
    if n.replace(" ", "_") in SECTION_ALIASES:
        return n.replace(" ", "_")
    if _NAME_TO_CANON.get(n, "_end") != "_end":
        return _NAME_TO_CANON[n]
    for canon, aliases in SECTION_ALIASES.items():
        if any(n.startswith(a) or a.startswith(n) for a in aliases):
            return canon
    return n


def extract_section(text: str, section: str, max_chars: int = SECTION_MAX_CHARS) -> str:
    """Return the body of the requested section, or '' if no heading is found."""
    canon = canonical_section(section)
    targets = set(SECTION_ALIASES.get(canon, [canon]))
    lines = text.splitlines()

    heads = []  # (line_index, lowercase name, is_numbered)
    for i, line in enumerate(lines):
        if len(line) > 90:
            continue
        m = _HEAD_RE.match(line)
        if not m:
            continue
        num = (m.group(1) or "").strip().rstrip(".")
        raw_name = m.group(2).strip()
        name = raw_name.lower()
        known = name in _NAME_TO_CANON
        top_numbered = bool(num) and "." not in num
        generic = top_numbered and raw_name[0].isupper() and len(name.split()) <= 8
        if known or generic:
            heads.append((i, name, bool(num)))

    # numbered headings first (more reliable), then earliest position
    cands = sorted((h for h in heads if h[1] in targets), key=lambda h: (not h[2], h[0]))
    for idx, _, _ in cands:
        end = next((h[0] for h in heads if h[0] > idx), len(lines))
        body = re.sub(r"\s+", " ", " ".join(lines[idx + 1:end])).strip()
        if len(body) >= 200:
            return body[:max_chars]

    # inline fallbacks: "Abstract—We propose ..." / "5 Conclusion We presented ..."
    flat = re.sub(r"\s+", " ", text)
    if canon == "abstract":
        m = re.search(r"(?i)\babstract\b[\s.:\u2014\u2013\-]*", flat[:8000])
        if m:
            rest = flat[m.end():]
            stop = re.search(r"(?i)\b(?:1\.?\s+|I\.\s+)?introduction\b", rest)
            body = rest[:stop.start()] if stop else rest
            if len(body.strip()) >= 100:
                return body.strip()[:max_chars]
        return ""
    variants = sorted({v for a in targets for v in (a.title(), a.upper())}, key=len, reverse=True)
    alt = "|".join(re.escape(v) for v in variants)
    m = re.search(rf"\b\d{{1,2}}\.?\s+(?:{alt})\b[\s.:]*", flat)
    if m:
        rest = flat[m.end():]
        stop = re.search(r"\b(?:References|REFERENCES|Acknowledg(?:e)?ments?)\b", rest)
        body = (rest[:stop.start()] if stop else rest).strip()
        if len(body) >= 200:
            return body[:max_chars]
    return ""


# %% [markdown]
# ## Corpus layer — read every PDF, chunk it, and index it (hybrid retriever)
#
# Reads all PDFs in `PDF_DIR`, extracts light metadata (title / authors / year), chunks
# the text, and keeps a per-paper record (metadata + abstract + full text) for the new
# list_papers / read_section tools. Everything is cached in `corpus_cache.pkl`.

# %%
def _read_pdf(path: Path) -> dict | None:
    try:
        try:
            from pypdf import PdfReader
        except ImportError:                       # older name
            from PyPDF2 import PdfReader           # type: ignore
    except ImportError as e:
        raise RuntimeError("pypdf is required to read PDFs: pip install pypdf") from e

    try:
        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            return None
        meta = reader.metadata or {}
        title = (getattr(meta, "title", None) or "").strip()
        authors = (getattr(meta, "author", None) or "").strip()
    except Exception as exc:                       # a corrupt / image-only PDF shouldn't kill the run
        print(f"  [skip] {path.name}: {exc}")
        return None

    head = "\n".join(pages[:2])
    # junk metadata titles ("untitled", "Microsoft Word - draft.docx", "paper.pdf") -> ignore
    if len(title) < 6 or title.lower() in ("untitled", "title") or \
            title.lower().startswith("microsoft word") or title.lower().endswith((".pdf", ".docx", ".dvi")):
        title = ""
    if not title:                                # heuristic: first substantial line of page 1
        for line in head.splitlines():
            s = line.strip()
            if len(s) > 15:
                title = s
                break
    year = ""
    m = re.search(r"\b(19|20)\d{2}\b", head) or re.search(r"\b(19|20)\d{2}\b", path.name)
    if m:
        year = m.group(0)

    return {
        "doc_id": path.name,
        "path": str(path),
        "title": title or path.stem,
        "authors": authors,
        "year": year,
        "text": text,
    }


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _corpus_signature(pdf_dir: Path) -> str:
    parts = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    sig = "|".join(parts) + f"|{CHUNK_CHARS}:{CHUNK_OVERLAP}:{EMBED_MODEL}:{CACHE_VERSION}"
    return hashlib.md5(sig.encode()).hexdigest()


def build_corpus(pdf_dir: str) -> tuple[list[str], list[dict], dict[str, dict]]:
    """Return (chunk_texts, chunk_metas, papers). Cached across runs."""
    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus folder not found: {pdf_dir}")

    sig = _corpus_signature(root)
    if CACHE_PATH.exists():
        try:
            cached = pickle.loads(CACHE_PATH.read_bytes())
            if cached.get("sig") == sig and "papers" in cached:
                print(f"Loaded cached corpus ({len(cached['texts'])} chunks, "
                      f"{len(cached['papers'])} papers) from {CACHE_PATH.name}")
                return cached["texts"], cached["metas"], cached["papers"]
        except Exception:
            pass

    pdfs = sorted(root.glob("*.pdf"))
    print(f"Parsing {len(pdfs)} PDFs from {root} ...")
    texts: list[str] = []
    metas: list[dict] = []
    papers: dict[str, dict] = {}
    for p in pdfs:
        doc = _read_pdf(p)
        if not doc:
            continue
        chunks = _chunk(doc["text"], CHUNK_CHARS, CHUNK_OVERLAP)
        for n, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({
                "chunk_id": f"{doc['doc_id']}#c{n}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "authors": doc["authors"],
                "year": doc["year"],
            })
        abstract = extract_section(doc["text"], "abstract", 600) or \
            re.sub(r"\s+", " ", doc["text"])[:600]
        papers[doc["doc_id"]] = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "authors": doc["authors"],
            "year": doc["year"],
            "abstract": abstract,
            "n_chunks": len(chunks),
            "text": doc["text"],                   # raw text WITH line breaks (for sections)
        }
    print(f"Built {len(texts)} chunks from {len(papers)} readable PDFs ({len(pdfs)} found).")
    try:
        CACHE_PATH.write_bytes(pickle.dumps(
            {"sig": sig, "texts": texts, "metas": metas, "papers": papers}))
    except Exception as e:
        print(f"  (could not write cache: {e})")
    return texts, metas, papers


def _format_paper(p: dict) -> str:
    line = f"- {p['doc_id']} ({p.get('year') or 'n.d.'}) | {p['title'][:120]}"
    if p.get("authors"):
        line += f" | {p['authors'][:60]}"
    if p.get("abstract"):
        line += f"\n    {p['abstract'][:200]}"
    return line


# %%
class HybridRetriever:
    """Hybrid retriever + cross-encoder reranker + paper-level access.

    Semantic component: local sentence-transformers embeddings + cosine similarity.
    Lexical component: BM25 (falls back to token-overlap if rank_bm25 is absent).
    Scores from each are min-max normalized and fused with HYBRID_ALPHA; the top
    RERANK_CANDIDATES are then rescored by a cross-encoder (if available).
    """

    def __init__(self, texts: list[str], metas: list[dict], papers: dict[str, dict],
                 alpha: float = HYBRID_ALPHA, use_reranker: bool = USE_RERANKER):
        self.texts = texts
        self.metas = metas
        self.papers = papers
        self.alpha = alpha
        self._doc_idx: dict[str, list[int]] = {}
        for i, m in enumerate(metas):
            self._doc_idx.setdefault(m["doc_id"], []).append(i)

        # --- lexical index ---
        self._tokenized = [tokenize(t) for t in texts]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized)
        except ImportError:
            self._bm25 = None
            print("  [note] rank_bm25 not installed -> using simple token-overlap lexical scoring.")

        # --- semantic index ---
        self._embedder = None
        self._doc_emb = None
        try:
            st_mod = importlib.import_module("sentence_transformers")
            st_cls = getattr(st_mod, "SentenceTransformer")
            print(f"Loading embedding model '{EMBED_MODEL}' and encoding {len(texts)} chunks ...")
            self._embedder = st_cls(EMBED_MODEL)
            emb = self._embedder.encode(texts, batch_size=64, show_progress_bar=True,
                                        normalize_embeddings=True)
            self._doc_emb = np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"  [note] semantic embeddings unavailable ({e}) -> lexical-only mode.")

        # --- NEW: cross-encoder reranker ---
        self._reranker = None
        if use_reranker:
            try:
                st_mod = importlib.import_module("sentence_transformers")
                ce_cls = getattr(st_mod, "CrossEncoder")
                print(f"Loading reranker '{RERANK_MODEL}' ...")
                self._reranker = ce_cls(RERANK_MODEL, max_length=512)
            except Exception as e:
                print(f"  [note] reranker unavailable ({type(e).__name__}: {str(e)[:120]}) -> "
                      "no reranking. If the model isn't downloaded yet, set "
                      "ALLOW_HF_DOWNLOAD = True for one run.")

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    # ---- scorers ----
    def _lexical_scores(self, query: str) -> np.ndarray:
        if self._bm25 is not None:
            return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        q = set(tokenize(query))
        return np.asarray([len(q & set(toks)) for toks in self._tokenized], dtype=float)

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if self._embedder is None or self._doc_emb is None:
            return None
        qv = self._embedder.encode([query], normalize_embeddings=True)[0]
        return self._doc_emb @ np.asarray(qv, dtype=np.float32)

    def _fused(self, query: str) -> np.ndarray:
        lex = _normalize(self._lexical_scores(query))
        sem = self._semantic_scores(query)
        if sem is None:
            return lex
        return self.alpha * _normalize(sem) + (1 - self.alpha) * lex

    def _hit(self, i: int, score: float) -> dict:
        m = self.metas[i]
        return {**m, "text": self.texts[i], "score": float(score), "hybrid_score": float(score)}

    # ---- NEW: reranking ----
    def rerank(self, query: str, hits: list[dict], k: int | None = None) -> list[dict]:
        """Rescore hits against `query` with the cross-encoder (score in [0,1]); sort desc.

        Without a reranker, falls back to sorting by the existing score.
        """
        if not hits:
            return []
        if self._reranker is None:
            out = sorted(hits, key=lambda h: h["score"], reverse=True)
            return out[:k] if k else out
        pairs = [(query, h["text"]) for h in hits]
        raw = np.asarray(self._reranker.predict(pairs, batch_size=32, show_progress_bar=False),
                         dtype=float).reshape(-1)
        probs = raw if (raw.min() >= 0 and raw.max() <= 1) else 1.0 / (1.0 + np.exp(-raw))
        out = [{**h, "score": float(p), "hybrid_score": h.get("hybrid_score", h["score"])}
               for h, p in zip(hits, probs)]
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:k] if k else out

    # ---- tool actions ----
    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        score = self._fused(query)
        n = RERANK_CANDIDATES if self._reranker is not None else k
        order = np.argsort(score)[::-1][:n]
        hits = [self._hit(int(i), score[int(i)]) for i in order if score[int(i)] > 0]
        if self._reranker is not None:
            hits = self.rerank(query, hits, k)
        return hits

    def search_by_author(self, author: str = "", year: str = "",
                         query: str = "", k: int = TOP_K) -> list[dict]:
        author = (author or "").lower().strip()
        year = str(year or "").strip()

        def match(m: dict) -> bool:
            a_ok = bool(author) and author in (m.get("authors") or "").lower()
            y_ok = bool(year) and year == str(m.get("year") or "")
            if author and year:
                return a_ok and y_ok
            return a_ok or y_ok

        idxs = [i for i, m in enumerate(self.metas) if match(m)]
        if not idxs and author and year:                    # relax AND -> OR if nothing matched
            idxs = [i for i, m in enumerate(self.metas)
                    if author in (m.get("authors") or "").lower()
                    or year == str(m.get("year") or "")]
        if not idxs:
            return []
        score = self._fused(query) if query else np.zeros(len(self.texts))
        idxs.sort(key=lambda i: score[i], reverse=True)
        if query and self._reranker is not None:
            cands = [self._hit(i, score[i]) for i in idxs[:RERANK_CANDIDATES]]
            return self.rerank(query, cands, k)
        return [self._hit(i, score[i]) for i in idxs[:k]]

    def neighbors(self, seed_text: str, k: int = TOP_K, exclude: set[str] | None = None) -> list[dict]:
        """Semantically nearest chunks to a seed passage — the fallback for follow_citation."""
        exclude = exclude or set()
        score = self._fused(seed_text)
        order = np.argsort(score)[::-1]
        out = []
        for i in order:
            i = int(i)
            if self.metas[i]["chunk_id"] in exclude:
                continue
            if score[i] <= 0:
                break
            out.append(self._hit(i, score[i]))
            if len(out) >= k:
                break
        return out

    # ---- NEW: paper-level access for list_papers / read_section ----
    def resolve_doc_id(self, raw: str) -> str | None:
        """Map what the planner wrote (file name, partial name or title) to a real doc_id."""
        r = (raw or "").strip().strip("[]").strip()
        if not r:
            return None
        if r in self.papers:
            return r
        low = {k.lower(): k for k in self.papers}
        rl = r.lower()
        if rl in low:
            return low[rl]
        if rl + ".pdf" in low:
            return low[rl + ".pdf"]
        for k, p in self.papers.items():
            if rl in k.lower() or rl in (p.get("title") or "").lower():
                return k
        titles = {(p.get("title") or "").lower(): k for k, p in self.papers.items()}
        close = difflib.get_close_matches(rl, list(low) + list(titles), n=1, cutoff=0.6)
        if close:
            return low.get(close[0]) or titles.get(close[0])
        return None

    def search_in_doc(self, doc_id: str, query: str, k: int = 3) -> list[dict]:
        idxs = list(self._doc_idx.get(doc_id, []))
        if not idxs:
            return []
        score = self._fused(query)
        idxs.sort(key=lambda i: score[i], reverse=True)
        cands = [self._hit(i, score[i]) for i in idxs[:RERANK_CANDIDATES]]
        return self.rerank(query, cands, k)

    def read_section(self, doc_id: str, section: str, question: str = "") -> tuple[str, str]:
        """Return (section_text, how_it_was_found)."""
        p = self.papers.get(doc_id)
        if not p:
            return "", "unknown paper"
        txt = extract_section(p["text"], section)
        if txt:
            return txt, "extracted by heading"
        hits = self.search_in_doc(doc_id, f"{section.replace('_', ' ')} {question}".strip(), k=3)
        if hits:
            joined = " ... ".join(h["text"] for h in hits)
            return joined[:SECTION_MAX_CHARS], "heading not found; closest passages in this paper"
        return "", "not found"

    def list_papers(self, keyword: str = "", year: str = "",
                    limit: int = LIST_PAPERS_MAX) -> tuple[list[dict], int]:
        """Return (papers_to_show, total_matching). Year filters; keyword ranks by topic."""
        items = list(self.papers.values())
        year = str(year or "").strip()
        if year:
            items = [p for p in items if str(p.get("year") or "") == year]
        if keyword:
            score = self._fused(keyword)
            best: dict[str, float] = {}
            for i, s in enumerate(score):
                d = self.metas[i]["doc_id"]
                if s > best.get(d, 0.0):
                    best[d] = float(s)
            items = [p for p in items if best.get(p["doc_id"], 0.0) > 0]
            items.sort(key=lambda p: best[p["doc_id"]], reverse=True)
            total = len(items)
            if self._reranker is not None and items:
                pseudo = [{"text": f"{p['title']}. {p.get('abstract', '')}",
                           "score": best[p["doc_id"]], "_doc": p}
                          for p in items[:limit * 2]]
                items = [h["_doc"] for h in self.rerank(keyword, pseudo)]
            return items[:limit], total
        items.sort(key=lambda p: (str(p.get("year") or "0000"), p.get("title") or ""))
        return items[:limit], len(items)


# %% [markdown]
# ## The agent — a LangGraph state machine that chooses tools
#
# A `plan` node picks an action, a conditional edge routes to the matching tool node,
# tool nodes loop back to `plan`, and `answer` / `clarify` end the run. State accumulates
# across the loop and three guards make sure it always terminates.

# %%
_CITE_RE = re.compile(r"[A-Z][A-Za-z\-]+(?:\s+et al\.?| and [A-Z][A-Za-z\-]+)?[,]?\s*\(?(?:19|20)\d{2}\)?")


class _AgentState(TypedDict, total=False):
    original_query: str
    pending_queries: list[str]
    executed_queries: list[str]
    pending_author: str
    pending_year: str
    pending_keyword: str             # NEW (list_papers)
    pending_doc_ids: list[str]       # NEW (read_section)
    pending_section: str             # NEW (read_section)
    docs: dict[str, dict]            # chunk_id -> hit (accumulates, deduped by key)
    catalog: str                     # NEW: accumulated list_papers output
    iterations: int
    next_action: str
    clarification_question: str
    clarified: bool
    done: bool
    answer: str
    context_used: str                # NEW: the context the answer was written from (for the judge)


class ToolUsingAgent:
    def __init__(self, retriever: HybridRetriever, llm: ChatOpenAI,
                 max_steps: int = MAX_STEPS, debug: bool = DEBUG,
                 enable_paper_tools: bool = ENABLE_PAPER_TOOLS):
        self.r = retriever
        self.llm = llm
        self._max_steps = max_steps
        self.debug = debug
        self.enable_paper_tools = enable_paper_tools
        self.plan_system, self.actions = build_plan_system(enable_paper_tools)
        self._calls = 0
        self._graph = self._build_graph()

    # ---------- helpers ----------
    def _ranked_docs(self, state: _AgentState, n: int) -> list[dict]:
        """Whole sections first (explicitly requested), then chunks reranked vs the ORIGINAL question."""
        docs = list(state.get("docs", {}).values())
        pinned = [d for d in docs if d.get("section")]
        rest = self.r.rerank(state["original_query"], [d for d in docs if not d.get("section")])
        return (pinned + rest)[:n]

    # ---------- nodes ----------
    def _plan_node(self, state: _AgentState) -> dict:
        # GUARD 1: hit the step cap -> answer without spending a planning call.
        if state.get("iterations", 0) >= self._max_steps:
            if self.debug:
                print(f"  [plan] step cap reached ({self._max_steps}) -> answer")
            log("PLAN", f"step cap reached ({self._max_steps}) -> answer")
            return {"next_action": "answer"}

        docs = state.get("docs", {})
        top = self._ranked_docs(state, DECIDE_CONTEXT_CHUNKS)
        passages = "\n\n---\n\n".join(
            f"[{d['doc_id']} | {d.get('authors', '')} {d.get('year', '')}"
            f"{' | SECTION: ' + d['section'] if d.get('section') else ''}]\n{d['text'][:500]}"
            for d in top
        ) or "(none yet)"
        catalog = state.get("catalog", "")
        if catalog:
            cat_txt = catalog[:CATALOG_PLAN_CHARS]
            if len(catalog) > CATALOG_PLAN_CHARS:
                cat_txt += "\n...[catalog truncated]"
        else:
            cat_txt = "(not listed yet)"
        executed = state.get("executed_queries", [])
        user = (
            f"Question: {state['original_query']}\n\n"
            f"Actions already run: {executed or '(none)'}\n\n"
            f"Already asked the user a clarifying question: {state.get('clarified', False)}\n\n"
            f"Paper catalog:\n{cat_txt}\n\n"
            f"Passages found so far ({len(docs)} total, showing top {len(top)}):\n\n{passages}"
        )
        self._calls += 1
        resp = self.llm.invoke([SystemMessage(content=self.plan_system), HumanMessage(content=user)])
        raw = _strip_fences(resp.content if hasattr(resp, "content") else str(resp))
        try:                                           # GUARD 2: bad JSON -> answer
            d = json.loads(raw)
            action = d.get("action", "answer")
            reasoning = d.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            action, d, reasoning = "answer", {}, "parse-fail -> stop"

        # paper tools switched off (for a "before" run) -> plain search instead
        if action in ("list_papers", "read_section") and not self.enable_paper_tools:
            action, reasoning = "semantic_search", f"{action} disabled -> search instead"
            if not d.get("queries"):
                d["queries"] = [state["original_query"]]

        # don't clarify twice
        if action == "clarify" and state.get("clarified"):
            action, reasoning = "semantic_search", "already clarified -> search instead"
            if not d.get("queries"):
                d["queries"] = [state["original_query"]]

        # don't re-run queries we've already executed (semantic_search only)
        new_q = [q for q in (d.get("queries") or []) if q not in executed]
        if action == "semantic_search" and not new_q:
            action, reasoning = "answer", "no new queries left -> stop"

        keyword = str(d.get("keyword", "") or "").strip()
        year = str(d.get("year", "") or "").strip()
        if action == "list_papers" and f"list_papers(year='{year}', keyword='{keyword}')" in executed:
            action, reasoning = "answer", "catalog already listed with these filters -> stop"

        section = canonical_section(str(d.get("section", "") or ""))
        doc_ids = [str(x).strip() for x in (d.get("doc_ids") or []) if str(x).strip()]
        if action == "read_section" and not doc_ids:     # planner forgot -> best papers so far
            doc_ids = list(dict.fromkeys(x["doc_id"] for x in top))[:MAX_SECTION_DOCS]
        new_ids = [x for x in doc_ids if f"read_section:{x}:{section}" not in executed]
        if action == "read_section" and not new_ids:
            action, reasoning = "answer", "no new sections to read -> stop"

        if self.debug:
            print(f"  [plan] action={action}  ({reasoning[:70]})")
        log("PLAN", json.dumps({
            "question": state["original_query"][:120], "step": state.get("iterations", 0),
            "action": action, "reasoning": reasoning, "queries": new_q, "keyword": keyword,
            "year": year, "doc_ids": new_ids, "section": section,
        }, ensure_ascii=False))

        return {
            "next_action": action,
            "pending_queries": new_q,
            "pending_author": d.get("author", "") or "",
            "pending_year": year,
            "pending_keyword": keyword,
            "pending_doc_ids": new_ids[:MAX_SECTION_DOCS],
            "pending_section": section,
            "clarification_question": d.get("clarification", "") or "",
        }

    def _retrieve_node(self, state: _AgentState) -> dict:
        docs = dict(state.get("docs", {}))
        executed = list(state.get("executed_queries", []))
        for q in state.get("pending_queries", []):
            hits = self.r.search(q)
            for h in hits:
                docs[h["chunk_id"]] = h
            executed.append(q)
            if self.debug:
                print(f"  [semantic_search] '{q}' -> +{len(hits)} hits (total {len(docs)})")
        return {"docs": docs, "executed_queries": executed,
                "pending_queries": [], "iterations": state.get("iterations", 0) + 1}

    def _by_author_node(self, state: _AgentState) -> dict:
        docs = dict(state.get("docs", {}))
        executed = list(state.get("executed_queries", []))
        author, year = state.get("pending_author", ""), state.get("pending_year", "")
        hits = self.r.search_by_author(author=author, year=year,
                                       query=state["original_query"])
        for h in hits:
            docs[h["chunk_id"]] = h
        executed.append(f"by_author(author='{author}', year='{year}')")
        if self.debug:
            print(f"  [find_by_author] author='{author}' year='{year}' -> +{len(hits)} hits")
        return {"docs": docs, "executed_queries": executed,
                "iterations": state.get("iterations", 0) + 1}

    def _list_papers_node(self, state: _AgentState) -> dict:
        executed = list(state.get("executed_queries", []))
        keyword, year = state.get("pending_keyword", ""), state.get("pending_year", "")
        items, total = self.r.list_papers(keyword=keyword, year=year)
        header = (f"list_papers(year='{year}', keyword='{keyword}') -> showing {len(items)} of "
                  f"{total} matching papers ({len(self.r.papers)} papers in the corpus)"
                  + (" [ranked by relevance to keyword]" if keyword else ""))
        body = "\n".join(_format_paper(p) for p in items) or "(no papers matched)"
        prev = state.get("catalog", "")
        catalog = (prev + "\n\n" if prev else "") + header + "\n" + body
        executed.append(f"list_papers(year='{year}', keyword='{keyword}')")
        if self.debug:
            print(f"  [list_papers] year='{year}' keyword='{keyword}' -> {len(items)} of {total}")
        return {"catalog": catalog, "executed_queries": executed,
                "iterations": state.get("iterations", 0) + 1}

    def _read_section_node(self, state: _AgentState) -> dict:
        docs = dict(state.get("docs", {}))
        executed = list(state.get("executed_queries", []))
        section = state.get("pending_section") or "abstract"
        for raw in state.get("pending_doc_ids", [])[:MAX_SECTION_DOCS]:
            executed.append(f"read_section:{raw}:{section}")
            doc_id = self.r.resolve_doc_id(raw)
            if not doc_id:
                if self.debug:
                    print(f"  [read_section] '{raw}' -> no such paper")
                continue
            text, how = self.r.read_section(doc_id, section, question=state["original_query"])
            if self.debug:
                print(f"  [read_section] {doc_id} :: {section} -> {len(text)} chars ({how})")
            if not text:
                continue
            p = self.r.papers[doc_id]
            key = f"{doc_id}#sec:{section}"
            docs[key] = {
                "chunk_id": key, "doc_id": doc_id, "title": p["title"],
                "authors": p["authors"], "year": p["year"], "section": section,
                "text": f"({section} section — {how}) {text}", "score": 1.0,
            }
        return {"docs": docs, "executed_queries": executed,
                "iterations": state.get("iterations", 0) + 1}

    def _follow_citation_node(self, state: _AgentState) -> dict:
        docs = dict(state.get("docs", {}))
        executed = list(state.get("executed_queries", []))
        top = self._ranked_docs(state, 3)

        # 1) try to read citation-like tokens out of the best passages
        cites: list[str] = []
        for d in top:
            cites += _CITE_RE.findall(d["text"])
        cites = list(dict.fromkeys(c.strip() for c in cites))[:2]

        added = 0
        if cites:
            for c in cites:
                hits = self.r.search(c)
                for h in hits:
                    if h["chunk_id"] not in docs:
                        docs[h["chunk_id"]] = h
                        added += 1
                executed.append(f"citation:{c}")
            if self.debug:
                print(f"  [follow_citation] tokens {cites} -> +{added} hits")
        else:
            # 2) fallback: semantic neighbors of the best passage (closely related work)
            if top:
                hits = self.r.neighbors(top[0]["text"], exclude=set(docs.keys()))
                for h in hits:
                    docs[h["chunk_id"]] = h
                    added += 1
                executed.append("related:neighbors")
            if self.debug:
                print(f"  [follow_citation] no citation tokens -> {added} related neighbors")
        return {"docs": docs, "executed_queries": executed,
                "iterations": state.get("iterations", 0) + 1}

    def _clarify_node(self, state: _AgentState) -> dict:
        q = state.get("clarification_question") or "Could you specify which topic or paper you mean?"
        if self.debug:
            print(f"  [clarify] {q}")
        return {"answer": f"[Clarifying question] {q}", "clarified": True, "done": True,
                "context_used": ""}

    def _answer_node(self, state: _AgentState) -> dict:
        top = self._ranked_docs(state, ANSWER_CONTEXT_CHUNKS)
        parts = []
        if state.get("catalog"):
            parts.append("PAPER CATALOG\n" + state["catalog"])
        parts += [f"[{d['doc_id']}]\n{d['text']}" for d in top]
        context = "\n\n".join(parts) or "(no documents)"
        self._calls += 1
        resp = self.llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['original_query']}"),
        ])
        ans = resp.content if hasattr(resp, "content") else str(resp)
        return {"answer": ans, "done": True, "context_used": context}

    # ---------- routing ----------
    def _route_from_plan(self, state: _AgentState) -> str:
        # GUARD 3: only these actions are legal; anything else falls through to answer.
        return {
            "semantic_search": "retrieve",
            "find_by_author": "retrieve_by_author",
            "list_papers": "list_papers",
            "read_section": "read_section",
            "follow_citation": "follow_citation",
            "clarify": "clarify",
            "answer": "answer",
        }.get(state.get("next_action", "answer"), "answer")

    def _build_graph(self):
        g = StateGraph(_AgentState)
        g.add_node("plan", self._plan_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("retrieve_by_author", self._by_author_node)
        g.add_node("list_papers", self._list_papers_node)
        g.add_node("read_section", self._read_section_node)
        g.add_node("follow_citation", self._follow_citation_node)
        g.add_node("clarify", self._clarify_node)
        g.add_node("answer", self._answer_node)

        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self._route_from_plan, {
            "retrieve": "retrieve",
            "retrieve_by_author": "retrieve_by_author",
            "list_papers": "list_papers",
            "read_section": "read_section",
            "follow_citation": "follow_citation",
            "clarify": "clarify",
            "answer": "answer",
        })
        for tool_node in ("retrieve", "retrieve_by_author", "list_papers",
                          "read_section", "follow_citation"):
            g.add_edge(tool_node, "plan")
        g.add_edge("clarify", END)
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str) -> dict:
        """Returns {'answer', 'context', 'steps', 'llm_calls', 'actions'}."""
        self._calls = 0
        init: _AgentState = {
            "original_query": question,
            "pending_queries": [], "executed_queries": [],
            "pending_author": "", "pending_year": "", "pending_keyword": "",
            "pending_doc_ids": [], "pending_section": "",
            "docs": {}, "catalog": "", "iterations": 0,
            "next_action": "", "clarification_question": "",
            "clarified": False, "done": False, "answer": "", "context_used": "",
        }
        final = self._graph.invoke(init, config={"recursion_limit": 25})
        return {
            "answer": final.get("answer", "(no answer produced)"),
            "context": final.get("context_used", ""),
            "steps": final.get("iterations", 0),
            "llm_calls": self._calls,
            "actions": final.get("executed_queries", []),
        }


# %% [markdown]
# ## Fixed single-pass baseline (Checkpoint 2.1 style) — for the evaluation comparison

# %%
def fixed_pipeline_answer(retriever: HybridRetriever, llm: ChatOpenAI, question: str) -> dict:
    """One search, one answer. No looping, no tool choice — the baseline to beat."""
    hits = retriever.search(question, k=TOP_K)
    context = "\n\n".join(f"[{h['doc_id']}]\n{h['text']}" for h in hits) or "(no documents)"
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    ans = resp.content if hasattr(resp, "content") else str(resp)
    return {"answer": ans, "context": context, "steps": 1, "llm_calls": 1,
            "actions": [question]}


# %% [markdown]
# ## NEW — LLM-judge (1-5) and evaluation record keeping

# %%
def judge_answer(judge_llm: ChatOpenAI, question: str, answer: str, context: str) -> dict:
    """Score one response 1-5 on relevance, faithfulness, completeness and overall."""
    ctx = context or "(no passages — the system replied with a clarifying question)"
    if len(ctx) > JUDGE_CONTEXT_CHARS:
        ctx = ctx[:JUDGE_CONTEXT_CHARS] + "\n...[truncated]"
    msg = f"QUESTION:\n{question}\n\nPASSAGES:\n{ctx}\n\nRESPONSE:\n{answer}"
    try:
        resp = judge_llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=msg)])
        raw = _strip_fences(resp.content if hasattr(resp, "content") else str(resp))
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0) if m else raw)
        out: dict[str, Any] = {k: max(1, min(5, int(round(float(d[k]))))) for k in JUDGE_KEYS}
        out["rationale"] = str(d.get("rationale", ""))[:400]
    except Exception as e:
        out = {k: None for k in JUDGE_KEYS}
        out["rationale"] = f"judge failed: {type(e).__name__}: {e}"
    return out


def current_settings() -> dict:
    return {
        "llm_model": LLM_MODEL, "judge_model": JUDGE_MODEL, "temperature": TEMPERATURE,
        "max_steps": MAX_STEPS, "top_k": TOP_K, "hybrid_alpha": HYBRID_ALPHA,
        "use_reranker": USE_RERANKER, "rerank_candidates": RERANK_CANDIDATES,
        "paper_tools": ENABLE_PAPER_TOOLS, "embed_model": EMBED_MODEL,
    }


def save_eval_record(rec: dict) -> None:
    with EVAL_RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _avg(vals: list) -> str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return f"{sum(vals) / len(vals):.2f}" if vals else "  - "


def print_scoreboard(rows: list[dict]) -> None:
    print("\nLLM-JUDGE SCOREBOARD (1-5)  —  this run")
    print(f"{'Q':<4}{'system':<8}{'relev':>6}{'faith':>6}{'compl':>6}{'overall':>8}"
          f"{'steps':>7}{'calls':>7}")
    for r in rows:
        s = r["scores"]
        cells = "".join(f"{(s.get(k) if s.get(k) is not None else '-'):>6}" for k in JUDGE_KEYS[:3])
        ov = s.get("overall") if s.get("overall") is not None else "-"
        print(f"{r['q_id']:<4}{r['system']:<8}{cells}{ov:>8}{r['steps']:>7}{r['llm_calls']:>7}")
    for system in ("agent", "fixed"):
        sub = [r for r in rows if r["system"] == system]
        if sub:
            avgs = "".join(f"{_avg([r['scores'].get(k) for r in sub]):>6}" for k in JUDGE_KEYS[:3])
            print(f"{'avg':<4}{system:<8}{avgs}{_avg([r['scores'].get('overall') for r in sub]):>8}"
                  f"{_avg([r['steps'] for r in sub]):>7}{_avg([r['llm_calls'] for r in sub]):>7}")


def print_eval_history() -> None:
    """Average judge scores per (RUN_LABEL, system) across every run in eval_results.jsonl."""
    if not EVAL_RESULTS_PATH.exists():
        return
    groups: dict[tuple[str, str], list[dict]] = {}
    for line in EVAL_RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        groups.setdefault((r.get("label", "?"), r.get("system", "?")), []).append(r)
    print("\nEVALUATION HISTORY — average scores per run label (compare before vs after)")
    print(f"{'label':<46}{'system':<8}{'n':>4}{'faith':>7}{'compl':>7}{'overall':>8}{'calls':>7}")
    for (label, system), rs in groups.items():
        print(f"{label[:45]:<46}{system:<8}{len(rs):>4}"
              f"{_avg([r['scores'].get('faithfulness') for r in rs]):>7}"
              f"{_avg([r['scores'].get('completeness') for r in rs]):>7}"
              f"{_avg([r['scores'].get('overall') for r in rs]):>8}"
              f"{_avg([r['llm_calls'] for r in rs]):>7}")


# %% [markdown]
# ## Step 2 — Your agent design (the plan you describe in the written submission)

# %%
def my_agent_plan() -> dict[str, Any]:
    """Design of the agent implemented above, for the Research Paper Navigator scenario."""
    return {
        "scenario": "research_papers (~150 research-paper PDFs)",
        "tools": AVAILABLE_ACTIONS,
        "retrieval": (
            f"hybrid BM25 + embeddings retrieves {RERANK_CANDIDATES} candidates, a "
            f"cross-encoder reranks them to TOP_K={TOP_K}; the planner and the answer step "
            "rerank all accumulated passages against the original question"
            if USE_RERANKER else "hybrid BM25 + embeddings, no reranking"),
        "stop_condition": (
            "The planner returns action='answer' once the accumulated passages cover every "
            "part of the question. Guards force termination regardless: (1) a hard cap of "
            "MAX_STEPS tool rounds, checked before the planning call; (2) any JSON parse "
            "failure defaults to answer; (3) if the planner repeats a search, catalog "
            "listing or section read that already ran, it is redirected to answer."
        ),
        "system_prompt_idea": (
            "A single planner prompt lists the actions and their JSON contract: search for "
            "topics, find_by_author for named authors, list_papers for collection-wide "
            "questions, read_section when specific papers' sections are needed, "
            "follow_citation for related work, clarify at most once, and answer as soon as "
            "the passages suffice."
        ),
        "evaluation": "LLM-judge scores 1-5 (relevance, faithfulness, completeness, overall) "
                      "for agent and baseline, stored per RUN_LABEL in eval_results.jsonl",
        "test_tasks": [t["q"] for t in TEST_QUESTIONS],
    }


# %% [markdown]
# ## Step 3 — Test questions that need more than one tool / more than one round

# %%
TEST_QUESTIONS = [
    {
        "q": ("What are the main limitations reported for the methods in this collection, "
              "and what related or follow-up work is cited to address them?"),
        "exercises": "semantic_search -> read_section(limitations) -> follow_citation -> answer",
    },
    {
        "q": ("Summarize the papers from 2023 in this collection and how their contributions "
              "differ from the earlier work on the same topic."),
        "exercises": "list_papers(year=2023) -> read_section(abstract) / semantic_search -> answer",
    },
    {
        "q": "Tell me about the evaluation.",
        "exercises": "clarify (too vague to search — evaluation of what?)",
    },
]


# %% [markdown]
# ## Step 4 — Run the agent on each question, compare against the baseline, and judge both

# %%
def run() -> None:
    print(f"Checkpoint 5.1 — agent-based RAG over the full corpus  |  scenario: {SCENARIO}")
    print(f"Corpus: {PDF_DIR}")
    print(f"Run label: {RUN_LABEL}\n")

    if not Path(PDF_DIR).exists():
        print(f"[!] Corpus folder not found: {PDF_DIR}")
        print("    Fix PDF_DIR at the top of this file, then rerun.")
        print("\nYour agent plan (design still prints without the corpus):")
        print(json.dumps(my_agent_plan(), indent=2))
        return

    texts, metas, papers = build_corpus(PDF_DIR)
    retriever = HybridRetriever(texts, metas, papers)
    llm = make_llm()
    agent = ToolUsingAgent(retriever, llm)
    judge_llm = make_llm(temperature=0.0, model=JUDGE_MODEL) if RUN_JUDGE else None
    run_id = datetime.now().isoformat(timespec="seconds")
    settings = {**current_settings(), "reranker_loaded": retriever.has_reranker}
    rows: list[dict] = []

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["q"]
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        print(f"(exercises: {item['exercises']})\n")

        print("--- AGENT (multi-step) ---")
        agent_res = agent.run(q)
        print(f"\nAgent answer:\n{agent_res['answer']}\n")

        print("--- FIXED PIPELINE (single search, Checkpoint 2.1 baseline) ---")
        fixed_res = fixed_pipeline_answer(retriever, llm, q)
        print(f"Fixed answer:\n{fixed_res['answer']}\n")

        for system, res in (("agent", agent_res), ("fixed", fixed_res)):
            scores = (judge_answer(judge_llm, q, res["answer"], res["context"])
                      if judge_llm else {k: None for k in JUDGE_KEYS})
            if judge_llm:
                print(f"[judge:{system}] " + "  ".join(f"{k}={scores.get(k)}" for k in JUDGE_KEYS)
                      + f"\n    {scores.get('rationale', '')}")
            row = {
                "run_id": run_id, "label": RUN_LABEL, "settings": settings,
                "q_id": f"Q{i}", "question": q, "system": system,
                "answer": res["answer"], "scores": scores, "steps": res["steps"],
                "llm_calls": res["llm_calls"],        # agent/baseline calls (judge not counted)
                "actions": res["actions"],
            }
            rows.append(row)
            save_eval_record(row)
            log(f"Q{i} {system.upper()}",
                f"Q: {q}\nActions: {res['actions']}\nA: {res['answer']}\nJudge: {scores}")

    print("\n" + "=" * 72)
    if RUN_JUDGE:
        print_scoreboard(rows)
        print_eval_history()
    print("\n" + "=" * 72)
    print("Your agent plan:")
    print(json.dumps(my_agent_plan(), indent=2))
    print("=" * 72)
    print(f"Done. Planner decisions are logged to {LOG_PATH.name}; judge scores are in "
          f"{EVAL_RESULTS_PATH.name}.")


if __name__ == "__main__":
    run()
