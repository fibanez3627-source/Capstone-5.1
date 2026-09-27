r"""Capstone Checkpoint 5.1 — Agent-Based RAG over the FULL research-paper corpus (completed).

This is the completed version of "5.1 Capstone starter modified.py". It follows the
Lab 5.1 README (the retrieve -> analyze loop, accumulating state, three stopping guards)
and the Lab 5.2 tool-using-agent snippets (a LangGraph StateGraph whose `plan` node
chooses among several tools). The toy 6-document sample is gone: the agent now runs over
every PDF in your capstone corpus.

Scenario: Research Paper Navigator (~150 research-paper PDFs).
Tools the agent can choose from:
    semantic_search  -> hybrid (embedding + lexical) search over paper chunks
    find_by_author   -> retrieve chunks filtered by author and/or year metadata
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
    #   OPENROUTER_API_KEY=sk-or-v1-...
sentence-transformers / rank-bm25 are optional — the retriever degrades gracefully to
lexical-only if either is missing (it just prints a note and keeps working).
--------------------------------------------------------------------------------
"""

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import hashlib
import importlib
import json
import os
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
TOP_K = 4                            # chunks returned per search
CHUNK_CHARS = 1100                    # ~ a paragraph or two
CHUNK_OVERLAP = 150
HYBRID_ALPHA = 0.5                    # 0 = pure lexical, 1 = pure semantic
EMBED_MODEL = "all-MiniLM-L6-v2"      # local sentence-transformers model (no extra API)
DECIDE_CONTEXT_CHUNKS = 15           # passages shown to the PLANNER (kept cheap)
ANSWER_CONTEXT_CHUNKS = 30           # passages shown to the ANSWER prompt (fuller)
DEBUG = True

LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"
CACHE_PATH = Path.cwd() / "corpus_cache.pkl"

# %%
PLAN_SYSTEM = (
    "You are the planner for an agentic research-paper assistant. You take exactly ONE "
    "action per step. The available actions are:\n"
    "- semantic_search: search the paper collection by meaning. Give 1-2 'queries'. Use "
    "for topics, concepts, methods, results.\n"
    "- find_by_author: retrieve papers filtered by 'author' and/or 'year' (a 4-digit "
    "year). Use when the question names an author or a year/time range.\n"
    "- follow_citation: from the papers already found, pull in cited or closely related "
    "work. Use AFTER an initial search when the question asks about follow-up, prior, "
    "related, or subsequent work.\n"
    "- clarify: ask the user ONE short question in 'clarification'. Use ONLY when the "
    "request is too ambiguous to search at all (missing which topic/paper/metric). Never "
    "clarify more than once.\n"
    "- answer: stop and answer, once the collected passages cover every part of the "
    "question.\n\n"
    "You are given the question, the queries already run, and the passages found so far. "
    "Choose the next action. Respond with ONLY a JSON object:\n"
    '{"action": "semantic_search|find_by_author|follow_citation|clarify|answer", '
    '"queries": ["..."], "author": "", "year": "", "clarification": "", '
    '"reasoning": "one sentence"}\n'
    "Do NOT repeat a query already run. Prefer to answer as soon as the passages are "
    "enough — extra rounds cost real money."
)

ANSWER_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided passages, "
    "citing the source file names in [brackets] where you rely on them. If the passages "
    "do not contain the answer, say so plainly instead of guessing."
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


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
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


# %% [markdown]
# ## Corpus layer — read every PDF, chunk it, and index it (hybrid retriever)
#
# This replaces the six-line toy corpus. It reads all PDFs in `PDF_DIR`, extracts light
# metadata (title / authors / year) for the `find_by_author` tool, chunks the text, and
# builds a hybrid index (local embeddings + BM25). The whole index is cached to
# `corpus_cache.pkl`, so only the first run pays the parsing/embedding cost — the same
# "reuse the DB" idea as the README's `chroma_db/`.

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
    if not title:                                  # heuristic: first substantial line of page 1
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
    sig = "|".join(parts) + f"|{CHUNK_CHARS}:{CHUNK_OVERLAP}:{EMBED_MODEL}"
    return hashlib.md5(sig.encode()).hexdigest()


def build_corpus(pdf_dir: str) -> tuple[list[str], list[dict]]:
    """Return (chunk_texts, chunk_metas). Cached across runs."""
    root = Path(pdf_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus folder not found: {pdf_dir}")

    sig = _corpus_signature(root)
    if CACHE_PATH.exists():
        try:
            cached = pickle.loads(CACHE_PATH.read_bytes())
            if cached.get("sig") == sig:
                print(f"Loaded cached corpus ({len(cached['texts'])} chunks) from {CACHE_PATH.name}")
                return cached["texts"], cached["metas"]
        except Exception:
            pass

    pdfs = sorted(root.glob("*.pdf"))
    print(f"Parsing {len(pdfs)} PDFs from {root} ...")
    texts: list[str] = []
    metas: list[dict] = []
    for p in pdfs:
        doc = _read_pdf(p)
        if not doc:
            continue
        for n, ch in enumerate(_chunk(doc["text"], CHUNK_CHARS, CHUNK_OVERLAP)):
            texts.append(ch)
            metas.append({
                "chunk_id": f"{doc['doc_id']}#c{n}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "authors": doc["authors"],
                "year": doc["year"],
            })
    print(f"Built {len(texts)} chunks from {len(pdfs)} PDFs.")
    try:
        CACHE_PATH.write_bytes(pickle.dumps({"sig": sig, "texts": texts, "metas": metas}))
    except Exception as e:
        print(f"  (could not write cache: {e})")
    return texts, metas


# %%
class HybridRetriever:
    """Lab 2.2's hybrid retriever, standing in as the agent's search *tool*.

    Semantic component: local sentence-transformers embeddings + cosine similarity.
    Lexical component: BM25 (falls back to token-overlap if rank_bm25 is absent).
    Scores from each are min-max normalized and fused with HYBRID_ALPHA.
    """

    def __init__(self, texts: list[str], metas: list[dict], alpha: float = HYBRID_ALPHA):
        self.texts = texts
        self.metas = metas
        self.alpha = alpha

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
            sentence_transformer_mod = importlib.import_module("sentence_transformers")
            sentence_transformer_cls = getattr(sentence_transformer_mod, "SentenceTransformer")
            print(f"Loading embedding model '{EMBED_MODEL}' and encoding {len(texts)} chunks ...")
            self._embedder = sentence_transformer_cls(EMBED_MODEL)
            emb = self._embedder.encode(texts, batch_size=64, show_progress_bar=True,
                                        normalize_embeddings=True)
            self._doc_emb = np.asarray(emb, dtype=np.float32)
        except Exception as e:
            print(f"  [note] semantic embeddings unavailable ({e}) -> lexical-only mode.")

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
        return {**m, "text": self.texts[i], "score": float(score)}

    # ---- the three tool actions the agent calls ----
    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        score = self._fused(query)
        order = np.argsort(score)[::-1][:k]
        return [self._hit(int(i), score[int(i)]) for i in order if score[int(i)] > 0]

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


# %% [markdown]
# ## The agent — a LangGraph state machine that chooses tools
#
# Mirrors the README's graph and the Lab 5.2 snippet: a `plan` node picks an action, a
# conditional edge routes to the matching tool node, tool nodes loop back to `plan`, and
# `answer` / `clarify` end the run. State accumulates across the loop and three guards
# make sure it always terminates.

# %%
_CITE_RE = re.compile(r"[A-Z][A-Za-z\-]+(?:\s+et al\.?| and [A-Z][A-Za-z\-]+)?[,]?\s*\(?(?:19|20)\d{2}\)?")


class _AgentState(TypedDict, total=False):
    original_query: str
    pending_queries: list[str]
    executed_queries: list[str]
    pending_author: str
    pending_year: str
    docs: dict[str, dict]            # chunk_id -> hit (accumulates, deduped by key)
    iterations: int
    next_action: str
    clarification_question: str
    clarified: bool
    done: bool
    answer: str


class ToolUsingAgent:
    def __init__(self, retriever: HybridRetriever, llm: ChatOpenAI,
                 max_steps: int = MAX_STEPS, debug: bool = DEBUG):
        self.r = retriever
        self.llm = llm
        self._max_steps = max_steps
        self.debug = debug
        self._graph = self._build_graph()

    # ---------- nodes ----------
    def _plan_node(self, state: _AgentState) -> dict:
        # GUARD 1: hit the step cap -> answer without spending a planning call.
        if state.get("iterations", 0) >= self._max_steps:
            if self.debug:
                print(f"  [plan] step cap reached ({self._max_steps}) -> answer")
            return {"next_action": "answer"}

        docs = state.get("docs", {})
        top = sorted(docs.values(), key=lambda d: d["score"], reverse=True)[:DECIDE_CONTEXT_CHUNKS]
        passages = "\n\n---\n\n".join(
            f"[{d['doc_id']} | {d.get('authors','')} {d.get('year','')}]\n{d['text'][:500]}"
            for d in top
        ) or "(none yet)"
        executed = state.get("executed_queries", [])
        user = (
            f"Question: {state['original_query']}\n\n"
            f"Queries already run: {executed or '(none)'}\n\n"
            f"Already asked the user a clarifying question: {state.get('clarified', False)}\n\n"
            f"Passages found so far ({len(docs)} total, showing top {len(top)}):\n\n{passages}"
        )
        resp = self.llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user)])
        raw = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if raw.startswith("```"):                      # strip ```json ... ``` fences
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:                                           # GUARD 2: bad JSON -> answer
            d = json.loads(raw)
            action = d.get("action", "answer")
            reasoning = d.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            action, d, reasoning = "answer", {}, "parse-fail -> stop"

        # don't clarify twice
        if action == "clarify" and state.get("clarified"):
            action, reasoning = "semantic_search", "already clarified -> search instead"
            if not d.get("queries"):
                d["queries"] = [state["original_query"]]

        # don't re-run queries we've already executed. NOTE: only semantic_search is
        # driven by 'queries'; follow_citation derives its own queries from passages
        # already found, so it must NOT be redirected here just for lacking 'queries'.
        new_q = [q for q in (d.get("queries") or []) if q not in executed]
        if action == "semantic_search" and not new_q:
            action, reasoning = "answer", "no new queries left -> stop"

        if self.debug:
            print(f"  [plan] action={action}  ({reasoning[:70]})")

        return {
            "next_action": action,
            "pending_queries": new_q,
            "pending_author": d.get("author", "") or "",
            "pending_year": str(d.get("year", "") or ""),
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

    def _follow_citation_node(self, state: _AgentState) -> dict:
        docs = dict(state.get("docs", {}))
        executed = list(state.get("executed_queries", []))
        top = sorted(docs.values(), key=lambda d: d["score"], reverse=True)[:3]

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
        return {"answer": f"[Clarifying question] {q}", "clarified": True, "done": True}

    def _answer_node(self, state: _AgentState) -> dict:
        docs = sorted(state.get("docs", {}).values(),
                      key=lambda d: d["score"], reverse=True)[:ANSWER_CONTEXT_CHUNKS]
        context = "\n\n".join(f"[{d['doc_id']}]\n{d['text']}" for d in docs) or "(no documents)"
        resp = self.llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Passages:\n{context}\n\nQuestion: {state['original_query']}"),
        ])
        ans = resp.content if hasattr(resp, "content") else str(resp)
        return {"answer": ans, "done": True}

    # ---------- routing ----------
    def _route_from_plan(self, state: _AgentState) -> str:
        # GUARD 3: only these actions are legal; anything else falls through to answer.
        return {
            "semantic_search": "retrieve",
            "find_by_author": "retrieve_by_author",
            "follow_citation": "follow_citation",
            "clarify": "clarify",
            "answer": "answer",
        }.get(state.get("next_action", "answer"), "answer")

    def _build_graph(self):
        g = StateGraph(_AgentState)
        g.add_node("plan", self._plan_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("retrieve_by_author", self._by_author_node)
        g.add_node("follow_citation", self._follow_citation_node)
        g.add_node("clarify", self._clarify_node)
        g.add_node("answer", self._answer_node)

        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self._route_from_plan, {
            "retrieve": "retrieve",
            "retrieve_by_author": "retrieve_by_author",
            "follow_citation": "follow_citation",
            "clarify": "clarify",
            "answer": "answer",
        })
        g.add_edge("retrieve", "plan")
        g.add_edge("retrieve_by_author", "plan")
        g.add_edge("follow_citation", "plan")
        g.add_edge("clarify", END)
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str) -> str:
        init: _AgentState = {
            "original_query": question,
            "pending_queries": [], "executed_queries": [],
            "pending_author": "", "pending_year": "",
            "docs": {}, "iterations": 0,
            "next_action": "", "clarification_question": "",
            "clarified": False, "done": False, "answer": "",
        }
        final = self._graph.invoke(init, config={"recursion_limit": 25})
        return final.get("answer", "(no answer produced)")


# %% [markdown]
# ## Fixed single-pass baseline (Checkpoint 2.1 style) — for the evaluation comparison

# %%
def fixed_pipeline_answer(retriever: HybridRetriever, llm: ChatOpenAI, question: str) -> str:
    """One search, one answer. No looping, no tool choice — the baseline to beat."""
    hits = retriever.search(question, k=TOP_K)
    context = "\n\n".join(f"[{h['doc_id']}]\n{h['text']}" for h in hits) or "(no documents)"
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Passages:\n{context}\n\nQuestion: {question}"),
    ])
    return resp.content if hasattr(resp, "content") else str(resp)


# %% [markdown]
# ## Step 2 — Your agent design (the plan you describe in the written submission)

# %%
def my_agent_plan() -> dict[str, Any]:
    """Design of the agent implemented above, for the Research Paper Navigator scenario."""
    return {
        "scenario": "research_papers (~150 research-paper PDFs)",
        "tools": ["semantic_search", "find_by_author", "follow_citation", "clarify", "answer"],
        "stop_condition": (
            "The planner returns action='answer' once the accumulated passages cover every "
            "part of the question. Three guards force termination regardless: (1) a hard cap "
            "of MAX_STEPS tool rounds, checked before the planning call; (2) any JSON parse "
            "failure defaults to answer; (3) if the planner asks to search but every proposed "
            "query was already run, it is redirected to answer."
        ),
        "system_prompt_idea": (
            "A single PLAN_SYSTEM prompt lists the five actions and their JSON contract and "
            "tells the agent to search first, use find_by_author when an author or year is "
            "named, use follow_citation only after an initial search when the question asks "
            "about related/prior work, clarify at most once when the request is unsearchable, "
            "and answer as soon as the passages suffice."
        ),
        "test_tasks": [t["q"] for t in TEST_QUESTIONS],
    }


# %% [markdown]
# ## Step 3 — Test questions that need more than one tool / more than one round
#
# Each is chosen to exercise a different part of the agent. Swap the author name and year
# in Q2 for ones that actually appear in your corpus before you run the graded comparison.

# %%
TEST_QUESTIONS = [
    {
        "q": ("What are the main limitations reported for the methods in this collection, "
              "and what related or follow-up work is cited to address them?"),
        "exercises": "semantic_search -> follow_citation -> answer (should loop 2-3 rounds)",
    },
    {
        "q": ("Summarize the papers from 2023 in this collection and how their contributions "
              "differ from the earlier work on the same topic."),
        "exercises": "find_by_author(year=2023) + semantic_search -> answer (multi-tool loop)",
    },
    {
        "q": "Tell me about the evaluation.",
        "exercises": "clarify (too vague to search — evaluation of what?)",
    },
]


# %% [markdown]
# ## Step 4 — Run the agent on each question, and compare against the fixed baseline

# %%
def run() -> None:
    print(f"Checkpoint 5.1 — agent-based RAG over the full corpus  |  scenario: {SCENARIO}")
    print(f"Corpus: {PDF_DIR}\n")

    if not Path(PDF_DIR).exists():
        print(f"[!] Corpus folder not found: {PDF_DIR}")
        print("    Fix PDF_DIR at the top of this file, then rerun.")
        print("\nYour agent plan (design still prints without the corpus):")
        print(json.dumps(my_agent_plan(), indent=2))
        return

    texts, metas = build_corpus(PDF_DIR)
    retriever = HybridRetriever(texts, metas)
    llm = make_llm()
    agent = ToolUsingAgent(retriever, llm)

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["q"]
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        print(f"(exercises: {item['exercises']})\n")

        print("--- AGENT (multi-step) ---")
        agent_ans = agent.run(q)
        print(f"\nAgent answer:\n{agent_ans}\n")

        print("--- FIXED PIPELINE (single search, Checkpoint 2.1 baseline) ---")
        fixed_ans = fixed_pipeline_answer(retriever, llm, q)
        print(f"Fixed answer:\n{fixed_ans}\n")

        log(f"Q{i} AGENTIC", f"Q: {q}\nA: {agent_ans}")
        log(f"Q{i} FIXED", f"Q: {q}\nA: {fixed_ans}")

    print("\n" + "=" * 72)
    print("Your agent plan:")
    print(json.dumps(my_agent_plan(), indent=2))
    print("=" * 72)
    print(f"Done. Step-by-step decisions are logged to {LOG_PATH.name} for your write-up.")


if __name__ == "__main__":
    run()
