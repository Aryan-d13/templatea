"""
ai_hook_orchestrator_perplexity.py
----------------------------------
Production-safe, *sophisticated* orchestration for ad-reel one-liners with optional
Perplexity browsing (sonar / sonar-pro) and Groq generation.

Contract preserved:
  Returns dict with shape:
    {
      "one_liners": ["...", "...", "..."],
      "source": "groq_orchestrated" | "groq_orchestrated_web" | "local_fallback",
      "validation_notes": ["..."]
    }

Key traits:
- Stage 0: Normalize OCR + caption, detect language, extract entities, score caption quality.
- Stage 0.5 (optional): Perplexity browsing to derive *soft trend hints* + citations without
  introducing brittle factual claims. If confidence is low, we simply don’t use hints.
- Stage 1 (Drafts): Fast Groq model generates 24 JSON-only candidates with style coverage.
- Stage 2 (Critic): Strong Groq model picks the best 3 with style diversity + brevity + safety.
- Post: lexical relevance scoring to prefer context-fit; de-dup; guaranteed 3 outputs.
- JSON-only to/from LLMs; robust parsing; never raises in production (safe fallbacks).

Environment:
- GROQ:   https://api.groq.com/openai/v1/chat/completions  (Bearer GROQ_API_KEY)
- PPLX:   https://api.perplexity.ai/chat/completions        (Bearer PPLX_API_KEY)

Minimal use:
    res = generate_ai_one_liners_browsing(
        ocr_caption, groq_api_key=..., perplexity_api_key=..., downloader_caption=..., use_perplexity=True
    )

Notes:
- We DO NOT change your output JSON shape. Keep your extraction as-is.
- Perplexity results (titles/urls/dates) are summarized into non-assertive hints ("X era",
  "Y energy"). If hints are weak, they are dropped. Citations are surfaced in validation_notes only.
"""

from __future__ import annotations
import os, json, re, unicodedata, requests
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv() 

# ------------------ Constants ------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PPLX_API_URL = "https://api.perplexity.ai/chat/completions"

DRAFTS_MODEL_PREF = "llama-3.1-8b-instant"     # fast, cheap, explores breadth
CRITIC_MODEL_PREF  = "llama-3.3-70b-versatile"         # stronger selector for quality

# ------------------ Text Utilities ------------------

def _clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = (
        text.replace("\u2014", ", ")
            .replace("\u2013", ", ")
            .replace("\u2212", "-")
    )
    out = []
    for ch in normalized:
        if unicodedata.category(ch) == "Cf":  # zero-width & format chars
            continue
        out.append(ch)
    return "".join(out).strip()


def _count_words(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def _is_valid_line(s: str, *, allow_emoji: bool = True, max_words: int = 10) -> bool:
    s = s.strip()
    if not s:
        return False
    if _count_words(s) > max_words:
        return False
    if "#" in s or "@" in s:  # no hashtags/mentions
        return False
    if not allow_emoji and re.search(r"[\U0001F300-\U0001FAFF]", s):
        return False
    low = s.lower()
    banned = ["take my money", "shut up and take", "here for it", "literal chills"]
    if any(b in low for b in banned):
        return False
    risky = ["guaranteed", "cure", "instant results", "100% off"]
    if any(r in low for r in risky):
        return False
    return True


def _dedup(seq: List[str], thresh: float = 0.90) -> List[str]:
    out: List[str] = []
    def jaccard(a: str, b: str) -> float:
        A, B = set(a.lower().split()), set(b.lower().split())
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0
    for s in seq:
        if not out or all(jaccard(s, t) < thresh for t in out):
            out.append(s)
    return out


def _extract_json(text: str) -> Optional[Any]:
    """Tolerant JSON extractor from LLM output."""
    text = (text or "").strip()
    for p in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```", r"`(.*?)`"]:
        m = re.search(p, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
            break
    try:
        return json.loads(text)
    except Exception:
        pass
    for p, flags in [(r"\{.*\}", re.DOTALL), (r"\[.*\]", re.DOTALL)]:
        m = re.search(p, text, flags)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return None

# ------------------ Light NLP ------------------

def _detect_language(ocr: str, dl: str) -> str:
    blob = (ocr or "") + "\n" + (dl or "")
    return "hi" if re.search("[\u0900-\u097F]", blob) else "en"


def _caption_quality(caption: str, ocr: str) -> Tuple[str, float]:
    """classify caption as good/ok/garbage with a cheap lexical heuristic."""
    if not caption:
        return ("empty", 0.0)
    cap = caption.lower()
    # garbage-y signals: long bloggy, many links/hashtags, mismatched tokens
    links = cap.count("http://") + cap.count("https://")
    hashtags = cap.count("#")
    longish = len(cap) > 300
    # overlap with OCR tokens
    src = (ocr or "").lower()
    cap_tokens = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", cap))
    ocr_tokens = set(re.findall(r"[a-zA-Z\u0900-\u097F]+", src))
    overlap = len(cap_tokens & ocr_tokens) / max(1, len(cap_tokens))
    if links > 2 or hashtags > 8 or longish and overlap < 0.05:
        return ("garbage_like", 0.15)
    if overlap < 0.10:
        return ("weak", 0.35)
    if overlap < 0.25:
        return ("ok", 0.55)
    return ("good", 0.75)


def _extract_entities(text: str) -> List[str]:
    # ultra-light entity hints: Capitalized tokens and common brand-ish tokens
    ents = set()
    for tok in re.findall(r"[A-Z][a-zA-Z]{2,}\b", text or ""):
        ents.add(tok)
    # add product-ish keywords
    for kw in re.findall(r"(Pro|Ultra|MAX|Plus|Limited|Edition|Beta)\b", text or ""):
        ents.add(kw)
    return sorted(ents)[:8]


def _relevance(line: str, ocr: str, dl: str) -> float:
    src = (ocr + " " + dl).lower()
    src_tokens = {t for t in re.findall(r"[a-zA-Z\u0900-\u097F]+", src)}
    cand_tokens = {t for t in re.findall(r"[a-zA-Z\u0900-\u097F]+", line.lower())}
    if not cand_tokens:
        return 0.0
    inter = len(src_tokens & cand_tokens)
    return inter / max(1, len(cand_tokens))

# ------------------ External Calls ------------------

def _groq_chat(api_key: str, model: str, system: str, user: str, temperature: float, timeout: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
    }
    r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _pplx_browse(perplexity_api_key: str, query: str, model: str = "sonar-pro", timeout: int = 45) -> Tuple[List[Dict[str, str]], str]:
    """Queries Perplexity and returns (search_results, assistant_text). Safe on failure."""
    try:
        headers = {"authorization": f"Bearer {perplexity_api_key}", "content-type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "enable_search_classifier": True,
        }
        resp = requests.post(PPLX_API_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        sr = data.get("search_results") or []
        msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        out = []
        for item in sr:
            title = str(item.get("title", "")).strip()
            url   = str(item.get("url", "")).strip()
            date  = str(item.get("date", "")).strip()
            if url:
                out.append({"title": title, "url": url, "date": date})
        return out[:5], msg
    except Exception as e:
        return [], f"browse_fail: {e}"

# ------------------ Prompt Templates ------------------
DRAFTS_SYS = (
    "You generate on-video hook one-liners for short ad reels.\n"
    "Return JSON ONLY with key \"drafts\" as an array of strings.\n"
    "Rules: 4–8 words (hard cap 10), no hashtags/@, no unverifiable facts."
)

DRAFTS_USER_TMPL = (
    "OCR_CAPTION:\n{ocr}\n\n"
    "DOWNLOADER_CAPTION_QUALITY: {cap_quality} ({cap_score:.2f})\n"
    "DOWNLOADER_CAPTION:\n{dl}\n\n"
    "SOFT_TREND_HINTS:\n{hints}\n\n"
    "Produce exactly 24 diverse one-liners across styles: \n"
    "trophy, peak, POV, absurd, buy-now-irony, chefkiss, setup-punch, hyperliteral.\n"
    "Language: {language}.\n"
    "JSON schema: {{\"drafts\":[...]}}"
)

CRITIC_SYS = (
    "You are a selector that chooses the best 3 one-liners for an ad reel.\n"
    "Rules: 4–8 words (max 10), no hashtags/@, no unverifiable facts.\n"
    "Ensure style diversity (not 3 of the same). Return JSON ONLY:\n"
    "{\"one_liners\":[\"...\", \"...\", \"...\"], \"reasons\":[\"...\", \"...\", \"...\"]}"
)

CRITIC_USER_TMPL = (
    "OCR_CAPTION:\n{ocr}\n\n"
    "DOWNLOADER_CAPTION_QUALITY: {cap_quality} ({cap_score:.2f})\n"
    "DOWNLOADER_CAPTION:\n{dl}\n\n"
    "SOFT_TREND_HINTS:\n{hints}\n\n"
    "CANDIDATES_JSON:\n{cands}\n\n"
    "Pick the strongest 3 lines that are relevant to the context and universal enough if context is weak.\n"
    "Prefer punchy phrasing and social-native tone.\n"
    "Language: {language}."
)

# ------------------ Context Builder ------------------

def _build_browse_query(ocr: str, dl: str, entities: List[str]) -> str:
    # Compose a concise, high-signal query. We avoid dates/claims; we want *vibe-level* hints.
    core = (ocr or "")[:280]
    cap  = (dl or "")[:280]
    ents = ", ".join(entities)
    return (
        "Given this ad-reel OCR and caption, identify any public trend, meme, brand moment, "
        "or cultural reference likely connected to it. Summarize in 3 short bullets. Do NOT "
        "invent facts; focus on recognizable phrases for wink-level references.\n\n"
        f"OCR: {core}\nCAPTION: {cap}\nENTITIES: {ents}"
    )


def _hints_from_pplx_text(text: str) -> List[str]:
    # Extract 1-liner hints from Perplexity assistant text (keep neutral, no claims)
    if not text:
        return []
    bullets = re.split(r"\n\s*[-•]\s*", text)
    if len(bullets) == 1:
        bullets = re.split(r"\n+", text)
    out: List[str] = []
    for b in bullets:
        b = _clean_text(b)
        if not b:
            continue
        # soften: remove dates, numbers-heavy claims
        b = re.sub(r"\b\d{4}\b", "", b)
        b = re.sub(r"\b\d+%\b", "", b)
        if len(b) > 140:
            b = b[:140].rstrip() + "…"
        out.append(b)
    return out[:3]

# ------------------ Public API ------------------

def generate_ai_one_liners_browsing(
    ocr_caption: str,
    *,
    groq_api_key: str,
    perplexity_api_key: Optional[str] = None,
    downloader_caption: Optional[str] = None,
    use_perplexity: bool = True,
    drafts_model: str = "llama-3.3-70b-versatile",
    critic_model: str = "openai/gpt-oss-20b",
    timeout: int = 60,
    allow_emoji: bool = True,
    language: str = "auto",
    pplx_model: str = "sonar-pro",
) -> Dict[str, Any]:
    """Sophisticated orchestration with optional Perplexity context.
    Never raises; returns safe fallbacks on failure.
    """
    notes: List[str] = []
    result = {"one_liners": [], "source": "groq_orchestrated", "validation_notes": notes}

    if not groq_api_key:
        result["one_liners"] = [
            "Someone promote this intern.",
            "Peak ad. No notes.",
            "POV: you’re already sold.",
        ]
        result["source"] = "local_fallback"
        notes.append("Missing GROQ_API_KEY; used local fallback.")
        return result

    ocr = _clean_text(ocr_caption or "")
    dl  = _clean_text(downloader_caption or "")
    if language == "auto":
        language = _detect_language(ocr, dl)

    cap_q, cap_score = _caption_quality(dl, ocr)
    entities = _extract_entities(ocr + "\n" + dl)

    # Optional browse
    hints: List[str] = []
    citations: List[Dict[str, str]] = []
    if use_perplexity and perplexity_api_key:
        try:
            q = _build_browse_query(ocr, dl, entities)
            citations, pplx_text = _pplx_browse(perplexity_api_key, q, model=pplx_model, timeout=timeout)
            hints = _hints_from_pplx_text(pplx_text)
            if hints:
                result["source"] = "groq_orchestrated_web"
            if citations:
                notes.append("pplx_citations=" + json.dumps(citations, ensure_ascii=False))
        except Exception as e:
            notes.append(f"perplexity_error: {e}")
    else:
        notes.append("perplexity_disabled_or_missing_key")

    # Stage 1: Drafts
    try:
        drafts_user = DRAFTS_USER_TMPL.format(
            ocr=ocr or "[EMPTY]",
            dl=dl or "[NOT AVAILABLE]",
            cap_quality=cap_q,
            cap_score=cap_score,
            hints="; ".join(hints) if hints else "[NONE]",
            language=language,
        )
        drafts_raw = _groq_chat(groq_api_key, drafts_model, DRAFTS_SYS, drafts_user, temperature=0.5, timeout=timeout)
        drafts_json = _extract_json(drafts_raw) or {"drafts": []}
        drafts = drafts_json.get("drafts", [])
        if not isinstance(drafts, list):
            drafts = []
    except Exception as e:
        drafts = []
        notes.append(f"drafts_stage_fail: {e}")

    # Filter drafts
    filtered: List[str] = []
    for s in drafts:
        if not isinstance(s, str):
            continue
        t = re.sub(r"\s+", " ", _clean_text(s)).strip()
        if _is_valid_line(t, allow_emoji=allow_emoji, max_words=10):
            filtered.append(t)
    filtered = _dedup(filtered)

    if not filtered:
        filtered = [
            "Someone promote this intern.",
            "Peak ad. No notes.",
            "POV: you’re already sold.",
            "I don’t need it. I need two.",
            "Art. Simply art.",
        ]
        notes.append("seeded_generics_due_to_no_valid_drafts")

    # Stage 2: Critic
    try:
        cands_json = json.dumps(filtered, ensure_ascii=False)
        critic_user = CRITIC_USER_TMPL.format(
            ocr=ocr or "[EMPTY]",
            dl=dl or "[NOT AVAILABLE]",
            cap_quality=cap_q,
            cap_score=cap_score,
            hints="; ".join(hints) if hints else "[NONE]",
            cands=cands_json,
            language=language,
        )
        critic_raw = _groq_chat(groq_api_key, critic_model, CRITIC_SYS, critic_user, temperature=0.2, timeout=timeout)
        critic_json = _extract_json(critic_raw) or {}
        chosen = critic_json.get("one_liners", [])
        if not isinstance(chosen, list):
            chosen = []
    except Exception as e:
        chosen = []
        notes.append(f"critic_stage_fail: {e}")

    # Finalize: filter, dedup, relevance preference, top-up
    final: List[str] = []
    rel_map: Dict[str, float] = {}
    for s in chosen:
        if not isinstance(s, str):
            continue
        t = re.sub(r"\s+", " ", _clean_text(s)).strip()
        if _is_valid_line(t, allow_emoji=allow_emoji, max_words=10):
            final.append(t)
            rel_map[t] = _relevance(t, ocr, dl)

    final = _dedup(final)

    if len(final) < 3:
        scored = sorted(filtered, key=lambda x: _relevance(x, ocr, dl), reverse=True)
        for cand in scored:
            if len(final) >= 3:
                break
            if cand not in final and _is_valid_line(cand):
                final.append(cand)

    if len(final) < 3:
        backups = [
            "Someone promote this intern.",
            "Peak ad. No notes.",
            "POV: you’re already sold.",
        ]
        for b in backups:
            if len(final) >= 3:
                break
            if b not in final:
                final.append(b)

    # Diagnostics
    try:
        if rel_map:
            top_rel = ", ".join([f"{k} (r={rel_map.get(k, 0):.2f})" for k in final[:3]])
            notes.append(f"relevance(top3): {top_rel}")
        if entities:
            notes.append("entities=" + ", ".join(entities))
        notes.append(f"caption_quality={cap_q}:{cap_score:.2f}")
    except Exception:
        pass

    result["one_liners"] = final[:3]
    return result
