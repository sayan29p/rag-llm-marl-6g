import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI

from config import (
    USE_GROQ,
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
    LLM_MODEL_OPENAI, OPENAI_API_KEY,
    M,
)

# ---------------------------------------------------------------------------
# Default hint returned whenever the LLM is unavailable or returns bad JSON
# ---------------------------------------------------------------------------
_DEFAULT_HINT: dict = {
    "preferred_nodes": [],
    "avoid_nodes"    : [],
    "urgency"        : "medium",
    "confidence"     : 0.5,
    "reasoning"      : "LLM unavailable - using neutral policy",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an intelligent coordinator for a 6G edge-cloud task-offloading "
    "network. Your role is to advise a multi-agent reinforcement learning "
    "system on how to distribute compute tasks across {M} edge nodes and the "
    "cloud. You receive a summary of the current network state and a set of "
    "past similar situations with the rewards they achieved. "
    "Based on this information, provide concise, actionable guidance to "
    "improve task scheduling decisions. "
    "You MUST respond with a single valid JSON object and nothing else — "
    "no markdown, no commentary, no code fences. "
    "Required keys:\n"
    "  preferred_nodes : list of node numbers (1-{M}) that are good offload targets right now\n"
    "  avoid_nodes     : list of node numbers (1-{M}) to avoid (congested or poor channel)\n"
    "  urgency         : one of \"high\", \"medium\", or \"low\"\n"
    "  confidence      : float 0.0-1.0 reflecting how confident you are\n"
    "  reasoning       : one sentence explaining your recommendation\n"
    "Example: {{\"preferred_nodes\": [2, 4], \"avoid_nodes\": [1], "
    "\"urgency\": \"high\", \"confidence\": 0.8, "
    "\"reasoning\": \"Nodes 2 and 4 have low queue depth and strong channels.\"}}"
).format(M=M)

_USER_TEMPLATE = (
    "=== Current Network State ===\n"
    "{state_text}\n\n"
    "=== Retrieved Past Situations ===\n"
    "{context_string}\n\n"
    "Based on the current state and past experience, provide your JSON recommendation."
)


class LLMCoordinator:
    """
    Thin wrapper that routes LLM calls to either Groq (USE_GROQ=True) or
    the OpenAI API (USE_GROQ=False).  Both paths use the same OpenAI-compatible
    client interface; only the base_url, api_key, and model differ.
    """

    def __init__(self):
        if USE_GROQ:
            key = GROQ_API_KEY if GROQ_API_KEY else os.environ.get("GROQ_API_KEY", "")
            self._client = OpenAI(
                api_key=key,
                base_url=GROQ_BASE_URL,
            )
            self._model = GROQ_MODEL
        else:
            key = OPENAI_API_KEY if OPENAI_API_KEY else os.environ.get("OPENAI_API_KEY", "")
            self._client = OpenAI(api_key=key)
            self._model  = LLM_MODEL_OPENAI

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hint(self, state_text: str, context_string: str) -> dict:
        """
        Query the LLM for scheduling guidance.

        Parameters
        ----------
        state_text     : str — serialized current observation (from StateSerializer)
        context_string : str — formatted past experiences (from VectorStore)

        Returns
        -------
        dict with keys: preferred_nodes, avoid_nodes, urgency, confidence, reasoning
        Always returns a valid dict; falls back to _DEFAULT_HINT on any error.
        """
        user_msg = _USER_TEMPLATE.format(
            state_text=state_text,
            context_string=context_string,
        )

        try:
            raw = self._call_openai(user_msg)
            return self._parse(raw)
        except Exception:
            return dict(_DEFAULT_HINT)

    # ------------------------------------------------------------------
    # Backend call (shared by both Groq and OpenAI paths)
    # ------------------------------------------------------------------

    def _call_openai(self, user_msg: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> dict:
        """
        Extract and validate a JSON hint from the LLM response string.
        Falls back to _DEFAULT_HINT if JSON is missing or malformed.
        """
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            hint = json.loads(text)
        except json.JSONDecodeError:
            return dict(_DEFAULT_HINT)

        valid_nodes = set(range(1, M + 1))

        preferred = [
            int(n) for n in hint.get("preferred_nodes", [])
            if int(n) in valid_nodes
        ]
        avoid = [
            int(n) for n in hint.get("avoid_nodes", [])
            if int(n) in valid_nodes
        ]
        urgency = hint.get("urgency", "medium")
        if urgency not in {"high", "medium", "low"}:
            urgency = "medium"

        try:
            confidence = float(hint.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = str(hint.get("reasoning", _DEFAULT_HINT["reasoning"]))

        return {
            "preferred_nodes": preferred,
            "avoid_nodes"    : avoid,
            "urgency"        : urgency,
            "confidence"     : confidence,
            "reasoning"      : reasoning,
        }
