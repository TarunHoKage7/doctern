"""Two separate local model adapters over Ollama's native /api/chat on loopback.

Structured output uses Ollama's `format` JSON-schema constraint, then Pydantic validation, with one
schema-repair attempt. Nothing here calls a cloud API.
"""
from __future__ import annotations

import json
import time
from typing import Type

import httpx
from pydantic import BaseModel, ValidationError

from .config import (CALL_TIMEOUT_S, GEMINI_API_KEY, GEMMA_BACKEND, GEMMA_MODEL, GENERATION, LIMITS,
                     MEDGEMMA_BACKEND, MEDGEMMA_MODEL, OLLAMA_URL, VERTEX_ACCESS_TOKEN,
                     VERTEX_PREDICT_URL)

SCHEMA_HINT = chr(10) * 2 + "Return ONLY one JSON object matching this JSON schema:" + chr(10)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ModelError(RuntimeError):
    pass


def first_json(raw: str):
    """Decode the first complete JSON object; ignore trailing text some models append."""
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object", raw, 0)
    return json.JSONDecoder().raw_decode(raw[start:])[0]


def model_identity(model: str) -> dict:
    if model == MEDGEMMA_MODEL and MEDGEMMA_BACKEND == "vertex":
        return {"model": model, "runtime": "vertex_ai endpoint (online)", "digest": "vertex:" + VERTEX_PREDICT_URL.split("/endpoints/")[-1],
                "quantization": "provider-managed"}
    if model == GEMMA_MODEL and GEMMA_BACKEND == "gemini_api":
        return {"model": model, "runtime": "gemini_api (hosted, online)", "digest": "hosted:" + model,
                "quantization": "provider-managed"}
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=10)
        r.raise_for_status()
        d = r.json().get("details", {})
        return {"model": model, "runtime": "ollama", "family": d.get("family"),
                "parameter_size": d.get("parameter_size"), "quantization": d.get("quantization_level"),
                "digest": _digest(model)}
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "runtime": "ollama", "available": False, "error": str(exc)[:200]}


def _digest(model: str) -> str | None:
    try:
        for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", []):
            if m["name"] == model or m["model"] == model:
                return m.get("digest")
    except Exception:  # noqa: BLE001
        return None
    return None


class LocalModelClient:
    role = "model"

    def __init__(self, model: str):
        self.model = model

    def _chat(self, system: str, user: str, schema: dict) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=CALL_TIMEOUT_S, json={
                "model": self.model, "stream": False, "format": schema, "think": False,
                "options": GENERATION, "keep_alive": "30m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelError(f"{self.model} timed out after {CALL_TIMEOUT_S}s") from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"{self.model} call failed: {exc}") from exc
        return r.json()["message"]["content"], time.perf_counter() - t0

    def structured(self, system: str, user: str, out: Type[BaseModel]) -> tuple[BaseModel, dict]:
        schema = out.model_json_schema()
        calls = []
        prompt = user
        for attempt in range(1 + LIMITS["schema_repairs"]):
            raw, secs = self._chat(system, prompt, schema)
            calls.append({"model": self.model, "attempt": attempt, "seconds": round(secs, 2), "raw": raw})
            try:
                return out.model_validate(first_json(raw)), {"calls": calls}
            except (json.JSONDecodeError, ValidationError) as e:
                exc = e
                prompt = (user + "\n\nYour previous output was invalid for the required JSON schema: "
                          + str(exc)[:600] + "\nReturn only corrected JSON.")
        err = ModelError(f"{self.model} output invalid after repair: {str(exc)[:300]}")
        err.calls = calls
        raise err from None


class HostedGemmaClient(LocalModelClient):
    """Gemma 4 served by the Gemini API (online). Chosen by the team for orchestration; MedGemma stays local."""

    def _chat(self, system: str, user: str, schema: dict) -> tuple[str, float]:
        t0 = time.perf_counter()
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": GENERATION["temperature"], "topP": GENERATION["top_p"],
                                     "seed": GENERATION["seed"], "maxOutputTokens": 8192,
                                     "responseMimeType": "application/json", "responseJsonSchema": schema}}
        try:
            r = httpx.post(GEMINI_URL.format(model=self.model), headers={"x-goog-api-key": GEMINI_API_KEY},
                           json=body, timeout=CALL_TIMEOUT_S)
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
        except httpx.TimeoutException as exc:
            raise ModelError(f"{self.model} timed out after {CALL_TIMEOUT_S}s") from exc
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            raise ModelError(f"{self.model} hosted call failed: {type(exc).__name__}") from exc
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return text, time.perf_counter() - t0


def GemmaClient():  # noqa: N802 - factory keeps the adapter name used across the app
    cls = HostedGemmaClient if GEMMA_BACKEND == "gemini_api" else LocalModelClient
    c = cls(GEMMA_MODEL)
    c.role = "gemma_orchestrator"
    return c


class VertexMedGemmaClient(LocalModelClient):
    """MedGemma deployed from Vertex AI Model Garden (vLLM serving, online).

    Uses the :predict chatCompletions request format. Vertex does not enforce the JSON schema, so the
    schema is included in the prompt and the app's parse + validation + one repair still apply.
    """

    def _chat(self, system: str, user: str, schema: dict) -> tuple[str, float]:
        t0 = time.perf_counter()
        body = {"instances": [{"@requestFormat": "chatCompletions", "max_tokens": GENERATION["num_predict"],
                               "temperature": GENERATION["temperature"], "messages": [
                                   {"role": "system", "content": system},
                                   {"role": "user", "content": user + SCHEMA_HINT + json.dumps(schema)}]}]}
        try:
            r = httpx.post(VERTEX_PREDICT_URL, json=body, timeout=CALL_TIMEOUT_S,
                           headers={"Authorization": f"Bearer {VERTEX_ACCESS_TOKEN}"})
            r.raise_for_status()
            pred = r.json()["predictions"]
            pred = pred[0] if isinstance(pred, list) and pred and isinstance(pred[0], dict) else pred
            text = pred["choices"][0]["message"]["content"] if isinstance(pred, dict) else str(pred)
        except httpx.TimeoutException as exc:
            raise ModelError(f"{self.model} timed out after {CALL_TIMEOUT_S}s") from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"{self.model} vertex call failed: {type(exc).__name__}") from exc
        return text, time.perf_counter() - t0


def MedGemmaClient():  # noqa: N802
    if MEDGEMMA_BACKEND == "vertex":
        c = VertexMedGemmaClient(MEDGEMMA_MODEL)
    else:
        c = LocalModelClient(MEDGEMMA_MODEL)
    c.role = "medgemma_reviewer"
    return c
