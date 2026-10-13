"""
In-app DeepEval runner for the Evaluation tab. Uses the same indexed
documents and retriever as the chatbot. Throttled + retried to survive
Gemini free-tier rate limits.
"""
import json
import os
import re
import time
from typing import Callable, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .retrieval import thread_chunks
from .tools.rag import rag_tool
from .utils import message_text

METRIC_NAMES = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Precision",
    "Contextual Recall",
]

llm = ChatGoogleGenerativeAI(
    model=os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash"),
    google_api_key=os.getenv("API_KEY"),
    temperature=0.3,
    max_retries=2,
)

CALL_GAP_S = 6  # ~10 requests/minute on the free tier


def _is_rate_limit(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return "429" in s or "ratelimit" in s or "resource_exhausted" in s or "quota" in s


def invoke_with_backoff(prompt: str, attempts: int = 4):
    delay = 15
    for i in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit(exc) or i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def generate_qa_pairs(thread_id: str, n: int = 4) -> List[dict]:
    """Draft question / ground-truth pairs from passages sampled evenly
    across the thread's documents (review them before trusting the scores)."""
    chunks = thread_chunks(thread_id)
    if not chunks:
        return []
    step = max(len(chunks) // n, 1)
    sampled = chunks[::step][:n]

    pairs = []
    for chunk in sampled:
        prompt = (
            "From the passage below, write ONE specific factual question that can "
            "be answered using only this passage, and a short correct answer.\n"
            'Reply with JSON only: {"question": "...", "answer": "..."}\n\n'
            f"Passage:\n{chunk.page_content[:1500]}"
        )
        raw = message_text(invoke_with_backoff(prompt).content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                pairs.append(
                    {"question": data["question"].strip(), "ground_truth": data["answer"].strip()}
                )
            except (ValueError, KeyError):
                pass
        time.sleep(CALL_GAP_S)
    return pairs


def run_deepeval(
    thread_id: str,
    qa_pairs: List[dict],
    threshold: float = 0.5,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    """Retrieve + answer each question, then score it with DeepEval's RAG
    metrics. Precision/recall are skipped for rows with no ground truth."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase
    from evaluate_rag import GeminiDeepEvalModel

    judge = GeminiDeepEvalModel(llm)
    rows = []
    total_steps = max(len(qa_pairs), 1) * len(METRIC_NAMES)
    done = 0

    for qa in qa_pairs:
        question = qa["question"]
        truth = (qa.get("ground_truth") or "").strip()
        retrieval = rag_tool.invoke({"query": question, "thread_id": thread_id})
        context = retrieval.get("context", [])
        answer = message_text(
            invoke_with_backoff(
                "Answer the question using only the context below. If the answer "
                "is not in the context, say you don't know.\n\n"
                "Context:\n" + "\n\n".join(context) + f"\n\nQuestion: {question}"
            ).content
        )
        case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context,
            expected_output=truth or None,
        )
        metrics = [
            ("Faithfulness", FaithfulnessMetric, False),
            ("Answer Relevancy", AnswerRelevancyMetric, False),
            ("Contextual Precision", ContextualPrecisionMetric, True),
            ("Contextual Recall", ContextualRecallMetric, True),
        ]
        row = {"question": question, "ground_truth": truth, "answer": answer,
               "scores": {}, "reasons": {}}
        for name, cls, needs_truth in metrics:
            done += 1
            if progress:
                progress(done / total_steps, f"{name}: {question[:50]}")
            if needs_truth and not truth:
                row["scores"][name], row["reasons"][name] = None, "No ground truth given"
                continue
            for attempt in range(3):
                try:
                    metric = cls(model=judge, threshold=threshold, include_reason=True)
                    metric.measure(case)
                    row["scores"][name] = float(metric.score)
                    row["reasons"][name] = metric.reason or ""
                    break
                except Exception as exc:  # noqa: BLE001
                    if _is_rate_limit(exc) and attempt < 2:
                        time.sleep(20 * (attempt + 1))
                        continue
                    row["scores"][name], row["reasons"][name] = None, f"Error: {str(exc)[:200]}"
                    break
            time.sleep(CALL_GAP_S)
        rows.append(row)
    return rows