"""
In-app DeepEval runner for the Evaluation tab. Uses the *same* indexed
documents, retriever and LLM the chatbot uses, so the scores describe the
real pipeline. (evaluate_rag.py is the equivalent command-line version.)
"""
import json
import re
from typing import Callable, List, Optional

from .llm import llm
from .retrieval import thread_chunks
from .tools.rag import rag_tool
from .utils import message_text

METRIC_NAMES = [
    "Faithfulness",
    "Answer Relevancy",
    "Contextual Precision",
    "Contextual Recall",
]


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
        raw = message_text(llm.invoke(prompt).content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
            pairs.append(
                {"question": data["question"].strip(), "ground_truth": data["answer"].strip()}
            )
        except (ValueError, KeyError):
            continue
    return pairs


def run_deepeval(
    thread_id: str,
    qa_pairs: List[dict],
    threshold: float = 0.5,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[dict]:
    """Retrieve + answer each question, then score it with DeepEval's RAG
    metrics. Precision/recall need a ground truth, so they're skipped for
    rows where it's blank."""
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
            llm.invoke(
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
        row = {"question": question, "ground_truth": truth, "answer": answer, "scores": {}, "reasons": {}}
        for name, cls, needs_truth in metrics:
            done += 1
            if progress:
                progress(done / total_steps, f"{name}: {question[:50]}")
            if needs_truth and not truth:
                row["scores"][name], row["reasons"][name] = None, "No ground truth given"
                continue
            try:
                metric = cls(model=judge, threshold=threshold, include_reason=True)
                metric.measure(case)
                row["scores"][name] = float(metric.score)
                row["reasons"][name] = metric.reason or ""
            except Exception as exc:  # noqa: BLE001 - keep going on a failed metric
                row["scores"][name], row["reasons"][name] = None, f"Error: {str(exc)[:200]}"
        rows.append(row)
    return rows
