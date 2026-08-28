"""
Standalone DeepEval evaluation for the hybrid BM25 + FAISS + cross-encoder-rerank
RAG pipeline defined in chatbot_backend.py.

This indexes a PDF through the same `ingest_pdf` / `rag_tool` path the chatbot
uses at runtime, generates answers with the same LLM, and scores the results
with DeepEval's RAG metrics (faithfulness, answer relevancy, contextual
precision, contextual recall). It is a developer/CI utility, not something the
app runs per-request.

The LLM-as-judge metrics are pointed at the app's own Gemini model (wrapped in
GeminiDeepEvalModel below) so this doesn't require a separate OpenAI key.

Usage:
    python evaluate_rag.py path/to/sample.pdf
    python evaluate_rag.py path/to/sample.pdf --thread-id my-eval-run
"""
import argparse
import sys

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

from chatbot_backend import ingest_pdf, rag_tool, llm

# Replace these with real question / ground-truth pairs for your document
# before drawing any conclusions from the scores.
DEFAULT_EVAL_SET = [
    {
        "question": "What is this document primarily about?",
        "ground_truth": "Replace with the actual expected answer for your document.",
    },
    {
        "question": "Summarize the key points of the document.",
        "ground_truth": "Replace with the actual expected answer for your document.",
    },
]


class GeminiDeepEvalModel(DeepEvalBaseLLM):
    """
    Adapts the chatbot's own ChatGoogleGenerativeAI instance to DeepEval's
    DeepEvalBaseLLM interface, so LLM-judged metrics (faithfulness, answer
    relevancy, etc.) use the same Gemini model the app already runs on
    instead of requiring a separate OpenAI API key.
    """

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "gemini-2.5-flash"


_judge_model = GeminiDeepEvalModel(llm)


def build_test_cases(pdf_path: str, thread_id: str, qa_pairs) -> list[LLMTestCase]:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    ingest_pdf(pdf_bytes, thread_id=thread_id, filename=pdf_path)

    test_cases = []
    for qa in qa_pairs:
        question = qa["question"]
        retrieval = rag_tool.invoke({"query": question, "thread_id": thread_id})
        context_chunks = retrieval.get("context", [])

        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not contained in the context, say you don't know.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )
        answer = llm.invoke(prompt).content

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=context_chunks,
                expected_output=qa["ground_truth"],
            )
        )

    return test_cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with DeepEval.")
    parser.add_argument("pdf_path", help="Path to a PDF to index and evaluate against.")
    parser.add_argument(
        "--thread-id",
        default="deepeval-eval",
        help="Thread id to use for ingestion/retrieval (default: deepeval-eval).",
    )
    args = parser.parse_args()

    test_cases = build_test_cases(args.pdf_path, args.thread_id, DEFAULT_EVAL_SET)

    metrics = [
        FaithfulnessMetric(model=_judge_model, threshold=0.5),
        AnswerRelevancyMetric(model=_judge_model, threshold=0.5),
        ContextualPrecisionMetric(model=_judge_model, threshold=0.5),
        ContextualRecallMetric(model=_judge_model, threshold=0.5),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)

    print("\nDeepEval evaluation results:")
    print(result)


if __name__ == "__main__":
    sys.exit(main())
