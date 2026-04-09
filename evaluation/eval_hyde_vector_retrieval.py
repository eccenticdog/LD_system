import argparse
import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from dotenv import find_dotenv, load_dotenv

from app.core.logger import logger
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    step_1_create_hyde_doc,
    step_2_search_embedding_hyde,
)

load_dotenv(find_dotenv())


@dataclass
class RetrievalEvalSample:
    sample_id: str
    query: str
    item_names: list[str]
    reference_context_ids: list[str]
    reference_contexts: list[str]
    metadata: dict[str, Any]


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        result = []
        for item in value:
            if item is None:
                continue
            result.append(str(item))
        return result
    raise ValueError(f"Expected a list-like value, got: {type(value)}")


def _load_raw_records(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        records = []
        with dataset_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSONL line {line_no} in {dataset_path}: {exc}"
                    ) from exc
        return records

    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Dataset file must be a JSON array or JSONL file: {dataset_path}")
    return payload


def load_eval_samples(dataset_path: Path) -> list[RetrievalEvalSample]:
    samples: list[RetrievalEvalSample] = []
    for idx, record in enumerate(_load_raw_records(dataset_path), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record #{idx} must be an object")

        query = (record.get("query") or record.get("user_input") or "").strip()
        if not query:
            raise ValueError(f"Dataset record #{idx} is missing `query` or `user_input`")

        item_names = _normalize_str_list(record.get("item_names"))
        reference_context_ids = _normalize_str_list(
            record.get("reference_context_ids")
            or record.get("reference_chunk_ids")
            or record.get("gold_chunk_ids")
        )
        reference_contexts = _normalize_str_list(record.get("reference_contexts"))

        if not reference_context_ids and not reference_contexts:
            raise ValueError(
                f"Dataset record #{idx} must provide `reference_context_ids` "
                f"or `reference_contexts` for retrieval evaluation"
            )

        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Dataset record #{idx} field `metadata` must be an object")

        samples.append(
            RetrievalEvalSample(
                sample_id=str(record.get("sample_id") or f"sample-{idx:03d}"),
                query=query,
                item_names=item_names,
                reference_context_ids=reference_context_ids,
                reference_contexts=reference_contexts,
                metadata=metadata,
            )
        )
    return samples


def run_hyde_vector_retrieval(
    sample: RetrievalEvalSample,
    *,
    req_limit: int,
    top_k: int,
) -> dict[str, Any]:
    started_at = perf_counter()
    hyde_doc = step_1_create_hyde_doc(sample.query)
    raw_results = step_2_search_embedding_hyde(
        rewritten_query=sample.query,
        hyde_doc=hyde_doc,
        item_names=sample.item_names or None,
        req_limit=req_limit,
        top_k=top_k,
    )
    latency_ms = round((perf_counter() - started_at) * 1000, 2)

    hits = raw_results[0] if raw_results else []
    retrieved_context_ids: list[str] = []
    retrieved_contexts: list[str] = []
    retrieved_hits: list[dict[str, Any]] = []

    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity") or {}
        chunk_id = entity.get("chunk_id") or hit.get("id") or f"rank-{rank}"
        content = str(entity.get("content") or "")
        item_name = entity.get("item_name")
        score = hit.get("distance")

        chunk_id_str = str(chunk_id)
        retrieved_context_ids.append(chunk_id_str)
        retrieved_contexts.append(content)
        retrieved_hits.append(
            {
                "rank": rank,
                "chunk_id": chunk_id_str,
                "item_name": item_name,
                "score": score,
                "content_preview": content[:200],
            }
        )

    return {
        "hyde_doc_preview": hyde_doc[:500],
        "hyde_doc_char_count": len(hyde_doc),
        "latency_ms": latency_ms,
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_hits": retrieved_hits,
    }


def compute_manual_metrics(
    *,
    retrieved_context_ids: list[str],
    reference_context_ids: list[str],
) -> dict[str, float]:
    if not reference_context_ids:
        return {}

    retrieved = [str(item) for item in retrieved_context_ids]
    reference = [str(item) for item in reference_context_ids]
    reference_set = set(reference)
    matched_flags = [1 if item in reference_set else 0 for item in retrieved]
    matched_unique = len(set(retrieved) & reference_set)

    mrr = 0.0
    first_hit_rank = None
    for rank, matched in enumerate(matched_flags, start=1):
        if matched:
            first_hit_rank = rank
            mrr = 1.0 / rank
            break

    return {
        "hit_rate": 1.0 if any(matched_flags) else 0.0,
        "recall_at_k": matched_unique / len(reference_set) if reference_set else 0.0,
        "precision_at_k": matched_unique / len(retrieved) if retrieved else 0.0,
        "mrr": mrr,
        "first_hit_rank": float(first_hit_rank) if first_hit_rank is not None else 0.0,
    }


def _safe_mean(values: Iterable[Any]) -> float | None:
    numeric_values = [float(value) for value in values if isinstance(value, numbers.Number)]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _load_ragas_symbols() -> dict[str, Any]:
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError as exc:
        raise SystemExit(
            "ragas is not installed. Run `uv sync` or install `ragas`, `datasets`, "
            "and `rapidfuzz` into the project environment first."
        ) from exc

    try:
        from ragas import SingleTurnSample
    except ImportError:
        from ragas.dataset_schema import SingleTurnSample

    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    non_llm_precision = None
    non_llm_recall = None
    try:
        from ragas.metrics import NonLLMContextPrecisionWithReference

        non_llm_precision = NonLLMContextPrecisionWithReference
    except ImportError:
        logger.warning("Current ragas version does not expose NonLLMContextPrecisionWithReference")

    try:
        from ragas.metrics import NonLLMContextRecall

        non_llm_recall = NonLLMContextRecall
    except ImportError:
        logger.warning("Current ragas version does not expose NonLLMContextRecall")

    return {
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "evaluate": evaluate,
        "IDBasedContextPrecision": IDBasedContextPrecision,
        "IDBasedContextRecall": IDBasedContextRecall,
        "NonLLMContextPrecisionWithReference": non_llm_precision,
        "NonLLMContextRecall": non_llm_recall,
    }


def build_ragas_dataset(
    ragas_symbols: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[Any, list[Any], list[str]]:
    SingleTurnSample = ragas_symbols["SingleTurnSample"]
    EvaluationDataset = ragas_symbols["EvaluationDataset"]
    IDBasedContextPrecision = ragas_symbols["IDBasedContextPrecision"]
    IDBasedContextRecall = ragas_symbols["IDBasedContextRecall"]
    NonLLMContextPrecisionWithReference = ragas_symbols["NonLLMContextPrecisionWithReference"]
    NonLLMContextRecall = ragas_symbols["NonLLMContextRecall"]

    ragas_samples = []
    has_reference_ids_for_all = all(record["reference_context_ids"] for record in records)
    has_reference_contexts_for_all = all(record["reference_contexts"] for record in records)

    for record in records:
        sample_kwargs = {
            "user_input": record["query"],
            "retrieved_contexts": record["retrieved_contexts"],
            "retrieved_context_ids": record["retrieved_context_ids"],
        }
        if record["reference_context_ids"]:
            sample_kwargs["reference_context_ids"] = record["reference_context_ids"]
        if record["reference_contexts"]:
            sample_kwargs["reference_contexts"] = record["reference_contexts"]
        ragas_samples.append(SingleTurnSample(**sample_kwargs))

    metrics = []
    metric_names = []

    if has_reference_ids_for_all:
        metrics.extend([IDBasedContextPrecision(), IDBasedContextRecall()])
        metric_names.extend(["id_based_context_precision", "id_based_context_recall"])
    else:
        logger.warning("Skipped ID-based ragas metrics because some samples are missing reference IDs")

    if has_reference_contexts_for_all and NonLLMContextPrecisionWithReference and NonLLMContextRecall:
        metrics.extend([NonLLMContextPrecisionWithReference(), NonLLMContextRecall()])
        metric_names.extend(
            [
                "non_llm_context_precision_with_reference",
                "non_llm_context_recall",
            ]
        )
    elif has_reference_contexts_for_all:
        logger.warning("Skipped non-LLM ragas context metrics because current ragas version lacks them")

    if not metrics:
        raise ValueError("No ragas metrics can run. Check your dataset fields and ragas installation.")

    return EvaluationDataset(samples=ragas_samples), metrics, metric_names


def _extract_ragas_rows(result: Any) -> list[dict[str, Any]]:
    scores = getattr(result, "scores", None)
    if scores is None:
        return []
    if hasattr(scores, "to_list"):
        return scores.to_list()
    if isinstance(scores, list):
        return scores
    try:
        return list(scores)
    except TypeError:
        return []


def evaluate_hyde_vector(
    *,
    dataset_path: Path,
    output_path: Path,
    top_k: int,
    req_limit: int,
    max_samples: int | None,
) -> dict[str, Any]:
    samples = load_eval_samples(dataset_path)
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise ValueError("No evaluation samples found")

    evaluated_records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        logger.info(
            f"[Eval] Running HyDE+Vector retrieval for sample {index}/{len(samples)}: {sample.sample_id}"
        )
        retrieval_result = run_hyde_vector_retrieval(
            sample,
            req_limit=req_limit,
            top_k=top_k,
        )
        manual_metrics = compute_manual_metrics(
            retrieved_context_ids=retrieval_result["retrieved_context_ids"],
            reference_context_ids=sample.reference_context_ids,
        )
        evaluated_records.append(
            {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "item_names": sample.item_names,
                "reference_context_ids": sample.reference_context_ids,
                "reference_contexts": sample.reference_contexts,
                "metadata": sample.metadata,
                **retrieval_result,
                "manual_metrics": manual_metrics,
            }
        )

    ragas_symbols = _load_ragas_symbols()
    ragas_dataset, ragas_metrics, ragas_metric_names = build_ragas_dataset(
        ragas_symbols,
        [
            {
                "query": record["query"],
                "retrieved_contexts": record["retrieved_contexts"],
                "retrieved_context_ids": record["retrieved_context_ids"],
                "reference_context_ids": record["reference_context_ids"],
                "reference_contexts": record["reference_contexts"],
            }
            for record in evaluated_records
        ],
    )

    ragas_result = ragas_symbols["evaluate"](dataset=ragas_dataset, metrics=ragas_metrics)
    ragas_rows = _extract_ragas_rows(ragas_result)

    for record, ragas_row in zip(evaluated_records, ragas_rows):
        record["ragas_metrics"] = {
            name: ragas_row.get(name)
            for name in ragas_metric_names
            if name in ragas_row
        }

    manual_metric_keys = sorted(
        {
            metric_name
            for record in evaluated_records
            for metric_name in record["manual_metrics"].keys()
        }
    )
    manual_summary = {
        metric_name: _safe_mean(
            record["manual_metrics"].get(metric_name) for record in evaluated_records
        )
        for metric_name in manual_metric_keys
    }
    ragas_summary = {
        metric_name: _safe_mean(
            (record.get("ragas_metrics") or {}).get(metric_name) for record in evaluated_records
        )
        for metric_name in ragas_metric_names
    }

    result_payload = {
        "pipeline": "hyde_vector",
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "sample_count": len(evaluated_records),
        "top_k": top_k,
        "req_limit": req_limit,
        "summary": {
            "ragas": ragas_summary,
            "manual": manual_summary,
        },
        "samples": evaluated_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result_payload, file, ensure_ascii=False, indent=2)

    return result_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use ragas to evaluate HyDE+Vector retrieval on the local project."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the evaluation dataset (.json or .jsonl).",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/hyde_vector_eval_result.json",
        help="Path to the output JSON report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final top-k size returned by HyDE+Vector retrieval.",
    )
    parser.add_argument(
        "--req-limit",
        type=int,
        default=10,
        help="Milvus candidate pool size before weighted rerank.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for the number of evaluation samples.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result = evaluate_hyde_vector(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        top_k=args.top_k,
        req_limit=args.req_limit,
        max_samples=args.max_samples,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved detailed report to: {args.output}")


if __name__ == "__main__":
    main()
