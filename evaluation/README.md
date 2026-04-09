# HyDE+Vector ragas local evaluation

This directory contains a local retrieval evaluation script for the project's
HyDE+Vector pipeline.

## What is evaluated

The script reuses the current online retrieval path in the project:

1. Generate a HyDE document with the configured LLM.
2. Convert `query + hyde_doc` into BGE-M3 dense and sparse vectors.
3. Run Milvus hybrid retrieval.
4. Evaluate the returned chunks with `ragas`.

The default ragas metrics are:

- `id_based_context_precision`
- `id_based_context_recall`

If every sample also provides `reference_contexts`, the script will additionally
try to run non-LLM context metrics supported by the installed ragas version.

The script also outputs a few practical retrieval metrics:

- `hit_rate`
- `recall_at_k`
- `precision_at_k`
- `mrr`

## Dataset format

Use `.jsonl` or `.json`. Each record should look like:

```json
{
  "sample_id": "hak180-power-001",
  "query": "HAK 180 的电源线支持什么电压和频率？",
  "item_names": ["HAK 180 烫金机"],
  "reference_context_ids": ["replace-with-real-chunk-id-1"],
  "reference_contexts": [
    "本设备通过 AC 220 V-240 V 50/60 Hz 电源供电。"
  ],
  "metadata": {
    "source": "manual",
    "note": "replace chunk ids with your gold labels"
  }
}
```

Notes:

- `reference_context_ids` is the most important field for HyDE+Vector recall evaluation.
- `reference_contexts` is optional but recommended because it enables extra ragas metrics.
- `item_names` is optional. If supplied, retrieval will keep the same item filter logic as the current project.

## Run

After syncing dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m evaluation.eval_hyde_vector_retrieval `
  --dataset evaluation/sample_hyde_retrieval_dataset.jsonl `
  --top-k 5 `
  --req-limit 10 `
  --output evaluation/results/hyde_vector_eval_result.json
```

## Output

The script writes a JSON report that includes:

- ragas aggregate metrics
- manual aggregate metrics
- per-sample retrieved chunk ids
- per-sample ragas scores
- HyDE document preview
- retrieval latency
