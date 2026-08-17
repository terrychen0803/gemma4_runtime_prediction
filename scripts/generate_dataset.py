from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from common import DEFAULT_DATASET_ROOT, DEFAULT_MODEL_CONFIG, load_json


SEED = 20260817
DEFAULT_SEQUENCE_LENGTHS = [256, 512, 1024]
DEFAULT_SAMPLES = 128
IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def build_source_text(tokenizer) -> str:
    conversations = [
        (
            "Explain why fixed tensor shapes improve repeatable GPU profiling.",
            "Fixed shapes keep padding, kernel selection, and arithmetic work constant across runs.",
        ),
        (
            "Summarize gradient accumulation in one paragraph.",
            "Gradient accumulation combines gradients from several micro-batches before one optimizer update.",
        ),
        (
            "What should be recorded for a cross-hardware runtime experiment?",
            "Record workload parameters, exact software versions, hardware identity, timing, and memory usage.",
        ),
        (
            "Describe LoRA fine-tuning.",
            "LoRA freezes base weights and trains small low-rank adapters in selected linear layers.",
        ),
    ]

    rendered: list[str] = []

    for prompt, response in conversations:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    return "\n".join(rendered)


def make_dataset(tokenizer, sequence_length: int, samples: int):
    import torch

    source_text = build_source_text(tokenizer)
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)

    if not source_ids:
        raise RuntimeError("Tokenizer produced no token IDs.")

    required = sequence_length + samples * 17
    repeated = source_ids * (required // len(source_ids) + 2)
    rows: list[list[int]] = []

    for index in range(samples):
        offset = (index * 17) % len(source_ids)
        rows.append(repeated[offset : offset + sequence_length])

    input_ids = torch.tensor(rows, dtype=torch.int32)
    attention_mask = torch.ones((samples, sequence_length), dtype=torch.bool)
    labels = input_ids.clone()
    labels[:, : sequence_length // 2] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic fixed-length Gemma 4 SFT tensors."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--sequence-lengths",
        default=",".join(str(value) for value in DEFAULT_SEQUENCE_LENGTHS),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the tokenizer to be downloaded instead of requiring the local cache.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.samples < 8:
        raise ValueError("--samples must be at least 8")

    sequence_lengths = [
        int(value.strip())
        for value in args.sequence_lengths.split(",")
        if value.strip()
    ]

    if not sequence_lengths or any(value < 32 for value in sequence_lengths):
        raise ValueError("Sequence lengths must all be at least 32.")

    output = args.output.resolve()

    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise SystemExit(
                f"Dataset directory is not empty: {output}\nUse --force to regenerate it."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    model_config = load_json(args.model_config.resolve())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config.get("model_revision"),
        local_files_only=not args.allow_download,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    files: dict[str, dict[str, object]] = {}

    for sequence_length in sequence_lengths:
        dataset = make_dataset(tokenizer, sequence_length, args.samples)
        path = output / f"seq_{sequence_length:04d}.pt"

        import torch

        torch.save(dataset, path)
        files[str(sequence_length)] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "samples": args.samples,
            "sequence_length": sequence_length,
            "supervised_tokens_per_sample": sequence_length - sequence_length // 2,
        }

    manifest = {
        "dataset": "gemma4_runtime_prediction_synthetic_sft_v1",
        "seed": SEED,
        "model_id": model_config["model_id"],
        "requested_model_revision": model_config.get("model_revision"),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "tokenizer_class": tokenizer.__class__.__name__,
        "ignore_index": IGNORE_INDEX,
        "files": files,
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

