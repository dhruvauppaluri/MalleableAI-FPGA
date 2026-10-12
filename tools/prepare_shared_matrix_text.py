"""Fetch a pinned, bounded WikiText-2 train/development pilot corpus.

This is an explicit opt-in download. The public WikiText test split is never
loaded or scored by this tool. These files are development data, not release
held-out evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


REPOSITORY = "Salesforce/wikitext"
CONFIGURATION = "wikitext-2-raw-v1"
LIMITS = {"train": 4096, "validation": 1024}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    revision = HfApi(token=False).dataset_info(
        REPOSITORY, revision="main").sha
    files = {}
    for split, limit in LIMITS.items():
        dataset = load_dataset(REPOSITORY, CONFIGURATION,
                               split=split, revision=revision)
        destination = root / ("dev.jsonl" if split == "validation" else "train.jsonl")
        count = 0
        with destination.open("x", encoding="utf-8") as stream:
            for row in dataset:
                value = row["text"].strip()
                if len(value) < 64:
                    continue
                stream.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")
                count += 1
                if count == limit:
                    break
            stream.flush()
            os.fsync(stream.fileno())
        if count < 100:
            raise RuntimeError(f"{split} supplied only {count} usable records")
        files[destination.name] = {"sha256": digest(destination), "records": count}
    manifest = {
        "schema_version": 1,
        "kind": "shared-matrix-real-text-development-corpus",
        "repository": REPOSITORY,
        "configuration": CONFIGURATION,
        "revision": revision,
        "files": files,
        "splits_loaded": ["train", "validation"],
        "test_scored": False,
        "release_heldout": False,
        "limitations": [
            "WikiText-2 is public and may overlap pretrained-model data.",
            "This bounded subset is for a development pilot only.",
        ],
    }
    with (root / "corpus-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output_dir": str(root), "revision": revision,
                      "files": files}), flush=True)


if __name__ == "__main__":
    main()
