"""Check the frozen controller suites against prior quality splits before use."""
import argparse
import json
from pathlib import Path

from malleable.llm.models import digest
from malleable.llm.quality import frozen_suite

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('LFM2.5-230M', 'Qwen3-0.6B', 'Qwen3.5-0.8B')
SPLITS = ('calibration', 'validation', 'held-out')


def grams(data, splits, width=8):
    return {tuple(row[i:i + width]) for split in splits for row in data[split]
            for i in range(max(0, len(row) - width + 1))}


def verify(new_root, old_root, text_suite):
    source = json.loads(text_suite.read_text())
    old_source = json.loads((ROOT / 'examples/llm-release-quality-text.json').read_text())
    source_files = source['source_files']
    new_names = [name for split in SPLITS for name in source_files[split]]
    old_names = {name for split in SPLITS for name in old_source['source_files'][split]}
    if len(new_names) != len(set(new_names)) or set(new_names) & old_names:
        raise ValueError('new source documents overlap prior or new splits')
    sources = {name: digest(ROOT / name) for name in new_names}
    report = {'schema_version': 1, 'text_suite_sha256': digest(text_suite),
              'source_document_hashes': sources, 'ngram_width': 8, 'models': {}}
    for model in MODELS:
        path = new_root / f'{model}.json'
        data, hashes = frozen_suite(path)
        previous, _ = frozen_suite(old_root / model / 'quality-suite.json')
        if (data.get('source_text_sha256') != report['text_suite_sha256']
            or data.get('source_document_hashes') != sources):
            raise ValueError(f'{model}: text source hash mismatch')
        previous_grams = grams(previous, SPLITS)
        for split in SPLITS:
            current = grams(data, (split,))
            sibling = grams(data, tuple(other for other in SPLITS if other != split))
            if current & previous_grams or current & sibling:
                raise ValueError(f'{model}/{split}: old or cross-split 8-token overlap')
        counts = {split: sum(len(row) - 1 for row in data[split]) for split in SPLITS}
        if counts['validation'] < 1024 or counts['held-out'] < 1024:
            raise ValueError(f'{model}: insufficient target count')
        report['models'][model] = {'suite_sha256': digest(path),
                                   'prior_suite_sha256': digest(old_root / model / 'quality-suite.json'),
                                   'split_hashes': hashes, 'target_counts': counts,
                                   'prior_8_token_overlap': 0, 'cross_split_8_token_overlap': 0}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-root', type=Path, default=ROOT / 'docs/evidence/controller-quality-20261005/suites')
    parser.add_argument('--old-root', type=Path, default=ROOT / 'build/cloud-quality')
    parser.add_argument('--text-suite', type=Path,
                        default=ROOT / 'examples/llm-controller-quality-text-20261005.json')
    args = parser.parse_args()
    print(json.dumps(verify(args.new_root, args.old_root, args.text_suite), indent=2))


if __name__ == '__main__':
    main()
