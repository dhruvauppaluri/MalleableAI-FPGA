"""Verify a replacement quality campaign against every published/consumed boundary."""
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


def source_names(path):
    data = json.loads(path.read_text())
    return {name for split in SPLITS for name in data['source_files'][split]}


def verify(new_root, prior_roots, text_suite, prior_text_suites):
    source = json.loads(text_suite.read_text())
    names = [name for split in SPLITS for name in source['source_files'][split]]
    barred = set().union(*(source_names(path) for path in prior_text_suites))
    if len(names) != len(set(names)) or set(names) & barred:
        raise ValueError('replacement source documents overlap a prior frozen campaign')
    if any(Path(name).parts[:2] == ('docs', 'adr') for name in names):
        raise ValueError('ADR documents are barred by the inaccessible recovery tombstone')
    sources = {name: digest(ROOT / name) for name in names}
    report = {'schema_version': 1, 'text_suite_sha256': digest(text_suite),
              'source_document_hashes': sources, 'ngram_width': 8,
              'prior_campaigns': [str(path) for path in prior_roots], 'models': {}}
    for model in MODELS:
        path = new_root / f'{model}.json'
        data, hashes = frozen_suite(path)
        if (data.get('source_text_sha256') != report['text_suite_sha256']
            or data.get('source_document_hashes') != sources):
            raise ValueError(f'{model}: text source hash mismatch')
        prior = [frozen_suite(root / f'{model}.json')[0] for root in prior_roots]
        prior_grams = set().union(*(grams(item, SPLITS) for item in prior))
        overlaps = {}
        for split in SPLITS:
            current = grams(data, (split,))
            sibling = grams(data, tuple(other for other in SPLITS if other != split))
            overlaps[split] = {'prior': len(current & prior_grams),
                               'cross_split': len(current & sibling)}
            if overlaps[split]['prior'] or overlaps[split]['cross_split']:
                raise ValueError(f'{model}/{split}: prior or cross-split 8-token overlap')
        counts = {split: sum(len(row) - 1 for row in data[split]) for split in SPLITS}
        if counts['validation'] < 1024 or counts['held-out'] < 1024:
            raise ValueError(f'{model}: insufficient target count')
        report['models'][model] = {'suite_sha256': digest(path),
                                   'split_hashes': hashes, 'target_counts': counts,
                                   'overlaps': overlaps}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, action='append', required=True)
    parser.add_argument('--text-suite', type=Path, required=True)
    parser.add_argument('--prior-text-suite', type=Path, action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.new_root, args.prior_root, args.text_suite,
                            args.prior_text_suite), indent=2))


if __name__ == '__main__':
    main()
