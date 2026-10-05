# Zephyrus cloud evidence archive

This branch publishes the checksummed Zephyrus standalone acceptance and checkpoint-lineage handoff for a cloud workspace. The 11 numbered parts reconstruct one gzip tar archive. It contains 34 byte-for-byte Zephyrus evidence files, the three checkpoint download manifests, a read-only held-out consumption export, Qwen3 candidate calibration data, a verified-file inventory, and recovery scripts. The existing main branch also holds all 30 indexed benchmark exports in docs/evidence/benchmarks-20261003-14of20 and docs/evidence/benchmarks-20261003-complete.

From a checkout of this branch:

    cat docs/evidence/cloud-handoff-20261004-archive/part-* > /tmp/cloud-handoff-20261004.tar.gz
    echo '1a36341516073952ad568bb5075ac683e8e5a15fe2b92ea0f8476677d2993be8  /tmp/cloud-handoff-20261004.tar.gz' | sha256sum -c -
    tar -xzf /tmp/cloud-handoff-20261004.tar.gz -C docs/evidence
    python3 docs/evidence/cloud-handoff-20261004/verify_handoff.py

Read docs/evidence/cloud-handoff-20261004/README.md inside the archive before continuing. The public original checkpoints are pinned to exact Hugging Face revisions and hashes; they are not embedded in Git. The 1.7 GB calibrated Qwen3 derived tensor file is likewise reconstructed and hash-checked from the official checkpoint plus published parameters. No held-out or benchmark rerun is authorized by this transfer.

Archive SHA-256: 1a36341516073952ad568bb5075ac683e8e5a15fe2b92ea0f8476677d2993be8
Standalone schema-v3 manifest SHA-256: 23f51698475d75850bdfcd8222fe87c866b55e76986b7ec3b782b30b752a2118
