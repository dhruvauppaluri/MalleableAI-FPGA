# Continue the local LLM release on the Zephyrus

Use this guide to move from the M1 Pro Mac to the Zephyrus with the Ryzen AI 9
and RTX 5070 Ti. Windows stays installed. WSL2 adds an Ubuntu workspace inside
Windows; no dual boot or replacement operating system is needed.

The code is on branch `codex/local-llm-platform`, in draft
[PR #3](https://github.com/dhruvauppaluri/MalleableAI-FPGA/pull/3).
The current task handoff is [STATUS.md](STATUS.md). Read it before continuing;
its evidence and remaining work can change after this guide was written.

## 1. Prepare Windows and WSL2

Install or update the NVIDIA Windows driver for the laptop. Then open PowerShell
as Administrator and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if prompted. Open Ubuntu from the Start menu and create its Linux user
account. If WSL is already installed, inspect it instead of reinstalling:

```powershell
wsl --update
wsl --list --verbose
```

The Ubuntu distribution should show version `2`. If it shows `1`, use its exact
listed name with `wsl --set-version DISTRIBUTION_NAME 2`.

See [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
The NVIDIA Windows driver supplies GPU access inside WSL. Do not install a
Linux NVIDIA display driver in Ubuntu; follow
[NVIDIA's CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## 2. Get the existing branch

Run the remaining shell commands in the **Ubuntu terminal**, unless marked
otherwise. Install the basic tools:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3 python3-venv python3-pip \
  iverilog yosys autoconf flex libfl-dev bison help2man
mkdir -p ~/projects
cd ~/projects
git clone --branch codex/local-llm-platform \
  https://github.com/dhruvauppaluri/MalleableAI-FPGA.git
cd MalleableAI-FPGA
git status --short
git branch --show-current
```

Keep the checkout in `~/projects`, inside Ubuntu's filesystem. Microsoft's
[filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
recommends storing files on the same operating system as the tools using them.
Avoid running these Linux builds from `/mnt/c/...`.

If the checkout already exists, inspect its branch and changes before fetching
updates. Preserve local edits; do not replace the folder with another copy.

## 3. Carry over models and experiment history

If the Mac already has `build/Zephyrus-Transfer`, copy that entire prepared
folder and follow its `START-HERE.md`. It includes an offline Git bundle,
local data snapshots and checksums. The folder is ignored local data and is
not downloaded by cloning GitHub. Use its fresh-job-directory instructions
to avoid dispatching historical queued jobs on the new machine.

Git transfers code and documentation. It does **not** transfer ignored local
models, databases, traces, or release evidence.

On the Mac, stop the workbench cleanly and let running jobs finish or explicitly
cancel them before copying databases. Preserve the existing canceled jobs.
Copy these folders when present, retaining their paths relative to the repo:

| Folder | Contents |
| --- | --- |
| `build/models` | Three downloaded checkpoints, download manifests, frozen quality suites and reference caches |
| `build/llm-runs` | CLI experiment database, artifacts, acceptance records and traces |
| `build/ide-jobs` | Durable workbench jobs and research records |
| `build/release-evidence` | Local verification reports and 100-token tiny RTL evidence |
| `build/real-benchmarks` | Performance stores, if created |
| Other explicitly configured stores | Results saved with custom `--store` or `--job-root` paths |

An external SSD or network transfer works. Preserve entire store directories,
including their SQLite files, any associated journal files, and artifact files.
Do not transfer only the database. Keep the Mac copy as a backup.

For example, in the **Mac terminal**, after stopping writers and confirming
that the four listed directories exist:

```bash
cd '/Users/dhruvauppaluri/Documents/ChatGPT/Research Project 2'
tar -czf /Users/dhruvauppaluri/Downloads/malleable-local-data.tar.gz \
  build/models build/llm-runs build/ide-jobs build/release-evidence
shasum -a 256 /Users/dhruvauppaluri/Downloads/malleable-local-data.tar.gz
```

Omit missing directories and add other stores you actually used. Transfer the
archive to the Zephyrus. In Ubuntu, check its SHA-256 against the Mac output,
then extract it into the fresh repository. Replace the example archive path:

```bash
cd ~/projects/MalleableAI-FPGA
sha256sum /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/malleable-local-data.tar.gz
tar -xzf /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/malleable-local-data.tar.gz
```

Extract into a fresh checkout with no existing stores at those paths. If it
already contains results, copy into a separate staging directory and reconcile
them first; do not overwrite its databases.

Rebuild `.venv`, `node_modules`, and simulator binaries on Linux. Mac binaries
cannot serve as the Linux build cache. Preserve transferred historical evidence
with its original provenance; produce separate evidence for new Zephyrus runs.

If copying checkpoints is impractical, this separate command explicitly
downloads the three supported official checkpoints:

```bash
.venv/bin/python tools/download_llm_models.py --destination build/models
```

Run it only after setting up Python below. Copy the existing frozen suites and
history even if you download weights again. Startup and inference do not
download models.

## 4. Install the pinned simulator and Python/frontend environment

The repository's CI uses Verilator `v5.050`. Build the same version in a separate
tools directory:

```bash
mkdir -p ~/tools
git clone --depth 1 --branch v5.050 \
  https://github.com/verilator/verilator.git ~/tools/verilator-v5.050
cd ~/tools/verilator-v5.050
autoconf
./configure
make -j2
sudo make install
verilator --version
cd ~/projects/MalleableAI-FPGA
```

If that tools directory already exists, check it before cloning. `libfl-dev`
above supplies `FlexLexer.h`, required by the Verilator build.

Install Node.js 22 using the [official Node.js download instructions](https://nodejs.org/en/download)
for Linux, inside Ubuntu. Confirm `node --version` reports version 22 and that
`npm --version` succeeds. Then create a new Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch wheel using the
[official PyTorch selector](https://pytorch.org/get-started/locally/): Linux,
Pip, Python, and a CUDA build compatible with the installed Windows driver and
RTX 5070 Ti. Run its command inside the active virtual environment. Record the
chosen package versions; do not copy the CPU-only wheel command from CI.

```bash
python -m pip install -e '.[llm,ide,test]'
npm --prefix frontend ci
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda") + 1)'
```

The last command must actually compute on the GPU. If it fails, resolve driver,
WSL, or wheel compatibility before collecting CUDA evidence. RTL simulation can
still run on the CPU. Keep the laptop plugged in, with adequate ventilation.
With 32 GB RAM, check Ubuntu's available memory using `free -h`; WSL memory
limits and simultaneous model loads can affect whether a job fits.

## 5. Verify and open the workbench

```bash
make verify-llm PYTHON=.venv/bin/python
make release-tiny PYTHON=.venv/bin/python
.venv/bin/python -m malleable.ide --model-root build/models
```

Open `http://127.0.0.1:8765` in the Windows browser. Leave the Ubuntu terminal
running while using the workbench. The service defaults to `build/ide-jobs`.
The test suite needs no checkpoints; real-model release checks require them.

Verilator runs on the CPU. CUDA helps GPU model execution and suitable reference
evaluations; it does not automatically accelerate RTL simulation. Compare the
same workload and settings on both machines before claiming a CPU speedup.

## 6. Continue the remaining release tasks

Use [STATUS.md](STATUS.md) for the latest task state and
[local-llm-platform.md](local-llm-platform.md) for exact commands and contracts.
At this handoff, the sequence is:

1. Investigate Qwen3 INT8 quality. Its short RTL execution matched ISA state,
   but held-out next-token agreement was 83.50%, below the 90% requirement.
   Check conversion, tokenizer/evaluation alignment, and floating reference;
   fix reproducible defects and retain failed candidates as research results.
2. Complete frozen quality evaluation and fresh full-RTL acceptance for
   Qwen3.5 and LFM2.5. Do not resume historical canceled jobs automatically.
3. Run the staged 30 performance experiments, then predictor and RL held-out
   evaluations, and complete workbench integration/accessibility tests.
4. Assemble and pass the strict standalone release manifest.
5. Establish the Zephyrus GPU-only baseline and greedy hybrid agreement/performance
   after standalone passes. The local Qwen3-1.7B verifier is needed for this stage.
6. Assemble the full-release evidence, check CI, and update draft PR #3 for review.

The separate, explicit verifier download is:

```bash
.venv/bin/python tools/download_llm_models.py \
  --destination build/models --include-qwen3-verifier
```

Use `gpu-generate` and `hybrid-generate` only with a passing standalone manifest,
as described in the platform guide. Keep the quality limits at NLL degradation
at most 5% and next-token agreement at least 90%. Full release remains incomplete
until its required measured results and manifests pass. No automatic main merge.

## 7. Prompt for the agent on the Zephyrus

Open this Ubuntu checkout in an editor/agent that can execute commands in WSL.
Confirm its terminal is Ubuntu and its working directory is the checkout. Give
it this prompt:

> Continue the approved local LLM release on this Zephyrus. Read AGENTS.md,
> docs/STATUS.md, docs/zephyrus-handoff.md, and docs/local-llm-platform.md first.
> Work on codex/local-llm-platform and update existing draft PR #3 against
> codex/model-aware-baseline. Inspect transferred models and complete stores;
> preserve historical identities, evidence and canceled jobs. Set up the pinned
> Linux RTL tools, frontend dependencies and CUDA-enabled PyTorch in WSL2.
> Verify GPU computation and run make verify-llm. Investigate Qwen3's quality
> failure, finish Qwen3.5/LFM acceptance, the staged benchmarks, learning
> evaluations and UI tests. Run CUDA/hybrid acceptance after standalone passes.
> Keep the approved quality thresholds and scope. Update docs/STATUS.md with
> results and remaining gates, publish reviewable commits, inspect/fix CI, and
> do not merge automatically. Check current CLI help before constructing jobs.

Repository files carry the handoff; the new agent does not need this chat.
Checkpoints, datasets and private files stay out of commits. AWS and physical
FPGA deployment remain outside this release.
