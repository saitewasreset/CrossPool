# Observer prototype

This checkout-local experiment archives existing Observer snapshots, reconstructs
Fabric Invocation dependencies in SQLite, and exports one Chrome JSON trace per
confirmed physical GPU. It adds no production hooks or runtime recorder.

The supported experiment is `serving-001` in `tests/tests.toml`: Qwen3-0.6B,
attention TP=1/DP=1, one FfnAgent, one Executor Lane, and Decode Full plus Prefill
Breakable. Other topologies require a separate prototype extension.

## Prepare before reserving GPUs

Copy the updated checkout, including `local_scripts/`, to the remote project.
The provided wrappers target the Ubuntu/A100 machine's CUDA Toolkit 13.2. They
use the repository's managed uv environment and compile only `80-real`.
Preparation downloads dependencies, builds native code, checks the model path,
and collects the exact test without starting a serving deployment.

```bash
cd /home/LAB/luowenye27/CrossPool
bash local_scripts/observability/prepare.sh \
  --models /home/LAB/yezr/models \
  --output .xpool-cache/observer-preparation
```

Preparation exclusively creates its directory. Choose a new directory for a
second preparation; existing evidence is never automatically removed. Rebuild
after native or dependency changes. Do not invoke the A100 wrapper on the local
RTX 4070 SUPER; use the ordinary repository installation for local development.

## Run during the reservation

Reserve exactly two GPUs and substitute their full physical UUIDs below, with
the attention GPU first. The batch normalizes its own configuration and device
view, and clears inherited MPS addresses and CrossPool environment overrides.
The daemon owns attention MPS; FFN executes directly. Existing unrelated MPS
controllers are neither adopted nor stopped. Startup conflicts remain failures.

```bash
bash local_scripts/observability/run.sh \
  --config .xpool-cache/observer-preparation/xpool.toml \
  --devices GPU-ATTENTION-UUID GPU-FFN-UUID \
  --output .xpool-cache/observer-runs/experiment-001
```

The batch runs only the selected E2E case with strict requirements, using the
canonical `xtest` supervisor and managed cleanup. It streams merged stdout and
stderr to stdout while retaining the same bytes in `execution.log`. Preparation
does the same for collection in `collection.log`. Each shell wrapper also saves
its complete command output, including build output and entry-point errors, in
a unique `.xpool-cache/observer-prepare.*.log` or `observer-run.*.log`; its path
is printed at startup. Failed commands retain their original exit status.
Harness-owned per-process logs retain their existing file destinations.
After execution, it archives each Attempt separately and
converts each valid archive. A prior failed startup is never merged with a later
successful Attempt. Missing registration evidence produces unknown placement;
the FfnAgent Graph snapshot can independently confirm its GPU.

The result directory contains:

- `execution.json`, `execution.log`, `summary.json`, `source.patch`, and the
  exact prototype source used for the run;
- the complete retained `xtest` result tree, including JUnit and Attempt logs;
- `archives/attempt-N/run_manifest.json` and byte-preserved `raw/` inputs;
- `derived/attempt-N/timeline.sqlite`, `quality.json`, one `GPU-*.trace.json`
  per confirmed domain, and available Prefill/Decode query examples.

Return the sibling `experiment-001.tar.gz` bundle, including failed runs. Exit
zero means the selected test passed and a candidate representative archive was
produced. Exit one means test or representative-evidence failure; exit two means
a setup or input error. Real Trace acceptance still requires raw-file review and
visual inspection after the output is returned. Unverified cleanup requires
operator inspection; the script never deletes retained owners or restarts MPS.
If cleanup is unverified, the script skips archival/conversion and excludes the
potentially live `xtest` subtree from the bundle. That subtree remains at the
reported result path for operator inspection after resource retirement.

## Offline operations

Use the managed project environment with its installed native declarations.
These operations load types but do not initialize CUDA or execute native device
operations. Paths are explicit; new output directories must not already exist.

```bash
uv run --no-sync python -m local_scripts.observability archive \
  --source /path/to/retired/attempt-1 \
  --context /path/to/context.json --output /path/to/archive
uv run --no-sync python -m local_scripts.observability convert \
  --manifest /path/to/archive/run_manifest.json --output /path/to/derived
uv run --no-sync python -m local_scripts.observability query \
  --database /path/to/derived/timeline.sqlite \
  --generation GENERATION_HEX --instance 0 --sequence INVOCATION_SEQUENCE
```

`context.json` follows `RunContext` in `models.py`. Unavailable fields remain
null or explicitly unknown. A confirmed participant UUID requires a provenance
reference. Snapshot capture time is unknown; archive timestamps describe file
reading, not device execution. The remote batch supplies source/build identity,
effective launch configuration, exact request/response files, actual record
capacity, and registration-based process/placement evidence.

Conversion verifies the inventory's lengths and SHA-256 checksums before and
after publication. Schema corruption and structural identity conflicts fail
with source context. Missing artifacts, boundaries, unknown mapping, abnormal
exit and snapshot loss remain visible in `quality.json` and Invocation queries.
Failed conversion leaves its error in the remote archive; partial derived output
is diagnostic evidence and must not be treated as completed conversion.

SQLite stores exact integer device nanoseconds, source-local identities and
source record indices. Zero timestamps remain absent endpoints. Causal edges
carry endpoint references and participant scope, without a duration. Aggregate
conditions are never expanded into per-participant arrival observations.
Transport and Graph files remain independent evidence; Transport IDs are not
joined to Fabric IDs. Snapshot-level loss affects every associated query and
cannot be localized to one Invocation.

Open each `GPU-*.trace.json` separately in [Perfetto](https://ui.perfetto.dev).
Each file uses its own origin and converts nanosecond differences to Chrome's
microseconds. Raw integers remain in SQLite. Tracks group events by Generation, PE, role and
Lane, and reuse each activity row across Invocations. Only overlapping intervals
of the same activity require additional parallel rows. Invocation, Lease and
source references remain in event arguments instead of track names. The export follows
[Perfetto's Chrome JSON guidance](https://perfetto.dev/docs/getting-started/other-formats).
It contains no cross-GPU flows or calibrated unified clock. Host Graph event
times are retained as evidence and are not projected onto the GPU domain.

Inspect the Prefill and Decode queries against their source records and the two
GPU views. Acceptance requires complete round trips, different Leases reusing
the same Lane, and zero Fabric loss. Synthetic unit fixtures only test these
rules; they are never representative real Serving evidence. Performance and
clock qualification remain outside this prototype.

## Local verification

The prototype tests are CPU-only. Repository pytest still requires its normal
native ABI/dispatcher preflight. The canonical runner selects these tests
without requesting GPU resources:

```bash
uv run xtest run --suite unit --suite integration \
  tests/suites/unit/observability tests/suites/integration/observability
uv run ruff format --check local_scripts/observability \
  tests/suites/unit/observability tests/suites/integration/observability
uv run ruff check local_scripts/observability \
  tests/suites/unit/observability tests/suites/integration/observability
uv run ty check local_scripts/observability \
  tests/suites/unit/observability tests/suites/integration/observability
```
