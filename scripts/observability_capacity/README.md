# Observability Capacity Experiment

This user-run experiment estimates the unified timeline's resource demand before
its Collector exists. It uses the canonical `xbench` supervisor and measured
production Hooks. It does not qualify a production Collector or establish an
absolute traffic bound.

## Preparation and execution

Run from the repository root with its pinned uv environment. Rebuild/install the
changed project first, following the [installation instructions](../../docs/tutorials/quick-start.md#install):

```bash
uv sync --group dev --reinstall-package xpool --no-build-isolation-package xpool
```

Do not reuse an old
Native extension: the counter binding and generated stubs must match the source.

Copy [runtime.example.toml](runtime.example.toml) to a machine-local configuration
and set `vendor.model_base_uri`, or supply absolute paths for both model entries.
The base must include Qwen3-0.6B and DeepSeek-V2-Lite-Chat. Their actual deployment
geometry comes from the existing two-Device scenes. Check both models' weights
and tokenizer metadata before reserving Devices.

```bash
uv run python -m scripts.observability_capacity.run run \
  --config /absolute/path/to/capacity.local.toml \
  --devices GPU-ATTENTION-UUID GPU-FFN-UUID \
  --output /absolute/path/to/new-capacity-results
```

Use two distinct full GPU UUIDs, attention first. The default budget is 7,200
seconds; `--seconds` accepts 600 through 7,200. The operator owns the remote
execution and reservation. The script never logs in remotely or changes compute
mode. Attention MPS remains owned by the CrossPool daemon. Paths under
`/home/incoming` are rejected. Existing output directories are not reused.

`local_scripts/observability_capacity/run.py` is a machine-local convenience
entry point to the same maintained module. When transferring the change, include
the tracked source and tests as well as any local wrapper. The module command
above works without the wrapper.

The batch first runs the canonical Native suite, including the counter concurrency
case, with a ten-minute admission limit; failure stops the batch. It records
hardware inventory, topology, CPU/NUMA information, filesystem,
free space, revision, source diff and hashes. It first runs paired idle copy/file
probes, with buffered publication and controlled `fdatasync`. Then it screens
128/1024, 1024/128 and 512/512 input/output token lengths at 1, 4, 16 and 64
offered RPS, visiting both models before increasing load. It reserves time for
controls instead of promising the entire matrix will fit the reservation.

High event-rate and attachment-rate candidates are selected separately. The
event candidate receives an additional max-inflight=32 comparison; controls use
max-inflight=128. Five-minute controls compare disabled counting, counting, and
two counting runs with concurrent copy/file probes, one buffered and one with
controlled `fdatasync`. The default probe
schedule writes at most 5 GiB of synthetic payload files, below its 8 GiB limit;
benchmark logs, workload files and histograms are additional retained evidence.
Each probe checks free space and retains its complete and partial files.

Each attempt caps planned requests at 32,768 before serving startup. Request
timeouts are 90 seconds from HTTP start; client queueing remains separate.
Overloaded queues may require interruption rather than complete drain. Deadline
interruptions preserve the canonical partial measurement and cleanup evidence.
Unverified benchmark cleanup stops admission of further Device work. Failed or
unexecuted points remain failures or omissions, never successful empty results.

## Evidence and interpretation

- `outcomes.json`: commands, points, return codes, cleanup and missing evidence.
- `counter.*.json`: per-process/Device ten-ms histograms, raw origin, last activity,
  allocation size, arithmetic failure and coverage loss. Counters do not stop
  when legacy observer storage fills. Storage defaults to 120,000 buckets,
  or about 7.3 MiB per Device process, plus bounded Host graph counters. The Host
  object-size estimate is reported separately and conservatively counts shared
  integer references; it is not a process RSS measurement.
- Canonical xbench artifacts: TPS, throughput buckets, TTFT, ITL, queueing,
  failures and measurement-window classification. Input TPS includes cache hits.
- `drain.json`: copy submission/completion, Device copy time, Writer start,
  publication, Host wait/release and optional sync timing. A Host monotonic origin
  permits overlap checks against xbench client throughput buckets. CRC and hashes verify
  synthetic files. `EXPDRAIN` files are deliberately not `XPOOLC03` events.
- `estimates.json` and `summary.txt`: per-domain rate, 10/100/1000-ms peaks,
  quota lifetime, five-minute demand and 10/25/50/100/200-ms capacity candidates.

```bash
uv run python -m scripts.observability_capacity.run report \
  /absolute/path/to/capacity-results
```

The report preserves corrupt-domain errors while reporting valid domains.
Device histograms include startup, warmup and shutdown; they are not aligned to
the client window. Host monotonic and Device globaltimer domains remain separate.
Per-domain peak sums are conservative and do not prove simultaneous peaks.

Transport/Fabric event counts follow actual Hooks. Association and Routing event
counts project their real publication boundaries. Routing payload is actual rows
times effective top-k times eight bytes: int32 IDs plus float32 weights. Graph
Device events are estimates from Compute Hooks and Host forward boundaries;
Host replay counters are retained separately to avoid adding forward and replay
twice. Native Graph structure file bytes are measured separately.

File demand explicitly assumes one KiB overhead per Routing attachment and
full-chunk headers plus up to one partial-chunk header per default 100-ms poll. These are rough sizing
assumptions, not measured serialization bounds. The report shows raw payload
and this estimate separately, including metadata reserve and twice-rate disk
candidates. It never presents an empty histogram as zero demand.

Capacity replay retains Device event slots until copy completion and Host blocks
until publication; Writer delay does not directly retain Device slots. Arrival
placement is conservative within ten-ms buckets, including an extra boundary
bucket allowance. Persistent overload remains unsustainable regardless of buffer
size. Host forward/eager Graph queues, Routing pools, legacy reconstruction,
metadata and allocation alignment require additional budget beyond these payload
candidates. Synthetic service floors are divided among same-Device producers
and the three default serving producers sharing the Writer filesystem.

Finite buffered writes can be served by page cache. `fdatasync` measurements
expose writeback waits but still do not establish indefinite disk sustainability.
The script does not clear shared caches or change system policy. Probe contention
and counting can perturb serving. Quota-limited probes may occupy only a short
part of a five-minute run; compare their overlapping Host-time buckets as well as
whole-window controls before interpreting
the estimated production ceiling. Real no-loss acceptance requires the future
Collector on the target machine.

## Focused checks

```bash
uv run xtest run tests/suites/unit/devkit/test_capacity_experiment.py \
  tests/suites/integration/devkit/test_capacity_experiment.py
```

The Native `capacity_counter_test.cu` exercises concurrent attempts, collective
thread selection, bounded coverage and safe post-completion readout. The Host
`capacity_counter_index_test.cpp` checks timestamp boundaries and is suitable for
ASAN/UBSAN. A missing Device is an explicit unexecuted Device qualification.
