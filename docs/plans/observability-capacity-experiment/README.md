# Observability Capacity Experiment

## Goal

Estimate the unified timeline's production rate before implementing its Collector.
Measure fixed events and Routing attachments separately, then compare their combined
load with bounded asynchronous Device-to-Host and file-publication experiments.
TPS describes workload and perturbation, not a fixed bytes-per-token conversion.

## Accepted Changes

- Add disabled-by-default independent Device counters through `xpool.config`.
  Keep ten-millisecond buckets and read only after production quiescence. Count
  actual Hook branches even after legacy record storage fills. Preserve raw
  Device clock domains, allocation bytes, window overflow and unfinished tails.
- Use Qwen3-0.6B and DeepSeek-V2-Lite-Chat two-Device deployments sequentially.
  Screen input/output lengths 128/1024, 1024/128 and 512/512 with increasing
  Poisson load; supplement high-rate candidates with concurrency changes.
- Compare disabled counters, enabled counters and counters with concurrent drain.
  Preserve xbench timing, latency, throughput, queue and failure evidence.
- Measure pinned-Host asynchronous copies, publication, bounded queues and
  controlled fdatasync separately. Limit newly created payload files to 8 GiB;
  never clear system caches or alter system policy.
- Simulate collection periods 10, 25, 50, 100 and 200 ms, ownership until transfer
  completion, bounded Host residency and disk quota. Report measured demand and
  twice-demand capacity candidates. Persistent overload is unsustainable.
- Report five-minute disk demand and lifetime under the existing per-process
  1 GiB quota, reserving metadata and counting attachments and temporary files.

## Execution

Provide user-run scripts and instructions under `local_scripts`. A batch reserves
two Devices, has a two-hour deadline and retains failures, skipped work and cleanup
evidence. Hardware, topology, filesystem and free space are discovered at execution.
Use short screening windows, repeat high-production candidates and prioritize both
models. The operator runs remote experiments; local validation does not qualify
real serving or target-machine bandwidth.

## Validation

Test event mapping, concurrent counters, histogram exhaustion, safe readout,
attachment sizing, burst boundaries, in-flight ownership, Writer stalls, sustained
overload, disk quotas, I/O failures, interruption and evidence preservation.
Run repository-managed Python and Native checks; attempt Host ASAN/UBSAN.
Retain source identity, effective configuration, raw data and checksums.

## Boundaries

The unified timeline contract remains unchanged. Synthetic drain data is not an
implemented Collector. Uninstrumented target boundaries remain estimates or
unavailable; missing evidence never becomes a validated capacity recommendation.
Finite write tests do not prove indefinite disk sustainability. Keep this plan
until user-confirmed cleanup.
