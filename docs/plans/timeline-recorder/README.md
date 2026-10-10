# Independent Timeline Recorder

## Goal

Implement bounded runtime collection independently of the existing Observers.
The daemon owns Session identity, expected Producers and disk grants; each
process owns its Collector and Writer. The supported boundary remains one Linux
host, CUDA devices, managed attention MPS and SGLang.

## Baseline

Existing Observers retain mutable records and require quiescence for snapshots.
They are not concurrent collection buffers. The Observer prototype demonstrates
separate GPU Chrome JSON projections, not runtime recycling or crash recovery.

## Accepted Changes

Use Native Host/Device pools with immutable records, release commits, acquire
collection, atomic reservation/sealing and non-wrapping epochs. Seal excludes new
reservations. Collection requires every reservation to commit or terminate.
Device slots recycle only after complete Host receipt; Host slots recycle only
after publication or explicit discard. Reservation retries are limited to 16;
contention loss is explicit. Inference never waits for collection, grants or I/O.

Defaults are 8 MiB per Device Producer, 32 MiB Host storage per Producer, 1 MiB
maximum Chunk, 100 ms collection, 1 GiB Session including 8 MiB metadata, and
5 seconds additional shutdown wait capped by the production deadline. Control,
IPC identity, metadata, staging and Writer-held storage count against budgets.
Host Producers are process-local; Device Producers are process/device-local.
Record from initialization through exit, retaining actual coverage windows.

Provide Timeline-only shared Transport endpoint and operation identity in an
optional IPC region. Instance Claim generates each operation identity on Device;
Request publication makes it visible to AtnAgent. Replay does not reuse identity.
Identity faults degrade collection, not production. Record lifecycle and minimal
Transport, Lane Lease and Compute protocol boundaries across all four roles.
Compute brackets include branch dispatch overhead. Do not guess Fabric links.

## Interface And Data Changes

Add immutable `debug.timeline` configuration and registered environment options.
Enable and output directory are paired. Disabled collection allocates no pools,
identity regions or additional timing nodes. Expose Native sizing, Producer
lifecycle and owning Chunk leases under `xpool.native.devkit.timeline`; generated
stubs remain authoritative. Native owns atomics, CUDA transfers and storage;
Python owns Session RPC, independent Collector/Writer scheduling and files.

Use versioned little-endian fixed-width binary records. Published headers retain
Session/Producer identity, Chunk sequence, counts, sequence range, byte length,
collection window and SHA-256. Attempt counters include drops; overflow disables
the identity domain rather than wrapping. Quality counters have reserved space.

Grant small batches on demand, deduplicating requests by Producer creation
identity and grant sequence. Charge before creating files. Retain crash grants;
unreachable daemon permits only existing credit. Publish unique `.partial` files
by same-directory atomic rename after closing. Preserve failed partial files and
published prefixes; never rotate data. Budget metadata update coexistence.

Add `xpool timeline verify/export`. Validate identity, schema, checksum, sequences
and quality; project each confirmed Host/GPU clock domain separately. Export to
an explicit new directory outside the Session. Preserve integer timestamps and
source endpoints. Distinguish complete, degraded and invalid outcomes.

## Implementation And Validation

1. Configuration, Session identity, expected participants and idempotent grants;
   unit and CPU integration coverage for validation, quota and failure paths.
2. Native bounded pools, commit/seal/acquire, async transfer, owning leases and
   epochs; Host and single-device concurrency, overflow, IPC and Graph coverage.
3. Independent Writer, binary publication and deadline-aware shutdown; test slow
   I/O, abandoned reservations, live leases, partial publication and crash tails.
4. Production hooks and offline CLI; test identity across Replay and Observer
   independence. Supply user-run two-device Qwen3-0.6B batch qualification with
   raw artifacts, pressure cases and report-only overhead comparisons.

Each layer runs focused tests and repository format/type/lint checks. Attempt
Host ASAN/UBSAN outside the sandbox. Multi-device evidence remains outstanding
until the batch results are returned. Do not publish unqualified targets as
accepted current architecture.

## Out Of Scope

Request/Batch Context, Transport-to-Fabric mapping, complete Fabric/KV semantics,
OTLP projection, clock calibration, cross-domain durations, cross-host support
and full query workflows remain later work. No compatibility reader for old
Observer files. Normal close is distinct from loss-free coverage. Preserve
unknown crash tails; never rescue CUDA or files inside signal handlers.
