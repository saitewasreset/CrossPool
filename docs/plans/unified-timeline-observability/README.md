# Unified Timeline Observability: Shared Contract

## Goal

Define the versioned artifact contract that allows bounded runtime collection and offline
reconstruction to be implemented independently. The target is one single-Host
Transport/Fabric/Graph/Routing timeline with protocol-derived causality,
paired Host/Device clock sampling, and explicit uncertainty and data quality.

The independently usable contracts are:

- [Shared contract](shared-contract.md): events, identities, Native interfaces,
  encoding, configuration, lifecycle, attachments, and event reconstruction.
- [Calibration](calibration.md): deterministic model and numerical contract.
- [Reference fixtures](fixtures/README.md): fixed inputs, byte vectors, expected
  reconstruction and calibration outputs.

## Baseline

Current implementation owners are [Devkit](../../designs/devkit.md),
[Transport](../../designs/transport.md), [Fabric](../../designs/fabric.md),
[FFN Execution](../../designs/ffn-execution.md), and
[Control Plane](../../designs/control-plane.md).

Transport and Fabric mutate one bounded Record per local operation and read
snapshots after quiescence or synchronization. Their local trace identifiers
are not cross-process identities. The concrete correlation seam is
`transport::execute_request()` calling
`fabric::atnagent::execute(fabric_arena, transport_arena)`; the Fabric call
constructs the actual InvocationKey before Submission publication.
The InvocationKey comment attributing its sequence to an Instance rank must
be corrected when that path is instrumented; its allocation algorithm remains
production-owned.

Native Graph snapshots describe installed structure. SGLang Graph events use
Host `monotonic_ns()` and currently append and flush each event synchronously.
Routing snapshots expose InvocationKey, layer, and tensors; their internal
headers also retain Lane/Lease. Routing payload overflow currently returns
without a quality counter. These are collection changes required by the target,
not facilities already provided by the baseline.

Current retirement additionally queues PE-wide outbound NVSHMEM quiet after
local producer completion. Collector teardown must respect that boundary, as
well as Instance IPC detach and CUDA completion. Source declarations remain
authoritative for implemented interfaces; proposed declarations below become
authoritative only when implemented.

## Accepted Changes

- Register observation sources only for Instance Rank, AtnAgent, and FfnAgent
  processes. The Daemon distributes identities and brokers descriptors without
  becoming a Producer. Each Generation publishes its immutable `pe_members`
  mapping from PE to the member's Host/Device Producers before referencing
  chunks. Coordinator events belong to the first FfnAgent PE and share its
  Producer and local Fabric trace-ID space.
- Replace mutable Device observation Records with immutable fixed-size events.
  Carry the operation facts known at each Hook; distinguish absent facts from
  valid zero values. Keep production RequestMetadata unchanged.
- Allocate a fresh shared Transport operation identity after every successful
  Mailbox Claim, independently of event capacity. Publish the association at
  actual entry into Fabric Submission preparation, then observe publication
  separately. Preserve rank-local many-to-one and Graph one-to-many relations.
- Add fresh Graph execution identities and bounded, Stream-ordered dynamic
  Context. Separate Host calls, Device activity, templates, and long-lived Lane
  service Graphs. Attach Routing evidence by exact attachment identity.
  Dynamic Context storage is process-wide in the actual Graph owner:
  Instance Rank for SGLang, FfnAgent for native FFN. The IPC region carries only
  Operation identity and is allocated whenever an enabled consumer needs it,
  including independent Transport observation. Diagnostics consume committed
  events; direct mutable-storage reads require quiescence or synchronization.
- Provide one bounded Collector/Writer per process, shared by independently
  enabled observers and unified mode. A process-local background collection
  thread periodically drives asynchronous Device-to-Host event transfer and
  completion checks, reclaiming Device slots after complete transfer without
  waiting for file publication. Writer file I/O must not block this thread.
  Add `debug.timeline` through
  `xpool.config`; collection policy is immutable before native initialization
  and Graph Capture.
- Produce versioned raw Events, metadata, quality checkpoints and Graph/Routing
  attachments. Reconstruct causal relations and valid intervals from retained
  evidence without preserving existing Observer getters or serialized shapes.
  Independent Observer flags select collection classes; unified mode enables
  all four classes and calibration through the shared Collector.
- Use ten-attempt calibration blocks and a positive-slope affine feasible
  region, with two held-out validation attempts. Preserve raw clocks, model
  assumptions, unavailable intervals, and numerical uncertainty.
- Downstream export uses standard native Perfetto protobufs and Trace Processor.
  ClockSnapshot represents an accepted mapping; it supplies no calibration
  evidence by itself. Uncalibrated domains retain separate timing artifacts.

## Resource and Lifecycle Contract

| Resource | Default and accounting |
| --- | --- |
| Device collection | 8 MiB per process/Device, split by formula into the event ring, per-Endpoint `observation_region`s, process-wide Graph Context, and per-process calibration sampler storage; see [the device collection budget composition](shared-contract.md#device-collection-budget-composition) |
| Host collection | 32 MiB per process, including queues, staging, Writer-held blocks and metadata |
| Raw chunk | At most 1 MiB including the 128-byte header |
| Collection and periodic sampling | 100 ms; at most one in-flight sample per sampler |
| Runtime disk | 1 GiB per process, including metadata, temporary files, and attachments; reserve 8 MiB for metadata |
| Additional shutdown wait | At most 5 seconds, bounded by the remaining production shutdown deadline |

Device admission uses actual requested bytes for every distinct simultaneously
live collection allocation, including packing, ReadToken, required state and
alignment in addition to the categories listed above. The logical capacity
example alone cannot prove that the runtime fits the budget.

Routing tensor pools and Graph structural artifacts keep their independent
memory capacities. Added identity fields enter their sizing and admission.
The event budget is not a bound on all Observer memory. Startup rejects a
configuration whose minimum allocations or metadata reserve exceed its budget;
it never silently expands a budget. Report each allocation category separately.

Hooks perform no file I/O and never wait for the Writer, a free slot, or disk
quota. Full collection storage drops new events and counts the loss. Disk
exhaustion stops collection, preserves published files, and records the stop
reason. Atomic same-directory publication provides a recoverable complete-file
prefix under a healthy filesystem; it does not promise power-loss durability.
Timed-out CUDA or IPC users retain their resources until production retirement
proves safe release. Crashed or unpublished tails remain unknown.
Transfer failures use public io_error and failed export while retaining the first
stop reason. Completion uncertainty independently requires resource retention,
an unknown tail, and non-exact totals; see [quality and transfer failures](
shared-contract.md#quality-attachments-and-configuration).

## Anomalous Transport Events

[Anomalous Transport evidence](shared-contract.md#anomalous-transport-evidence)
must preserve received invalid request fields and actual local terminal
boundaries through an explicit anomaly representation. Invalid values remain
raw observed facts rather than legal defaults. A terminal Event never supplies
missing earlier timestamps or proves full-path completeness.

TransportAnomaly (kind 150) and RawRequest retain their validation and wire
payload. Ordinary Event enum validation stays strict. No typed legacy Record
or snapshot evidence report is produced.

## Implementation and Validation

Downstream work starts from the same versioned fixtures: strict raw parsing and
causal reconstruction; deterministic calibration; runtime collection and
event reconstruction; protocol/Graph identities; Routing attachments; paired
sampling; finally native Perfetto/query integration.

Both Native and offline implementations must independently read and write the
valid binary vector, reject the invalid vectors by structured code, reconstruct
the same identities and causal relations, and produce the exact calibration
expectations.
Fixture changes require a coordinated contract version update; never support
two conflicting interpretations of v6.

Contract and fixture version 6 supersedes versions 1 through 5 without a
compatibility mode. Use XPOOLC06 chunks and version 6 metadata, quality and IPC
descriptors; update raw vectors and hashes together. Event size stays 256 bytes,
bytes 136–143 are reserved zero, and presence bit 8 is forbidden. Local trace
IDs and snapshot operation counters are removed; Transport Operation and Fabric
Invocation plus local Producer/role/boundary identify protocol observations.
Calibration retains algorithm_version=1 and its exact numerical expectations.

Graph structure and Routing tensors remain attachments, with no requirement to
preserve existing Observer export shapes. Routing safetensors uses version,
Run, Producer and per-attachment records rather than old snapshot counters.
Independent collection retains its current mode boundary: calibration is
installed only by unified mode. Partial class selection remains useful but
cannot supply disabled-class evidence or an uncalibrated cross-domain timeline.

Focused tests must cover independent-observer IPC allocation and policy mismatch,
Generation-scoped PE reassignment, Host/Device member ownership, malformed and
conflicting member maps, event/metadata PE disagreement, Coordinator placement,
Daemon exclusion from collection and missing-artifact checks,
actual Graph-owner isolation, entry association retained after failure or absent
return, rejection before Submission preparation, diagnostic read preconditions,
and concurrent reservation/commit/reclamation, multiple
CTAs/Streams, IPC identity lifetime, repeated Replay and Context isolation,
partial events, opening/middle/terminal loss, repeated-fact recovery,
new-identity semantic duplicates and conflicts, reserved bytes and presence bits,
unknown final counts, periodic background transfer during serving, Device slot
reclamation before file publication, slow Writer isolation, bounded Host
exhaustion, I/O failure, quota exhaustion, and background-thread shutdown.
Follow [test conventions](../../../tests/README.md), repository-managed Ruff,
type, and clang-format checks. Native Host tests attempt ASAN/UBSAN with the
required execution approval.

Device admission checks must include implementation allocations beyond the
logical example, shared frame/token/control counted once, Host/Device placement,
alignment, over-budget rejection, overflow, and unknown required sizes. Transfer
failure checks must cover public io_error, bounded specific diagnostics, retained
in-flight resources and charges, and a later uncertain transfer after an earlier
stop reason. Also verify that a file I/O failure with independently proven
completion can retain exact counts, while unknown work never gains invented
copied/exported/lost counts.

Real acceptance follows [Qualification](../../designs/qualification.md).
Prepare a two-Device batch for user execution: Qwen3-0.6B Dense followed by
DeepSeek-V2-Lite-Chat MoE, Prefill/Decode/Replay, actual attention MPS and direct
FFN processes, idle/serving/stress sampling, bounded-capacity runs, and
observation-disabled comparisons. Retain separate Attempts, physical Device
mapping, configurations, raw events/samples/attachments, checksums, and losses.
Report measured coverage, drift, uncertainty, and overhead without an invented
accuracy or performance threshold. Fixtures cannot substitute for these runs.

## Out of Scope

This contract specification contains documents and fixed machine-readable data.
It adds no runtime implementation, parser, generator, or persistent test code.
Full user Request/Batch/Step context, KV causal chains, exact per-participant
observations derived from aggregate conditions, per-CUDA-node profiling,
cross-Host support, other platforms, continuous monitoring, and alerts remain
outside this workstream. Current design documents will absorb implemented
behavior after subsequent runtime implementation and acceptance.
