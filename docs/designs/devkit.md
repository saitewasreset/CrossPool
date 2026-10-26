# Devkit

Devkit observes real production execution through typed Hook Points. Transport,
Fabric, and FFN computation remain production responsibilities.

Debug configuration is process-global and immutable after native
initialization. Native owners read the installed value directly; Debug policy
stays out of runtime state, adapters, and Kernel arguments. The configuration
contains Transport, Fabric, Graph, and FFN Routing Observer options.

Each Observer records evidence from the real execution path and owns
process-local storage. Observer state is never stored in a Transport or Fabric
arena. Fabric and FFN Routing Observer allocations are included in FFN memory
admission through their native sizing queries. Transport Observer storage is a
normal attention-side allocation. The AtnAgent post-capture device-memory
observation therefore already reflects its bytes; it needs no duplicate
planning field.

Every Observer owns its record and snapshot values and exposes reads through
its own `xpool.native.devkit.<observer>` submodule. An enabled Observer that
cannot allocate or install fails startup. Exhausted Device record capacity
increments the Observer's dropped count; an Observer destination-bounds failure
omits that record rather than corrupting storage or changing execution.
Snapshot read failures are reported to the Host reader and do not alter an
already completed invocation.

## Hook contract

Host and Device call sites use the neutral typed surface:

```cpp
Point::hooks(Point::Context{...});
```

A production call site names only its owning Point. Host and Device adapters
form an explicit compile-visible catalog. Hook Points provide observation; the
production path owns control flow and execution selection.

Observe calls are additive, so the surrounding production code remains readable
and complete without them. Collective Device Points are reached by every
participating CTA thread; adapters that need only one observation select thread
zero themselves, while an Observer that copies a collective payload may use the
complete CTA.

## Protocol and Graph evidence

Protocol events correspond to real publication or observation boundaries.
Events are not inserted merely to separate adjacent calls or to expose a
debug-only algorithm.

Capacity evidence is recorded by the FfnAgent that selected the installed
Capacity and contains both live rows and selected row capacity. Delivery
evidence is recorded from the production delivery derivation. Coordinator
records do not infer either fact.

Routing evidence follows the collective `RoutingMetadataPublished` event after
the production payload and identity are published. Graph evidence records
Primary captures and installed Lane Graphs at their owning lifecycle Points.
Observers record evidence; runtime lifecycle and Graph topology remain owned by
production execution.
Native Graph snapshots expose CUDA node kinds as underlying integers; the
Python Graph Observer resolves official cuda-python enum names only when
constructing its JSON presentation.

## Experimental capacity counters

`debug.capacity_counter` enables independent, bounded ten-millisecond histograms
for capacity experiments. Configuration is resolved through `xpool.config` before
Native initialization. `enable` and an absolute `outdir` are supplied together;
`bucket_count` defaults to 120,000 and is bounded by 360,000. The environment
settings use the `XPOOL_DEBUG_CAPACITY_COUNTER_` prefix.

One allocation is installed on the process Device before producer launch and
Graph Capture. Its bytes are reported separately from timeline collection
budgets and are already resident before the FfnAgent free-memory registration.
Counters operate independently of legacy observer enablement and retention.
They count real Transport/Fabric boundaries, association/publication projections,
Routing payload bytes and a separate Graph-activity estimate. FfnAgent collective
boundaries select CTA thread zero. Window exhaustion and arithmetic overflow
invalidate complete-window estimates instead of appearing as zero production.

Readout is once per experimental process lifetime. Instance readout follows a
successful Native detach, which supplies Device synchronization. Agent readout
follows the local `DRAINED` report, including the production drain boundary.
Native finish selects the allocation's owning Device, copies the bounded
histogram, clears its Device view and frees it. Failed retirement supplies no
safe readout. Export errors propagate. Host forward/capture/replay counters use
their own monotonic domain; raw Device times are not a Host mapping.

The maintained [capacity experiment](../../scripts/observability_capacity/README.md)
provides user-run serving, synthetic asynchronous copy/publication probes and
offline estimation. Its `EXPDRAIN` files are synthetic I/O evidence. These
facilities do not implement the timeline Collector or establish real-machine
loss, overhead, sustainable bandwidth or clock-calibration acceptance.

## Serialized observations

Python Fabric and Transport writers own concrete JSON declarations shared by
readers and fixtures. These projections differ from native snapshots: they
flatten role facts, serialize enums, derive durations and add process or model
context. Matching-role native getters determine field types and nullability.
Incomplete records remain diagnostic input: AtnAgent lane and lease facts are
nullable until AdmissionObserved, and Coordinator lane and lease facts until
Scheduled.

Transport snapshot filenames combine PID, endpoint site, the Model ID's
[URI component representation](control-plane.md#configuration-and-integration)
and Instance Rank. Each model retains its own process-local snapshot file;
the JSON payload carries its canonical scalar Model ID for readers.

The production SGLang Devkit package owns its lightweight Graph-event
declaration, shared by the event writer and readers independently of Graph
runner implementations. Native Graph snapshots retain their existing production
model. [Tooling](tooling.md#shared-serving-lifecycle) owns file-input validation
and the separate semantic qualification assertions.
