#include <xpool/devkit/timeline/recorder.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <time.h>
#include <unistd.h>

#include <xpool/debug/options.cuh>
#include <xpool/devkit/timeline/device.cuh>
#include <xpool/devkit/timeline/diagnostics.hpp>
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {

namespace {

std::shared_ptr<Recorder> host_owner;
std::shared_ptr<Recorder> device_owner;
std::uint64_t host_id = 0;
std::uint64_t device_id = 0;
int installed_device = -1;
std::mutex installation_mutex;

// Collector events belong to a separate nonblocking stream. Ignore captures
// in other threads only while querying their receipt; preserve the caller's
// capture policy and restrictions on events captured in this thread.
class CaptureModeGuard {
public:
  CaptureModeGuard() { C10_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&previous_)); }
  ~CaptureModeGuard() { C10_CUDA_CHECK_WARN(cudaThreadExchangeStreamCaptureMode(&previous_)); }
  CaptureModeGuard(const CaptureModeGuard &) = delete;
  CaptureModeGuard &operator=(const CaptureModeGuard &) = delete;

private:
  cudaStreamCaptureMode previous_ = cudaStreamCaptureModeThreadLocal;
};

cudaError_t query_receipt(cudaEvent_t event) {
  const auto capture_mode = CaptureModeGuard{};
  return cudaEventQuery(event);
}

std::uint64_t now() {
  timespec value{};
  TORCH_CHECK(clock_gettime(CLOCK_MONOTONIC, &value) == 0, "xpool timeline Host clock read failed");
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(value.tv_nsec);
}

PoolView geometry(std::uint8_t *base, std::size_t bytes, std::size_t chunk) {
  TORCH_CHECK(chunk >= 4096 && chunk <= (1U << 20) && chunk % 256 == 0, "xpool timeline invalid chunk geometry");
  TORCH_CHECK(bytes >= kControlBytes + 2 * chunk, "xpool timeline buffer lacks control and two chunks");
  return {.base = base,
          .chunks = std::min(kMaximumChunks, (bytes - kControlBytes) / chunk),
          .chunk_bytes = chunk,
          .capacity = (chunk - kHeaderBytes) / (sizeof(Record) + sizeof(std::uint64_t))};
}

XPOOL_KERNEL_FN void snapshot(PoolView source, Snapshot *output, bool stop, bool seal) {
  const auto index = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < source.chunks) {
    if (seal || stop)
      source.seal(index);
    output->ready_chunks[index] = source.acquire(index);
  }
  if (index == 0) {
    if (stop)
      Atomic{source.counters().stopped}.store(1);
    output->counters = source.counters().snapshot();
    auto pending = std::uint64_t{0};
    for (auto chunk = std::size_t{0}; chunk < source.chunks; ++chunk)
      pending += Atomic{source.control(chunk).state}.load() & kCountMask;
    output->pending_records = pending;
  }
}

XPOOL_KERNEL_FN void recycle(PoolView source, std::size_t index, std::uint64_t state, RecycleProgress *progress) {
  if (threadIdx.x == 0) {
    // Relaxed system stores expose progress without adding release fences to the
    // reclaim path. They are diagnostic observations, not a completion protocol.
    if (progress)
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>{progress->entered}.store(1,
                                                                                          cuda::memory_order_relaxed);
    const auto reclaimed = source.reclaim(index, state);
    if (progress) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>{progress->result}.store(reclaimed ? 1 : 2,
                                                                                         cuda::memory_order_relaxed);
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>{progress->exited}.store(1, cuda::memory_order_relaxed);
    }
  }
}

} // namespace

XPOOL_DEVICE_CONST PoolView device_view{};

std::size_t allocation_bytes(const Options &options, bool device) {
  if (!options.enable)
    return 0;
  // Leave a bounded 64 KiB allowance for optional IPC endpoint sidebands.
  const auto budget = device ? options.device_buffer_bytes : options.host_buffer_bytes;
  const auto reserved = device ? kControlBytes : std::max(kControlBytes, std::min(std::size_t{4U << 20}, budget / 4));
  const auto bytes = budget - std::min(budget, reserved);
  const auto view = geometry(nullptr, bytes, options.chunk_bytes);
  return kControlBytes + view.chunks * options.chunk_bytes;
}

Recorder::Recorder(const Options &options, int device)
    : options_(options), device_(device), source_bytes_(allocation_bytes(options, device >= 0)),
      host_bytes_(allocation_bytes(options, false)) {
  TORCH_CHECK(options.enable, "xpool timeline cannot create a disabled Recorder");
  if (device < 0) {
    host_storage_.resize(source_bytes_ / sizeof(std::uint64_t));
    source_ = geometry(reinterpret_cast<std::uint8_t *>(host_storage_.data()), source_bytes_, options.chunk_bytes);
    std::construct_at(&source_.metadata());
    for (auto i = std::size_t{0}; i < source_.chunks; ++i) {
      source_.control(i) = {};
      for (auto slot = std::size_t{0}; slot < source_.capacity; ++slot)
        source_.commits(i)[slot] = 4;
    }
    held_states_.resize(source_.chunks);
    return;
  }
  const auto guard = c10::cuda::CUDAGuard(device);
  std::uint8_t *allocation = nullptr;
  try {
    C10_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&allocation), source_bytes_));
    source_ = geometry(allocation, source_bytes_, options.chunk_bytes);
    C10_CUDA_CHECK(cudaMemset(allocation, 0, source_bytes_));
    const auto control = PoolControl{};
    C10_CUDA_CHECK(cudaMemcpy(allocation, &control, sizeof(control), cudaMemcpyHostToDevice));
    const auto commits = std::vector<std::uint64_t>(source_.capacity, 4);
    for (auto index = std::size_t{0}; index < source_.chunks; ++index)
      C10_CUDA_CHECK(cudaMemcpy(source_.commits(index), commits.data(), commits.size() * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice));
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&receipt_), host_bytes_,
                                 options_.diagnostics ? cudaHostAllocMapped : cudaHostAllocDefault));
    std::memset(receipt_, 0, kControlBytes);
    snapshot_host_ = std::construct_at(reinterpret_cast<Snapshot *>(receipt_));
    snapshot_device_ = &source_.metadata().snapshot;
    if (options_.diagnostics) {
      recycle_host_ = std::construct_at(reinterpret_cast<RecycleProgress *>(receipt_ + sizeof(Snapshot)));
      C10_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&recycle_device_), recycle_host_, 0));
    }
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    held_states_.resize((host_bytes_ - kControlBytes) / options.chunk_bytes);
  } catch (...) {
    const auto report = [](cudaError_t result, const char *operation) {
      if (result != cudaSuccess)
        std::cerr << "xpool timeline constructor cleanup failed operation=" << operation
                  << " detail=" << cudaGetErrorString(result) << '\n';
    };
    if (event_)
      report(cudaEventDestroy(event_), "event_destroy");
    if (stream_)
      report(cudaStreamDestroy(stream_), "stream_destroy");
    if (receipt_)
      report(cudaFreeHost(receipt_), "free_host");
    if (allocation)
      report(cudaFree(allocation), "free_device");
    throw;
  }
}

Recorder::~Recorder() {
  if (closed_ || device_ < 0)
    return;
  // Production explicitly closes only after drain. A retained timed-out owner
  // must not synchronize/free CUDA resources from a destructor during teardown.
  std::cerr << "xpool timeline retaining unclosed device storage device=" << device_ << '\n';
}

bool Recorder::record(const Record &record) {
  TORCH_CHECK(device_ < 0, "xpool timeline Host submit requires a Host Producer");
  auto value = record;
  if (!value.timestamp)
    value.timestamp = now();
  if (!value.site)
    value.site = static_cast<std::uint64_t>(gettid());
  return source_.commit(source_.reserve(), value);
}

void Recorder::stop() {
  const auto diagnostic = DiagnosticOperation(options_.diagnostics, device_, this, "stop");
  const auto lock =
      diagnose(options_.diagnostics, device_, this, "stop_lock", [&] { return std::unique_lock{mutex_}; });
  stopped_ = true;
  if (device_ < 0)
    Atomic{source_.counters().stopped}.store(1);
}

std::shared_ptr<Chunk> Recorder::collect(bool seal) {
  const auto diagnostic = DiagnosticOperation(options_.diagnostics, device_, this, "collect");
  const auto lock =
      diagnose(options_.diagnostics, device_, this, "collect_lock", [&] { return std::unique_lock{mutex_}; });
  TORCH_CHECK(!closed_, "xpool timeline Recorder is closed");
  TORCH_CHECK(chunk_sequence_ != std::numeric_limits<std::uint64_t>::max(), "xpool timeline Chunk sequence exhausted");
  if (device_ < 0) {
    if (Atomic{source_.counters().exhausted}.load())
      return {};
    const auto begin = now();
    for (auto index = std::size_t{0}; index < source_.chunks; ++index) {
      if (held_states_[index])
        continue;
      if (seal || stopped_)
        source_.seal(index);
      const auto state = source_.acquire(index);
      if (!state)
        continue;
      held_states_[index] = state;
      return std::make_shared<Chunk>(shared_from_this(), index, state & kCountMask, ++chunk_sequence_, state >> 32,
                                     begin, now());
    }
    return {};
  }
  if (latest_.exhausted)
    return {};
  const auto guard =
      diagnose(options_.diagnostics, device_, this, "device_guard", [&] { return c10::cuda::CUDAGuard(device_); });
  if (phase_ != Phase::Idle) {
    const auto result =
        diagnose(options_.diagnostics, device_, this, "event_query", [&] { return query_receipt(event_); });
    report_receipt(result);
    if (result == cudaErrorNotReady)
      return {};
    C10_CUDA_CHECK(result);
  }
  if (phase_ == Phase::Copy) {
    // Receipt is complete. Schedule reclaim only now; the Host lease remains
    // held independently while its Writer publishes the file.
    held_states_[copying_host_] = copying_state_;
    if (recycle_host_) {
      std::atomic_ref{recycle_host_->entered}.store(0, std::memory_order_relaxed);
      std::atomic_ref{recycle_host_->exited}.store(0, std::memory_order_relaxed);
      std::atomic_ref{recycle_host_->result}.store(0, std::memory_order_relaxed);
    }
    diagnose(options_.diagnostics, device_, this, "recycle_launch", [&] {
      recycle<<<1, 1, 0, stream_>>>(source_, copying_source_, copying_state_, recycle_device_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    diagnose(options_.diagnostics, device_, this, "event_record",
             [&] { C10_CUDA_CHECK(cudaEventRecord(event_, stream_)); });
    phase_ = Phase::Reclaim;
    return std::make_shared<Chunk>(shared_from_this(), copying_host_, copying_state_ & kCountMask, ++chunk_sequence_,
                                   copying_state_ >> 32, collection_begin_, now());
  }
  if (phase_ == Phase::Snapshot) {
    latest_ = snapshot_host_->counters;
    device_stopped_ = latest_.stopped != 0;
    const auto free = std::find(held_states_.begin(), held_states_.end(), 0);
    if (free != held_states_.end()) {
      for (auto index = std::size_t{0}; index < source_.chunks; ++index) {
        const auto state = snapshot_host_->ready_chunks[index];
        if (!state)
          continue;
        copying_source_ = index;
        copying_host_ = static_cast<std::size_t>(free - held_states_.begin());
        copying_state_ = state;
        // A pending receipt is charged as held before starting the transfer.
        held_states_[copying_host_] = state;
        diagnose(options_.diagnostics, device_, this, "records_copy", [&] {
          C10_CUDA_CHECK(cudaMemcpyAsync(receipt_ + kControlBytes + copying_host_ * options_.chunk_bytes + kHeaderBytes,
                                         source_.records(index), (state & kCountMask) * sizeof(Record),
                                         cudaMemcpyDeviceToHost, stream_));
        });
        diagnose(options_.diagnostics, device_, this, "event_record",
                 [&] { C10_CUDA_CHECK(cudaEventRecord(event_, stream_)); });
        phase_ = Phase::Copy;
        return {};
      }
    }
  }
  phase_ = Phase::Idle;
  collection_begin_ = now();
  diagnose(options_.diagnostics, device_, this, "snapshot_launch", [&] {
    snapshot<<<1, static_cast<unsigned>(kMaximumChunks), 0, stream_>>>(source_, snapshot_device_, stopped_, seal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  diagnose(options_.diagnostics, device_, this, "snapshot_copy", [&] {
    C10_CUDA_CHECK(
        cudaMemcpyAsync(snapshot_host_, snapshot_device_, sizeof(Snapshot), cudaMemcpyDeviceToHost, stream_));
  });
  diagnose(options_.diagnostics, device_, this, "event_record",
           [&] { C10_CUDA_CHECK(cudaEventRecord(event_, stream_)); });
  phase_ = Phase::Snapshot;
  return {};
}

void Recorder::report_receipt(cudaError_t result) {
  if (!options_.diagnostics)
    return;
  ++receipt_polls_;
  const auto observed = now();
  auto progress = RecycleProgress{};
  if (phase_ == Phase::Reclaim && recycle_host_) {
    progress.entered = std::atomic_ref{recycle_host_->entered}.load(std::memory_order_relaxed);
    progress.exited = std::atomic_ref{recycle_host_->exited}.load(std::memory_order_relaxed);
    progress.result = std::atomic_ref{recycle_host_->result}.load(std::memory_order_relaxed);
  }
  if (phase_ == reported_phase_ && result == reported_result_ && progress == reported_progress_ &&
      observed - receipt_report_ns_ < 5000000000ULL)
    return;
  const char *phase = "idle";
  switch (phase_) {
  case Phase::Idle:
    break;
  case Phase::Snapshot:
    phase = "snapshot";
    break;
  case Phase::Copy:
    phase = "copy";
    break;
  case Phase::Reclaim:
    phase = "reclaim";
    break;
  }
  std::fprintf(stderr,
               "xpool timeline receipt pid=%d tid=%d device=%d recorder=%p phase=%s result=%d "
               "source_slot=%zu epoch=%llu chunk_sequence=%llu recycle_entered=%u recycle_exited=%u "
               "recycle_result=%u polls=%llu observed_ns=%llu\n",
               getpid(), gettid(), device_, static_cast<const void *>(this), phase, static_cast<int>(result),
               copying_source_, static_cast<unsigned long long>(copying_state_ >> 32),
               static_cast<unsigned long long>(chunk_sequence_), progress.entered, progress.exited, progress.result,
               static_cast<unsigned long long>(receipt_polls_), static_cast<unsigned long long>(observed));
  std::fflush(stderr);
  reported_phase_ = phase_;
  reported_result_ = result;
  reported_progress_ = progress;
  receipt_report_ns_ = observed;
  receipt_polls_ = 0;
}

Counters Recorder::counters() const {
  const auto diagnostic = DiagnosticOperation(options_.diagnostics, device_, this, "counters");
  const auto lock =
      diagnose(options_.diagnostics, device_, this, "counters_lock", [&] { return std::unique_lock{mutex_}; });
  if (device_ >= 0)
    return latest_;
  return source_.counters().snapshot();
}

bool Recorder::drained() {
  const auto diagnostic = DiagnosticOperation(options_.diagnostics, device_, this, "drained");
  const auto lock =
      diagnose(options_.diagnostics, device_, this, "drained_lock", [&] { return std::unique_lock{mutex_}; });
  if (!stopped_ || std::any_of(held_states_.begin(), held_states_.end(), [](auto value) { return value != 0; }))
    return false;
  if (device_ < 0) {
    for (auto i = std::size_t{0}; i < source_.chunks; ++i)
      if (Atomic{source_.control(i).state}.load() & kCountMask)
        return false;
    return true;
  }
  if (!device_stopped_ || phase_ != Phase::Snapshot)
    return false;
  const auto guard =
      diagnose(options_.diagnostics, device_, this, "device_guard", [&] { return c10::cuda::CUDAGuard(device_); });
  const auto result =
      diagnose(options_.diagnostics, device_, this, "event_query", [&] { return query_receipt(event_); });
  report_receipt(result);
  if (result == cudaErrorNotReady)
    return false;
  C10_CUDA_CHECK(result);
  if (snapshot_host_->pending_records)
    return false;
  for (auto i = std::size_t{0}; i < source_.chunks; ++i)
    if (snapshot_host_->ready_chunks[i])
      return false;
  return latest_.attempted == latest_.committed + latest_.dropped;
}

void Recorder::close() {
  TORCH_CHECK(drained(), "xpool timeline cannot close live reservations, transfers or leases");
  const auto lock = std::lock_guard{mutex_};
  if (device_ >= 0) {
    const auto guard = c10::cuda::CUDAGuard(device_);
    C10_CUDA_CHECK(cudaEventDestroy(event_));
    event_ = nullptr;
    C10_CUDA_CHECK(cudaStreamDestroy(stream_));
    stream_ = nullptr;
    if (device_owner.get() == this) {
      const auto empty = PoolView{};
      C10_CUDA_CHECK(cudaMemcpyToSymbol(device_view, &empty, sizeof(empty)));
    }
    C10_CUDA_CHECK(cudaFree(source_.base));
    source_.base = nullptr;
    C10_CUDA_CHECK(cudaFreeHost(receipt_));
    receipt_ = nullptr;
    recycle_host_ = nullptr;
    recycle_device_ = nullptr;
  }
  closed_ = true;
}

std::span<const Record> Recorder::lease_records(std::size_t slot, std::size_t count) const {
  TORCH_CHECK(slot < held_states_.size() && held_states_[slot], "xpool timeline Chunk lease is released");
  const auto *records =
      device_ < 0
          ? source_.records(slot)
          : reinterpret_cast<const Record *>(receipt_ + kControlBytes + slot * options_.chunk_bytes + kHeaderBytes);
  return {records, count};
}

void Recorder::release(std::size_t slot) {
  const auto diagnostic = DiagnosticOperation(options_.diagnostics, device_, this, "release");
  const auto lock =
      diagnose(options_.diagnostics, device_, this, "release_lock", [&] { return std::unique_lock{mutex_}; });
  TORCH_CHECK(slot < held_states_.size() && held_states_[slot], "xpool timeline Chunk lease is released");
  if (device_ < 0)
    TORCH_CHECK(source_.reclaim(slot, held_states_[slot]), "xpool timeline Host reclaim failed");
  held_states_[slot] = 0;
}

Chunk::Chunk(std::shared_ptr<Recorder> owner, std::size_t slot, std::size_t count, std::uint64_t sequence,
             std::uint64_t epoch, std::uint64_t begin, std::uint64_t end)
    : owner_(std::move(owner)), slot_(slot), count_(count), sequence_(sequence), epoch_(epoch), begin_(begin),
      end_(end) {}
Chunk::~Chunk() {
  if (released_)
    owner_->release(slot_);
}
std::span<const Record> Chunk::records() const {
  TORCH_CHECK(owner_ && !released_, "xpool timeline Chunk lease is released");
  return owner_->lease_records(slot_, count_);
}
void Chunk::release() {
  TORCH_CHECK(owner_ && !released_, "xpool timeline Chunk lease is released");
  released_ = true;
}

void configure(std::uint64_t host, std::uint64_t device, int index) {
  const auto lock = std::lock_guard{installation_mutex};
  TORCH_CHECK(!host_id || (host_id == host && device_id == device && installed_device == index),
              "xpool timeline creation identity changed");
  if (host_id)
    return;
  TORCH_CHECK(host != 0 && (index < 0 || device != 0), "xpool timeline creation identity is missing");
  host_id = host;
  device_id = device;
  installed_device = index;
  host_owner = std::make_shared<Recorder>(xpool::debug::options().timeline);
}
std::shared_ptr<Recorder> host_recorder() {
  const auto lock = std::lock_guard{installation_mutex};
  return host_owner;
}
std::shared_ptr<Recorder> device_recorder() {
  const auto lock = std::lock_guard{installation_mutex};
  return device_owner;
}
std::uint64_t device_identity() { return device_id; }
void install_device() {
  const auto lock = std::lock_guard{installation_mutex};
  if (device_owner)
    return;
  TORCH_CHECK(device_id != 0 && installed_device >= 0, "xpool timeline Device identity is missing");
  device_owner = std::make_shared<Recorder>(xpool::debug::options().timeline, installed_device);
  const auto view = device_owner->view();
  C10_CUDA_CHECK(cudaMemcpyToSymbol(device_view, &view, sizeof(view)));
}

} // namespace xpool::devkit::timeline
