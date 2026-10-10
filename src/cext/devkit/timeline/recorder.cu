#include <xpool/devkit/timeline/recorder.hpp>

#include <algorithm>
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
#include <xpool/utils/checked.hpp>

namespace xpool::devkit::timeline {

namespace {

std::shared_ptr<Recorder> host_owner;
std::shared_ptr<Recorder> device_owner;
std::uint64_t host_id = 0;
std::uint64_t device_id = 0;
int installed_device = -1;
std::mutex installation_mutex;

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

XPOOL_KERNEL_FN void recycle(PoolView source, std::size_t index, std::uint64_t state) {
  if (threadIdx.x == 0)
    source.reclaim(index, state);
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
    C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&receipt_), host_bytes_, cudaHostAllocDefault));
    std::memset(receipt_, 0, kControlBytes);
    snapshot_host_ = std::construct_at(reinterpret_cast<Snapshot *>(receipt_));
    snapshot_device_ = &source_.metadata().snapshot;
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
  const auto lock = std::lock_guard{mutex_};
  stopped_ = true;
  if (device_ < 0)
    Atomic{source_.counters().stopped}.store(1);
}

std::shared_ptr<Chunk> Recorder::collect(bool seal) {
  const auto lock = std::lock_guard{mutex_};
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
  const auto guard = c10::cuda::CUDAGuard(device_);
  if (phase_ != Phase::Idle) {
    const auto result = cudaEventQuery(event_);
    if (result == cudaErrorNotReady)
      return {};
    C10_CUDA_CHECK(result);
  }
  if (phase_ == Phase::Copy) {
    // Receipt is complete. Schedule reclaim only now; the Host lease remains
    // held independently while its Writer publishes the file.
    held_states_[copying_host_] = copying_state_;
    recycle<<<1, 1, 0, stream_>>>(source_, copying_source_, copying_state_);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(event_, stream_));
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
        C10_CUDA_CHECK(cudaMemcpyAsync(receipt_ + kControlBytes + copying_host_ * options_.chunk_bytes + kHeaderBytes,
                                       source_.records(index), (state & kCountMask) * sizeof(Record),
                                       cudaMemcpyDeviceToHost, stream_));
        C10_CUDA_CHECK(cudaEventRecord(event_, stream_));
        phase_ = Phase::Copy;
        return {};
      }
    }
  }
  phase_ = Phase::Idle;
  collection_begin_ = now();
  snapshot<<<1, static_cast<unsigned>(kMaximumChunks), 0, stream_>>>(source_, snapshot_device_, stopped_, seal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaMemcpyAsync(snapshot_host_, snapshot_device_, sizeof(Snapshot), cudaMemcpyDeviceToHost, stream_));
  C10_CUDA_CHECK(cudaEventRecord(event_, stream_));
  phase_ = Phase::Snapshot;
  return {};
}

Counters Recorder::counters() const {
  const auto lock = std::lock_guard{mutex_};
  if (device_ >= 0)
    return latest_;
  return source_.counters().snapshot();
}

bool Recorder::drained() {
  const auto lock = std::lock_guard{mutex_};
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
  const auto guard = c10::cuda::CUDAGuard(device_);
  const auto result = cudaEventQuery(event_);
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
  const auto lock = std::lock_guard{mutex_};
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
