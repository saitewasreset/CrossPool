#pragma once

/// \file xpool/devkit/timeline/recorder.hpp
/// \brief Native Producer ownership and immutable Chunk leases.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <cuda_runtime_api.h>

#include <xpool/devkit/timeline/pool.hpp>

namespace xpool::devkit::timeline {

/// Immutable pool limits installed before native initialization.
struct Options {
  /// Enable independent recording; disabled sizing returns zero.
  bool enable = false;
  /// Device source bytes including commit/control and IPC allowance.
  std::size_t device_buffer_bytes = 8U << 20;
  /// Host source or pinned receipt budget including metadata allowance.
  std::size_t host_buffer_bytes = 32U << 20;
  /// Maximum published Chunk bytes; multiple of 256, from 4096 to 1 MiB.
  std::size_t chunk_bytes = 1U << 20;
  constexpr bool operator==(const Options &) const = default;
};

/// Calculate actual bounded allocation, including control and commit storage.
/// \throws c10::Error for invalid geometry.
std::size_t allocation_bytes(const Options &options, bool device);

class Recorder;

/// Move-independent lease holding its Producer's Host storage alive.
class Chunk {
public:
  /// Construct an owning read lease; only the Recorder creates valid leases.
  Chunk(std::shared_ptr<Recorder> owner, std::size_t slot, std::size_t count, std::uint64_t sequence,
        std::uint64_t epoch, std::uint64_t begin, std::uint64_t end);
  /// Retire a released lease only after its final buffer view is destroyed.
  ~Chunk();
  /// Read immutable named records until release; data access after release fails.
  std::span<const Record> records() const;
  /// Explicitly release after publication or discard. Double release fails.
  void release();
  /// Return monotonic Chunk identity.
  std::uint64_t sequence() const { return sequence_; }
  /// Return source epoch.
  std::uint64_t epoch() const { return epoch_; }
  /// Return Host collection window start in CLOCK_MONOTONIC nanoseconds.
  std::uint64_t begin_ns() const { return begin_; }
  /// Return Host receipt completion observation in the same Host clock.
  std::uint64_t end_ns() const { return end_; }

private:
  bool released_ = false;
  std::shared_ptr<Recorder> owner_;
  std::size_t slot_;
  std::size_t count_;
  std::uint64_t sequence_;
  std::uint64_t epoch_;
  std::uint64_t begin_;
  std::uint64_t end_;
};

/// One process-local Host or Device Producer and its bounded receipt pool.
class Recorder : public std::enable_shared_from_this<Recorder> {
public:
  /// Allocate bounded storage. Device=-1 is Host-only and performs no CUDA calls.
  Recorder(const Options &options, int device = -1);
  /// Cleanup only when collection has stopped; timed-out CUDA owners are retained.
  ~Recorder();
  /// Submit a Host record without collector/file waits; return accepted status.
  bool record(const Record &record);
  /// Close admission. Existing reservations retain their right to commit.
  void stop();
  /// Seal/acquire and asynchronously collect one ready chunk, or return no lease.
  /// Device calls never wait for an event; repeated polling drives receipt/reclaim.
  /// Receipt queries support concurrent Global capture in another Host thread
  /// and restore the calling thread's capture interaction mode.
  std::shared_ptr<Chunk> collect(bool seal = true);
  /// Return the latest bounded quality snapshot; device counters reflect poll time.
  Counters counters() const;
  /// Return whether no reservations/transfers/leases remain after stop.
  /// Device receipt queries use the same capture isolation as collect().
  bool drained();
  /// Retire resources after verified drain; reject live ownership.
  void close();
  /// Return source pool geometry for native Hook dispatch.
  PoolView view() const { return source_; }
  /// Return owning device, or -1 for Host.
  int device() const { return device_; }
  /// Return effective source allocation bytes.
  std::size_t source_bytes() const { return source_bytes_; }
  /// Return actual Host-owned allocation bytes including receipt control.
  std::size_t host_bytes() const { return host_bytes_; }
  /// Lease implementation: access source or receipt records while held.
  std::span<const Record> lease_records(std::size_t slot, std::size_t count) const;
  /// Lease implementation: release a held slot at the appropriate recycle seam.
  void release(std::size_t slot);

private:
  enum class Phase { Idle, Snapshot, Copy, Reclaim };
  Options options_;
  int device_;
  std::size_t source_bytes_;
  std::size_t host_bytes_;
  PoolView source_{};
  std::vector<std::uint64_t> host_storage_;
  std::uint8_t *receipt_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
  /// Typed sampling workspace inside the Device source control allocation.
  Snapshot *snapshot_device_ = nullptr;
  /// Pinned receipt read only after the sampling transfer event completes.
  Snapshot *snapshot_host_ = nullptr;
  // Per Host slot: zero is free; otherwise retain the acquired source state.
  std::vector<std::uint64_t> held_states_;
  mutable std::mutex mutex_;
  Counters latest_{};
  Phase phase_ = Phase::Idle;
  std::size_t copying_source_ = 0;
  std::size_t copying_host_ = 0;
  std::uint64_t copying_state_ = 0;
  std::uint64_t collection_begin_ = 0;
  std::uint64_t chunk_sequence_ = 0;
  bool stopped_ = false;
  bool device_stopped_ = false;
  bool closed_ = false;
};

/// Install immutable generation identity before production execution.
void set_generation(std::uint64_t high, std::uint64_t low, std::uint64_t pe);
/// Install process creation/Device identity before lifecycle hooks; changes fail.
void configure(std::uint64_t host_id, std::uint64_t device_id, int device);
/// Return the installed Host Producer, or no owner when collection is disabled.
std::shared_ptr<Recorder> host_recorder();
/// Return the Device Producer once installed by an endpoint/Fabric lifecycle hook.
std::shared_ptr<Recorder> device_recorder();
/// Install Device pool before Resident/Graph execution; requires a creation identity.
void install_device();
/// Return the configured Device Producer identity for IPC endpoint creation.
std::uint64_t device_identity();

} // namespace xpool::devkit::timeline
