#pragma once

/// \file xpool/devkit/timeline/pool.hpp
/// \brief Bounded immutable Timeline records and reservation ownership.

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda/atomic>

#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Immutable runtime fact with the version-one 128-byte little-endian wire layout.
/// Field order is part of the file contract, independent of pool ownership.
struct Record {
  /// Producer-owned attempted record sequence, assigned on commit.
  std::uint64_t sequence = 0;
  /// Raw nanoseconds in this Producer's declared clock domain.
  std::uint64_t timestamp = 0;
  /// Wire event kind; zero denotes an explicitly aborted reservation.
  std::uint64_t kind = 0;
  /// Host thread identity or Device protocol participant coordinate.
  std::uint64_t site = 0;
  /// High word of the immutable Fabric Generation identity.
  std::uint64_t generation_high = 0;
  /// Low word of the immutable Fabric Generation identity.
  std::uint64_t generation_low = 0;
  /// Device Producer identity of the shared Transport endpoint creator.
  std::uint64_t endpoint_creator = 0;
  /// Endpoint creation sequence within that creator.
  std::uint64_t endpoint_index = 0;
  /// Transport operation or Fabric Invocation sequence, scoped by event kind.
  std::uint64_t operation = 0;
  /// Configuration-order Instance index.
  std::uint64_t instance = 0;
  /// Canonical model FFN layer ordinal when applicable.
  std::uint64_t layer = 0;
  /// Executor Lane index when applicable.
  std::uint64_t lane = 0;
  /// Executor Lease sequence when applicable.
  std::uint64_t lease = 0;
  /// Production Result Code when applicable, independent of collection quality.
  std::uint64_t result = 0;
  /// Physical payload rows when applicable.
  std::uint64_t rows = 0;
  /// Reserved version-one wire word; must remain zero.
  std::uint64_t reserved = 0;
};

// Chunk buffer publication exposes this exact external wire format. Fail the
// build on a platform whose representation would add padding or reorder bytes.
static_assert(std::is_standard_layout_v<Record> && std::is_trivially_copyable_v<Record>);
static_assert(sizeof(Record) == 128);
static_assert(std::endian::native == std::endian::little);

/// Fixed Producer quality counters, independent of event capacity.
struct Counters {
  /// Acquire-load each field; concurrent writers do not form a global transaction.
  XPOOL_HOST_DEVICE_FN Counters snapshot();
  /// Attempt sequence, including reservations rejected before commit.
  std::uint64_t attempted = 0;
  /// Semantic records committed with release visibility.
  std::uint64_t committed = 0;
  /// Pre-commit drops, including explicit aborted reservations.
  std::uint64_t dropped = 0;
  /// Drops after bounded compare-exchange contention.
  std::uint64_t contention = 0;
  /// First observed dropped attempt; zero means none observed.
  std::uint64_t first_gap = 0;
  /// Coarsened upper gap observation; the Writer reports a conservative envelope.
  std::uint64_t last_gap = 0;
  /// Admission closed; previously reserved capabilities remain valid.
  std::uint64_t stopped = 0;
  /// A non-wrapping identity domain has exhausted its capacity.
  std::uint64_t exhausted = 0;
};

/// One chunk's atomic epoch, seal bit and reservation count.
struct ChunkState {
  /// Epoch in high 32 bits; seal and count in low 32 bits.
  std::uint64_t state = std::uint64_t{1} << 32;
};

/// Packed reservation control; no bit field access crosses an atomic seam.
inline constexpr std::uint64_t kSeal = std::uint64_t{1} << 31;
/// Reservation-count mask excluding the seal bit.
inline constexpr std::uint64_t kCountMask = kSeal - 1;
/// Pool control allocation in bytes, including asynchronous snapshot storage.
inline constexpr std::size_t kControlBytes = 65536;
/// Reserved Chunk wire-header bytes.
inline constexpr std::size_t kHeaderBytes = 256;
/// Maximum bounded source/receipt inventory.
inline constexpr std::size_t kMaximumChunks = 256;

/// Immutable control receipt copied after the sampling Kernel completes.
struct Snapshot {
  /// Acquired sealed source states, or zero when a Chunk is not ready.
  std::array<std::uint64_t, kMaximumChunks> ready_chunks{};
  /// Individually acquired Producer quality counters.
  Counters counters{};
  /// Reserved records not yet reclaimed, including already committed records.
  std::uint64_t pending_records = 0;
};

/// Source allocation control area; layout is private to the process-local pool.
struct PoolControl {
  /// Atomic Producer quality and admission state.
  Counters counters{};
  /// Independent atomic source Chunk reservation controls.
  std::array<ChunkState, kMaximumChunks> chunks{};
  /// Device sampling workspace; records occupy a separate allocation region.
  Snapshot snapshot{};
};

static_assert(std::is_trivially_copyable_v<Snapshot> && std::is_standard_layout_v<Snapshot>);
static_assert(sizeof(PoolControl) <= kControlBytes);
// Host backing storage uses uint64_t alignment; reject stronger control alignment.
static_assert(alignof(PoolControl) <= alignof(std::uint64_t));

/// Scope-appropriate atomics over the same plain declared storage.
template <typename T> struct Atomic {
  /// Naturally aligned shared word owned by the corresponding pool.
  T &value;
  /// Acquire-observe the current word.
  XPOOL_HOST_DEVICE_FN T load() const {
#if defined(__CUDA_ARCH__)
    return cuda::atomic_ref<T>{value}.load(cuda::memory_order_acquire);
#else
    return std::atomic_ref<T>{value}.load(std::memory_order_acquire);
#endif
  }
  /// Release-publish the new word.
  XPOOL_HOST_DEVICE_FN void store(T desired) const {
#if defined(__CUDA_ARCH__)
    cuda::atomic_ref<T>{value}.store(desired, cuda::memory_order_release);
#else
    std::atomic_ref<T>{value}.store(desired, std::memory_order_release);
#endif
  }
  /// Reserve a unique increment with relaxed atomic ordering.
  XPOOL_HOST_DEVICE_FN T add(T increment) const {
#if defined(__CUDA_ARCH__)
    return cuda::atomic_ref<T>{value}.fetch_add(increment, cuda::memory_order_relaxed);
#else
    return std::atomic_ref<T>{value}.fetch_add(increment, std::memory_order_relaxed);
#endif
  }
  /// Acquire-release claim; failed comparison acquire-observes the actual word.
  XPOOL_HOST_DEVICE_FN bool compare(T &expected, T desired) const {
#if defined(__CUDA_ARCH__)
    return cuda::atomic_ref<T>{value}.compare_exchange_strong(expected, desired, cuda::memory_order_acq_rel);
#else
    return std::atomic_ref<T>{value}.compare_exchange_strong(expected, desired, std::memory_order_acq_rel);
#endif
  }
};

XPOOL_HOST_DEVICE_FN inline Counters Counters::snapshot() {
  return {.attempted = Atomic{attempted}.load(),
          .committed = Atomic{committed}.load(),
          .dropped = Atomic{dropped}.load(),
          .contention = Atomic{contention}.load(),
          .first_gap = Atomic{first_gap}.load(),
          .last_gap = Atomic{last_gap}.load(),
          .stopped = Atomic{stopped}.load(),
          .exhausted = Atomic{exhausted}.load()};
}

/// Capability valid only for one reserved position in one epoch.
struct Ticket {
  /// Source Chunk ordinal.
  std::size_t chunk = 0;
  /// Reserved record position in that Chunk.
  std::size_t index = 0;
  /// Epoch under which the capability was created.
  std::uint64_t epoch = 0;
  /// Unique attempted record sequence.
  std::uint64_t sequence = 0;
  /// Whether reservation succeeded and grants a commit capability.
  bool valid = false;
};

/// Non-owning pool geometry used by Host and Device writers.
struct PoolView {
  /// Live backing allocation; null denotes disabled collection.
  std::uint8_t *base = nullptr;
  /// Number of source Chunks, bounded by kMaximumChunks.
  std::size_t chunks = 0;
  /// Stride including header allowance and commit markers.
  std::size_t chunk_bytes = 0;
  /// Fixed record/marker capacity per Chunk.
  std::size_t capacity = 0;

  /// Access the typed source control area while backing storage is live.
  XPOOL_HOST_DEVICE_FN PoolControl &metadata() const { return *reinterpret_cast<PoolControl *>(base); }
  /// Access fixed control counters at the pool origin.
  XPOOL_HOST_DEVICE_FN Counters &counters() const { return metadata().counters; }
  /// Access one source control; index must be within chunks.
  XPOOL_HOST_DEVICE_FN ChunkState &control(std::size_t index) const { return metadata().chunks[index]; }
  /// Access source record storage while protected by its reservation/lease.
  XPOOL_HOST_DEVICE_FN Record *records(std::size_t index) const {
    return reinterpret_cast<Record *>(base + kControlBytes + index * chunk_bytes + kHeaderBytes);
  }
  /// Access epoch-tagged terminal commit markers for one source Chunk.
  XPOOL_HOST_DEVICE_FN std::uint64_t *commits(std::size_t index) const {
    return reinterpret_cast<std::uint64_t *>(records(index) + capacity);
  }
  /// Count an explicit drop without consuming event capacity.
  XPOOL_HOST_DEVICE_FN void loss(std::uint64_t sequence, bool contention) const {
    auto &counts = counters();
    Atomic{counts.dropped}.add(std::uint64_t{1});
    if (contention)
      Atomic{counts.contention}.add(std::uint64_t{1});
    auto zero = std::uint64_t{0};
    Atomic{counts.first_gap}.compare(zero, sequence);
    // Concurrent updates may arrive out of order. This is an envelope, not a
    // precise interval list; the offline reader also checks actual sequences.
    auto previous = Atomic{counts.last_gap}.load();
    for (auto retry = 0; retry < 16 && previous < sequence; ++retry)
      if (Atomic{counts.last_gap}.compare(previous, sequence))
        break;
  }
  /// Reserve with bounded work; a sealed chunk never grants a new capability.
  XPOOL_HOST_DEVICE_FN Ticket reserve() const {
    if (base == nullptr || Atomic{counters().stopped}.load())
      return {};
    const auto sequence = Atomic{counters().attempted}.add(std::uint64_t{1}) + 1;
    // Stop halfway through the unsigned domain, leaving headroom larger than
    // physically possible in-flight Host/Device callers before the stop is seen.
    if (sequence >= std::uint64_t{1} << 63) {
      loss(sequence, false);
      Atomic{counters().exhausted}.store(1);
      Atomic{counters().stopped}.store(1);
      return {};
    }
    bool raced = false;
    auto attempts = std::size_t{0};
    // Geometry bounds scanning to 256 controls; failed CAS attempts are capped
    // separately, so sealed neighbours cannot hide an available distant chunk.
    for (auto probe = std::size_t{0}; probe < chunks && attempts < 16; ++probe) {
      const auto index = (sequence + probe) % chunks;
      auto state = Atomic{control(index).state}.load();
      if ((state & kSeal) || (state & kCountMask) >= capacity)
        continue;
      ++attempts;
      const auto desired = (state + 1) | (((state & kCountMask) + 1 == capacity) ? kSeal : 0);
      if (Atomic{control(index).state}.compare(state, desired))
        return {.chunk = index,
                .index = static_cast<std::size_t>(state & kCountMask),
                .epoch = state >> 32,
                .sequence = sequence,
                .valid = true};
      raced = true;
    }
    loss(sequence, raced);
    return {};
  }
  /// Publish or abort one capability; stale tickets never access record storage.
  XPOOL_HOST_DEVICE_FN bool commit(const Ticket &ticket, Record record, bool aborted = false) const {
    if (!ticket.valid || ticket.chunk >= chunks || ticket.index >= capacity)
      return false;
    const auto state = Atomic{control(ticket.chunk).state}.load();
    if ((state >> 32) != ticket.epoch || ticket.index >= (state & kCountMask))
      return false;
    auto pending = ticket.epoch << 2;
    // Epoch-tagged claim prevents stale/double commits across recycling. Acquire treats this intermediate marker
    // as incomplete, so storage cannot recycle until the release below.
    if (!Atomic{commits(ticket.chunk)[ticket.index]}.compare(pending, (ticket.epoch << 2) | 3))
      return false;
    record.sequence = ticket.sequence;
    if (aborted) {
      record.kind = 0;
      loss(ticket.sequence, false);
    }
    records(ticket.chunk)[ticket.index] = record;
    if (!aborted)
      Atomic{counters().committed}.add(std::uint64_t{1});
    Atomic{commits(ticket.chunk)[ticket.index]}.store((ticket.epoch << 2) | (aborted ? 2 : 1));
    return true;
  }
  /// Seal nonempty chunks; already qualified writers may finish their commits.
  XPOOL_HOST_DEVICE_FN std::uint64_t seal(std::size_t index) const {
    auto value = Atomic{control(index).state}.load();
    for (auto retry = 0; retry < 16; ++retry) {
      if ((value & kSeal) || !(value & kCountMask))
        return value;
      if (Atomic{control(index).state}.compare(value, value | kSeal))
        return value | kSeal;
    }
    return 0;
  }
  /// Acquire a fully committed sealed chunk; zero means not ready.
  XPOOL_HOST_DEVICE_FN std::uint64_t acquire(std::size_t index) const {
    const auto value = Atomic{control(index).state}.load();
    if (!(value & kSeal))
      return 0;
    for (auto slot = std::size_t{0}; slot < (value & kCountMask); ++slot) {
      const auto commit = Atomic{commits(index)[slot]}.load();
      if ((commit >> 2) != (value >> 32) || ((commit & 3) != 1 && (commit & 3) != 2))
        return 0;
    }
    return value;
  }
  /// Recycle only after receipt/publication as required by the owning pool.
  XPOOL_HOST_DEVICE_FN bool reclaim(std::size_t index, std::uint64_t state) const {
    if (acquire(index) != state)
      return false;
    const auto epoch = state >> 32;
    if (epoch == std::numeric_limits<std::uint32_t>::max()) {
      Atomic{counters().exhausted}.store(1);
      Atomic{counters().stopped}.store(1);
      return false;
    }
    for (auto slot = std::size_t{0}; slot < capacity; ++slot)
      Atomic{commits(index)[slot]}.store((epoch + 1) << 2);
    Atomic{control(index).state}.store((epoch + 1) << 32);
    return true;
  }
};

} // namespace xpool::devkit::timeline
