#pragma once

/// \file xpool/devkit/capacity_counter.hpp
/// \brief Independent, bounded experiment histograms; not timeline records.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <xpool/macros.hpp>

namespace xpool::devkit::capacity_counter {

/// Closed histogram column vocabulary, independent of legacy storage capacity.
enum class Metric : std::size_t {
  Transport,
  AtnAgent,
  Coordinator,
  FfnAgent,
  Association,
  RoutingPublished,
  RoutingBytes,
  GraphActivityEstimate,
  Count
};
inline constexpr std::size_t kColumns = static_cast<std::size_t>(Metric::Count);
inline constexpr std::uint64_t kBucketNs = 10000000;
using Bucket = std::array<std::uint64_t, kColumns>;

/// Return an in-range bucket or count for timestamps outside coverage.
/// This never subtracts an earlier timestamp from a later origin.
XPOOL_HOST_DEVICE_FN constexpr std::size_t bucket_index(std::uint64_t origin, std::uint64_t now, std::size_t count) {
  if (now < origin) {
    return count;
  }
  const auto index = (now - origin) / kBucketNs;
  return index < count ? static_cast<std::size_t>(index) : count;
}

/// Host-owned histogram obtained only after every local producer is quiescent.
struct Snapshot {
  /// Device globaltimer origin, zero if no activity occurred.
  std::uint64_t origin_ns = 0;
  /// Last observed Device time, not a Host clock mapping.
  std::uint64_t last_ns = 0;
  /// Attempts outside histogram coverage; any nonzero value invalidates a full-window estimate.
  std::uint64_t outside_window = 0;
  /// Counter arithmetic overflow; any nonzero value invalidates the estimate.
  std::uint64_t arithmetic_overflow = 0;
  /// Actual experimental Device allocation, separate from timeline budgets.
  std::size_t allocation_bytes = 0;
  /// Ten-millisecond buckets, including empty buckets before the last observation.
  std::vector<Bucket> buckets;
};

/// Allocate the configured experiment before any capture or producer launch.
/// \pre The caller has selected the process Device and no producer is active.
void initialize();
/// Read and release experiment storage after all local Device producers retire.
/// \pre All local Streams and resident producers are quiescent; this function does not synchronize them.
/// \throws c10::Error on disabled/uninitialized state or CUDA failure.
Snapshot finish();

} // namespace xpool::devkit::capacity_counter
