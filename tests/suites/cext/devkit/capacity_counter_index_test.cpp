#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <xpool/devkit/capacity_counter.hpp>

using xpool::devkit::capacity_counter::bucket_index;
using xpool::devkit::capacity_counter::kBucketNs;

TEST(CapacityCounterIndex, ExactBoundaryBelongsToNextBucket) {
  constexpr std::uint64_t origin = 123;
  EXPECT_EQ(bucket_index(origin, origin, 2), 0);
  EXPECT_EQ(bucket_index(origin, origin + kBucketNs - 1, 2), 0);
  EXPECT_EQ(bucket_index(origin, origin + kBucketNs, 2), 1);
  EXPECT_EQ(bucket_index(origin, origin + 2 * kBucketNs, 2), 2);
}

TEST(CapacityCounterIndex, UnavailableCoverageCannotBecomeAnInRangeSlot) {
  EXPECT_EQ(bucket_index(100, 99, 4), 4);
  EXPECT_EQ(bucket_index(0, std::numeric_limits<std::uint64_t>::max(), 4), 4);
  EXPECT_EQ(bucket_index(100, 100, 0), 0);
}
