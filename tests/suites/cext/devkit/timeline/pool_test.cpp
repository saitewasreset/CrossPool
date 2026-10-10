#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <xpool/devkit/timeline/pool.hpp>

namespace {
using xpool::devkit::timeline::PoolView;
using xpool::devkit::timeline::Record;

class TimelinePool : public ::testing::Test {
protected:
  std::vector<std::uint64_t> storage = std::vector<std::uint64_t>((65536 + 2 * 4096) / 8);
  PoolView view{reinterpret_cast<std::uint8_t *>(storage.data()), 2, 4096, 16};
  void SetUp() override {
    std::construct_at(&view.metadata());
    for (auto chunk = std::size_t{0}; chunk < 2; ++chunk) {
      view.control(chunk) = {};
      for (auto slot = std::size_t{0}; slot < view.capacity; ++slot)
        view.commits(chunk)[slot] = 4;
    }
  }
};

TEST_F(TimelinePool, SealWaitsForOutOfOrderCommitAndRejectsStaleEpoch) {
  auto first = view.reserve();
  auto second = view.reserve();
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(second.valid);
  const auto sealed = view.seal(first.chunk);
  EXPECT_EQ(view.acquire(first.chunk), 0);
  EXPECT_TRUE(view.commit(second, Record{}));
  EXPECT_EQ(view.acquire(first.chunk), 0);
  EXPECT_TRUE(view.commit(first, Record{}));
  EXPECT_EQ(view.acquire(first.chunk), sealed);
  EXPECT_TRUE(view.reclaim(first.chunk, sealed));
  EXPECT_FALSE(view.commit(first, Record{}));
  EXPECT_NE(view.control(first.chunk).state >> 32, first.epoch);
}

TEST_F(TimelinePool, PartialReclaimInitializesUnusedSlotsInTheNextEpoch) {
  const auto first = view.reserve();
  ASSERT_TRUE(view.commit(first, Record{}));
  const auto sealed = view.seal(first.chunk);
  ASSERT_TRUE(view.reclaim(first.chunk, sealed));
  for (auto attempt = 0; attempt < 16; ++attempt) {
    const auto ticket = view.reserve();
    ASSERT_TRUE(ticket.valid);
    ASSERT_TRUE(view.commit(ticket, Record{}));
  }
  EXPECT_EQ(view.counters().dropped, 0);
}

TEST_F(TimelinePool, AbortedReservationIsTerminalWithoutSemanticCommit) {
  const auto ticket = view.reserve();
  view.seal(ticket.chunk);
  ASSERT_TRUE(view.commit(ticket, Record{}, true));
  EXPECT_NE(view.acquire(ticket.chunk), 0);
  EXPECT_EQ(view.counters().committed, 0);
  EXPECT_EQ(view.counters().dropped, 1);
  EXPECT_FALSE(view.commit(ticket, Record{}));
}

TEST_F(TimelinePool, ConcurrentWritersNeverOverwriteAndLossIsCounted) {
  std::vector<std::thread> workers;
  for (auto thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      for (auto attempt = 0; attempt < 100; ++attempt) {
        auto record = Record{};
        record.kind = 42;
        view.commit(view.reserve(), record);
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  EXPECT_EQ(view.counters().attempted, 800);
  EXPECT_EQ(view.counters().committed + view.counters().dropped, 800);
  EXPECT_EQ(view.counters().committed, 32);
  for (auto chunk = std::size_t{0}; chunk < 2; ++chunk) {
    EXPECT_NE(view.acquire(chunk), 0);
    for (auto slot = std::size_t{0}; slot < 16; ++slot)
      EXPECT_EQ(view.records(chunk)[slot].kind, 42);
  }
}

TEST_F(TimelinePool, EpochExhaustionStopsInsteadOfWrapping) {
  view.control(1).state = std::uint64_t{0xffffffff} << 32;
  for (auto slot = std::size_t{0}; slot < view.capacity; ++slot)
    view.commits(1)[slot] = std::uint64_t{0xffffffff} << 2;
  const auto ticket = view.reserve();
  ASSERT_EQ(ticket.chunk, 1);
  ASSERT_TRUE(view.commit(ticket, Record{}));
  const auto sealed = view.seal(1);
  EXPECT_FALSE(view.reclaim(1, sealed));
  EXPECT_EQ(view.counters().exhausted, 1);
  EXPECT_FALSE(view.reserve().valid);
}

TEST_F(TimelinePool, SealedNeighboursDoNotHideAnAvailableDistantChunk) {
  storage.resize((65536 + 32 * 4096) / 8);
  view = {reinterpret_cast<std::uint8_t *>(storage.data()), 32, 4096, 16};
  std::construct_at(&view.metadata());
  for (auto chunk = std::size_t{0}; chunk < view.chunks; ++chunk) {
    view.control(chunk).state = (std::uint64_t{1} << 32) | xpool::devkit::timeline::kSeal;
    for (auto slot = std::size_t{0}; slot < view.capacity; ++slot)
      view.commits(chunk)[slot] = 4;
  }
  view.control(25) = {};
  const auto ticket = view.reserve();
  ASSERT_TRUE(ticket.valid);
  EXPECT_EQ(ticket.chunk, 25);
  EXPECT_TRUE(view.commit(ticket, Record{}));
  EXPECT_EQ(view.counters().dropped, 0);
}

TEST_F(TimelinePool, AttemptSequenceExhaustionKeepsQualityCountersBalanced) {
  view.counters().attempted = (std::uint64_t{1} << 63) - 1;
  view.counters().committed = view.counters().attempted;
  EXPECT_FALSE(view.reserve().valid);
  EXPECT_EQ(view.counters().attempted, view.counters().committed + view.counters().dropped);
  EXPECT_EQ(view.counters().dropped, 1);
  EXPECT_EQ(view.counters().exhausted, 1);
  EXPECT_FALSE(view.reserve().valid);
}
} // namespace
