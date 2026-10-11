#include <xpool/devkit/timeline/diagnostics.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>

#include <unistd.h>

namespace xpool::devkit::timeline {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto sample_interval = std::chrono::seconds(5);
constexpr auto slow_interval = std::chrono::seconds(1);

struct Observation {
  int pid;
  int tid;
  int device;
  const void *owner;
  const char *operation;
  Clock::time_point begin;
  Clock::time_point reported;
  int exceptions;
  bool emitted;
  bool stalled;
};

void emit(const Observation &value, const char *edge, Clock::time_point observed) {
  const auto begin = std::chrono::duration_cast<std::chrono::nanoseconds>(value.begin.time_since_epoch()).count();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(observed - value.begin).count();
  std::fprintf(stderr,
               "xpool timeline diagnostic pid=%d tid=%d device=%d owner=%p operation=%s edge=%s "
               "begin_ns=%lld elapsed_ns=%lld\n",
               value.pid, value.tid, value.device, value.owner, value.operation, edge, static_cast<long long>(begin),
               static_cast<long long>(elapsed));
  std::fflush(stderr);
}

class Monitor {
public:
  Monitor() : worker_([this](std::stop_token stop) { watch(stop); }) {}
  ~Monitor() {
    worker_.request_stop();
    wake_.notify_all();
  }
  void enter(const void *scope, int device, const void *owner, const char *operation) {
    // Static operation names keep this thread-local sampling table bounded by
    // the native diagnostic catalog, rather than by polls or owner lifetimes.
    thread_local std::map<std::string_view, Clock::time_point> samples;
    const auto observed = Clock::now();
    auto &previous = samples[operation];
    const auto sampled = previous == Clock::time_point{} || observed - previous >= sample_interval;
    if (sampled)
      previous = observed;
    auto value = Observation{
        getpid(), gettid(), device, owner, operation, observed, observed, std::uncaught_exceptions(), sampled, false};
    const auto lock = std::lock_guard{mutex_};
    active_.emplace(scope, value);
    if (sampled)
      emit(value, "enter", observed);
  }
  void leave(const void *scope) {
    const auto lock = std::lock_guard{mutex_};
    const auto found = active_.find(scope);
    if (found == active_.end())
      std::terminate();
    const auto &value = found->second;
    const auto observed = Clock::now();
    const auto failed = std::uncaught_exceptions() > value.exceptions;
    if (value.emitted || failed || observed - value.begin >= slow_interval)
      emit(value, failed ? "exception" : "exit", observed);
    active_.erase(found);
  }

private:
  void watch(std::stop_token stop) {
    auto lock = std::unique_lock{mutex_};
    while (!stop.stop_requested()) {
      wake_.wait_for(lock, std::chrono::milliseconds(100), [&] { return stop.stop_requested(); });
      const auto observed = Clock::now();
      for (auto &[scope, value] : active_) {
        if (observed - value.begin >= slow_interval &&
            (!value.stalled || observed - value.reported >= sample_interval)) {
          emit(value, "stalled", observed);
          value.emitted = true;
          value.stalled = true;
          value.reported = observed;
        }
      }
    }
  }
  std::mutex mutex_;
  std::condition_variable wake_;
  std::map<const void *, Observation> active_;
  // Destroy/join the worker before the registry and synchronization storage.
  std::jthread worker_;
};

Monitor &monitor() {
  // Timed-out native callers may survive interpreter/static teardown. Retain
  // this bounded process-local monitor until OS process retirement so its
  // registry cannot be destroyed underneath an active diagnostic scope.
  static auto *value = new Monitor;
  return *value;
}

} // namespace

DiagnosticOperation::DiagnosticOperation(bool enabled, int device, const void *owner, const char *operation)
    : enabled_(enabled) {
  if (enabled_)
    monitor().enter(this, device, owner, operation);
}
DiagnosticOperation::~DiagnosticOperation() {
  if (enabled_)
    monitor().leave(this);
}

} // namespace xpool::devkit::timeline
