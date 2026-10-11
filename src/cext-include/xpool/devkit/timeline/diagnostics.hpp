#pragma once

/// \file xpool/devkit/timeline/diagnostics.hpp
/// \brief Opt-in sampled Native boundaries with GIL-independent stall reporting.

#include <cstdint>

namespace xpool::devkit::timeline {

/// Track an active Host operation; successful pairs are sampled every five seconds.
/// A process-local monitor reports active operations after one second, then every
/// five seconds, without using the Python GIL, owner mutexes or CUDA APIs.
/// Names must refer to static strings. Disabled scopes allocate no monitor state.
class DiagnosticOperation {
public:
  DiagnosticOperation(bool enabled, int device, const void *owner, const char *operation);
  /// Retire tracking, reporting exceptions and completed slow operations.
  ~DiagnosticOperation();
  DiagnosticOperation(const DiagnosticOperation &) = delete;
  DiagnosticOperation &operator=(const DiagnosticOperation &) = delete;

private:
  bool enabled_;
};

/// Observe one operation while preserving its return value and exceptions.
template <typename F>
decltype(auto) diagnose(bool enabled, int device, const void *owner, const char *operation, F &&call) {
  const auto diagnostic = DiagnosticOperation(enabled, device, owner, operation);
  return call();
}

} // namespace xpool::devkit::timeline
