#pragma once

/// \file xpool/devkit/timeline/device.cuh
/// \brief Independent Device observation entry points and Transport sidebands.

#include <xpool/devkit/timeline/pool.hpp>
#include <xpool/macros.hpp>

namespace xpool::devkit::timeline {

/// Process-local device symbol; independent of production arena ownership.
extern XPOOL_DEVICE_CONST PoolView device_view;

/// Record one immutable Device fact without allocating or waiting for a consumer.
XPOOL_DEVICE_FN inline void record(Record value) {
  if (!device_view.base)
    return;
  device_view.commit(device_view.reserve(), value);
}

} // namespace xpool::devkit::timeline
