#include <cstdint>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.hpp"
#include <xpool/devkit/timeline/recorder.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace xpool::bindings {
void bind_timeline(py::module_ &devkit) {
  auto module = devkit.def_submodule("timeline", "Independent bounded raw Timeline collection.");
  using xpool::devkit::timeline::Chunk;
  using xpool::devkit::timeline::Counters;
  using xpool::devkit::timeline::Record;
  using xpool::devkit::timeline::Recorder;
  py::class_<Record>(module, "Record", "Immutable named fields of one fixed-width raw Timeline fact.")
      .def(py::init([](std::uint64_t sequence, std::uint64_t timestamp, std::uint64_t kind, std::uint64_t site,
                       std::uint64_t generation_high, std::uint64_t generation_low, std::uint64_t endpoint_creator,
                       std::uint64_t endpoint_index, std::uint64_t operation, std::uint64_t instance,
                       std::uint64_t layer, std::uint64_t lane, std::uint64_t lease, std::uint64_t result,
                       std::uint64_t rows, std::uint64_t reserved) {
             return Record{.sequence = sequence,
                           .timestamp = timestamp,
                           .kind = kind,
                           .site = site,
                           .generation_high = generation_high,
                           .generation_low = generation_low,
                           .endpoint_creator = endpoint_creator,
                           .endpoint_index = endpoint_index,
                           .operation = operation,
                           .instance = instance,
                           .layer = layer,
                           .lane = lane,
                           .lease = lease,
                           .result = result,
                           .rows = rows,
                           .reserved = reserved};
           }),
           py::arg("sequence") = 0, py::arg("timestamp") = 0, py::arg("kind") = 0, py::arg("site") = 0,
           py::arg("generation_high") = 0, py::arg("generation_low") = 0, py::arg("endpoint_creator") = 0,
           py::arg("endpoint_index") = 0, py::arg("operation") = 0, py::arg("instance") = 0, py::arg("layer") = 0,
           py::arg("lane") = 0, py::arg("lease") = 0, py::arg("result") = 0, py::arg("rows") = 0,
           py::arg("reserved") = 0)
      .def_readonly("sequence", &Record::sequence, "Producer-owned attempted record sequence, assigned on commit.")
      .def_readonly("timestamp", &Record::timestamp, "Raw nanoseconds in this Producer's declared clock domain.")
      .def_readonly("kind", &Record::kind, "Wire event kind; zero denotes an explicitly aborted reservation.")
      .def_readonly("site", &Record::site, "Host thread identity or Device protocol participant coordinate.")
      .def_readonly("generation_high", &Record::generation_high,
                    "High word of the immutable Fabric Generation identity.")
      .def_readonly("generation_low", &Record::generation_low, "Low word of the immutable Fabric Generation identity.")
      .def_readonly("endpoint_creator", &Record::endpoint_creator,
                    "Device Producer identity of the shared Transport endpoint creator.")
      .def_readonly("endpoint_index", &Record::endpoint_index, "Endpoint creation sequence within that creator.")
      .def_readonly("operation", &Record::operation,
                    "Transport operation or Fabric Invocation sequence, scoped by event kind.")
      .def_readonly("instance", &Record::instance, "Configuration-order Instance index.")
      .def_readonly("layer", &Record::layer, "Canonical model FFN layer ordinal when applicable.")
      .def_readonly("lane", &Record::lane, "Executor Lane index when applicable.")
      .def_readonly("lease", &Record::lease, "Executor Lease sequence when applicable.")
      .def_readonly("result", &Record::result,
                    "Production Result Code when applicable, independent of collection quality.")
      .def_readonly("rows", &Record::rows, "Physical payload rows when applicable.")
      .def_readonly("reserved", &Record::reserved, "Reserved version-one wire word; must remain zero.");
  py::class_<Counters>(module, "Counters", "Bounded quality counter snapshot.")
      .def_readonly("attempted", &Counters::attempted, "Attempt sequence including pre-commit drops.")
      .def_readonly("committed", &Counters::committed, "Successfully committed semantic records.")
      .def_readonly("dropped", &Counters::dropped, "Pre-commit losses, including explicit aborts.")
      .def_readonly("contention", &Counters::contention, "Losses after bounded reservation contention.")
      .def_readonly("exhausted", &Counters::exhausted, "Whether a non-wrapping identity domain stopped.");
  py::class_<Chunk, std::shared_ptr<Chunk>>(module, "Chunk", py::buffer_protocol(), "Owning immutable receipt lease.")
      .def_buffer([](Chunk &chunk) {
        const auto records = chunk.records();
        return py::buffer_info(const_cast<xpool::devkit::timeline::Record *>(records.data()), 1,
                               py::format_descriptor<std::uint8_t>::format(), 1, {records.size_bytes()}, {1}, true);
      })
      .def_property_readonly("sequence", &Chunk::sequence, "Monotonic Chunk identity within this Producer.")
      .def_property_readonly("epoch", &Chunk::epoch, "Source pool epoch of these immutable records.")
      .def_property_readonly("begin_ns", &Chunk::begin_ns,
                             "Host collection window start in CLOCK_MONOTONIC nanoseconds.")
      .def_property_readonly("end_ns", &Chunk::end_ns,
                             "Host receipt completion observation in CLOCK_MONOTONIC nanoseconds.")
      .def("release", &Chunk::release, "Mark published/discarded; views retain storage until destroyed.");
  py::class_<Recorder, std::shared_ptr<Recorder>>(module, "Recorder", "Bounded Host/Device Producer.")
      .def(py::init([](const xpool::devkit::timeline::Options &options, int device) {
             if (device >= 0) {
               xpool::RuntimeState::singleton().require_role(
                   {xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                   "timeline Recorder");
               TORCH_CHECK(xpool::RuntimeState::singleton().device("timeline Recorder") == device,
                           "xpool timeline Device differs from initialized runtime");
             }
             return std::make_shared<Recorder>(options, device);
           }),
           py::arg("options"), py::arg("device") = -1)
      .def("record", &Recorder::record, py::arg("record"), py::call_guard<py::gil_scoped_release>())
      .def("collect", &Recorder::collect, py::arg("seal") = true, py::call_guard<py::gil_scoped_release>())
      .def("counters", &Recorder::counters)
      .def("stop", &Recorder::stop)
      .def("drained", &Recorder::drained)
      .def("close", &Recorder::close, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("source_bytes", &Recorder::source_bytes,
                             "Actual source allocation including control and commit storage.")
      .def_property_readonly("host_bytes", &Recorder::host_bytes,
                             "Host source or receipt allocation bytes, excluding reserved metadata.");
  module.def("allocation_bytes", &xpool::devkit::timeline::allocation_bytes, py::arg("options"), py::arg("device"));
  module.def(
      "set_generation",
      [](std::uint64_t high, std::uint64_t low, std::uint64_t pe) {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
            "timeline generation");
        xpool::devkit::timeline::set_generation(high, low, pe);
      },
      py::arg("high"), py::arg("low"), py::arg("pe"));
  module.def(
      "configure",
      [](std::uint64_t host, std::uint64_t device_id, int device) {
        if (device >= 0) {
          xpool::RuntimeState::singleton().require_role(
              {xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
              "timeline configure");
          TORCH_CHECK(xpool::RuntimeState::singleton().device("timeline configure") == device,
                      "xpool timeline Device differs from initialized runtime");
        }
        xpool::devkit::timeline::configure(host, device_id, device);
      },
      py::arg("host_id"), py::arg("device_id"), py::arg("device"));
  module.def("host_recorder", &xpool::devkit::timeline::host_recorder);
  module.def("device_recorder", &xpool::devkit::timeline::device_recorder);
}
} // namespace xpool::bindings
