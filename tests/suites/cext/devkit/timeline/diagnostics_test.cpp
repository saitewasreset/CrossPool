#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <xpool/devkit/timeline/diagnostics.hpp>

TEST(TimelineDiagnostics, DisabledScopesRemainSilent) {
  testing::internal::CaptureStderr();
  {
    const auto operation = xpool::devkit::timeline::DiagnosticOperation(false, -1, nullptr, "disabled_probe");
  }
  EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

TEST(TimelineDiagnostics, SamplesFastOperationsAndPreservesExceptions) {
  testing::internal::CaptureStderr();
  for (auto index = 0; index < 100; ++index) {
    const auto operation = xpool::devkit::timeline::DiagnosticOperation(true, 2, nullptr, "sample_probe");
  }
  EXPECT_THROW(xpool::devkit::timeline::diagnose(true, 2, nullptr, "exception_probe",
                                                 [] { throw std::runtime_error("probe failure"); }),
               std::runtime_error);
  const auto output = testing::internal::GetCapturedStderr();
  auto lines = std::size_t{0};
  for (const auto character : output)
    if (character == '\n')
      ++lines;
  EXPECT_EQ(lines, 4);
  EXPECT_NE(output.find("operation=sample_probe edge=enter"), std::string::npos);
  EXPECT_NE(output.find("operation=sample_probe edge=exit"), std::string::npos);
  EXPECT_NE(output.find("operation=exception_probe edge=exception"), std::string::npos);
}

TEST(TimelineDiagnostics, MonitorReportsAStallBetweenSuccessfulSamples) {
  testing::internal::CaptureStderr();
  {
    const auto operation = xpool::devkit::timeline::DiagnosticOperation(true, 3, nullptr, "stalled_probe");
  }
  {
    // The second call is unsampled; only the independent monitor can expose it
    // while the owner remains asleep. No Python interpreter or CUDA API is used.
    const auto operation = xpool::devkit::timeline::DiagnosticOperation(true, 3, nullptr, "stalled_probe");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  }
  const auto output = testing::internal::GetCapturedStderr();
  EXPECT_NE(output.find("operation=stalled_probe edge=stalled"), std::string::npos);
  EXPECT_NE(output.find("device=3"), std::string::npos);
  EXPECT_NE(output.find("tid="), std::string::npos);
  EXPECT_NE(output.find("begin_ns="), std::string::npos);
  EXPECT_NE(output.rfind("operation=stalled_probe edge=exit"), std::string::npos);
}
