#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace cal_kernels {

#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
constexpr int kNativePasses = 7;
#else
constexpr int kNativePasses = 1;
#endif

struct SampleStats {
  double min;
  double median;
  double avg;
  double max;
};

inline SampleStats summarize_samples(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }

  const double median =
      (n & 1U) ? values[n / 2]
               : 0.5 * (values[n / 2 - 1] + values[n / 2]);
  return {values.front(), median, sum / static_cast<double>(n), values.back()};
}

inline double average_samples(const std::vector<double> &values,
                              std::size_t skip_front = 0) {
  if (skip_front >= values.size()) {
    return 0.0;
  }
  double sum = 0.0;
  for (std::size_t i = skip_front; i < values.size(); ++i) {
    sum += values[i];
  }
  return sum / static_cast<double>(values.size() - skip_front);
}

template <typename T>
inline SampleStats summarize_numeric(const std::vector<T> &values) {
  std::vector<double> converted;
  converted.reserve(values.size());
  for (const T &value : values) {
    converted.push_back(static_cast<double>(value));
  }
  return summarize_samples(converted);
}

template <typename T>
inline std::vector<double> normalize_samples(const std::vector<T> &values,
                                             double denom) {
  std::vector<double> normalized;
  normalized.reserve(values.size());
  for (const T &value : values) {
    normalized.push_back(static_cast<double>(value) / denom);
  }
  return normalized;
}

}  // namespace cal_kernels
