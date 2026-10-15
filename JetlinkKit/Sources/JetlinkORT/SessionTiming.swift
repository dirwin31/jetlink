import Foundation

/// Bounded, allocation-free recording after initialization. Owned by the engine
/// and protected by its caller's inference lock. Sorting happens only at flush.
struct SessionTiming {
  private var samples: [UInt64]
  private var cursor = 0
  private var count = 0
  private(set) var completed: UInt64 = 0
  var failed: UInt64 = 0

  init(capacity: Int = 1200) {
    precondition(capacity > 0)
    samples = Array(repeating: 0, count: capacity)
  }

  mutating func record(_ nanoseconds: UInt64) {
    samples[cursor] = nanoseconds
    cursor = (cursor + 1) % samples.count
    count = min(count + 1, samples.count)
    completed += 1
  }

  struct Summary {
    let completed: UInt64
    let failed: UInt64
    let windowCount: Int
    let meanMs: Double
    let p50Ms: Double
    let p95Ms: Double
    let p99Ms: Double
    let maxMs: Double

    var description: String {
      "completed=\(completed) failed=\(failed) last=\(windowCount) "
        + String(
          format: "mean=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f ms", meanMs, p50Ms, p95Ms, p99Ms,
          maxMs)
    }
  }

  func summary() -> Summary {
    let sorted = samples.prefix(count).sorted()
    // Nearest-rank quantiles, including the last observation for p100.
    func percentile(_ fraction: Double) -> Double {
      guard !sorted.isEmpty else { return 0 }
      return Double(sorted[max(0, Int(ceil(fraction * Double(count))) - 1)]) / 1e6
    }
    return Summary(
      completed: completed, failed: failed, windowCount: count,
      meanMs: count == 0 ? 0 : sorted.reduce(0.0) { $0 + Double($1) / 1e6 } / Double(count),
      p50Ms: percentile(0.50), p95Ms: percentile(0.95), p99Ms: percentile(0.99),
      maxMs: percentile(1))
  }
}
