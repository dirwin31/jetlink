import Foundation
import Testing

@testable import JetlinkORT

@Suite("ORT session timing")
struct SessionTimingTests {
  @Test("An empty window is finite; failures do not create successful samples")
  func empty() {
    var timing = SessionTiming()
    timing.failed += 1
    let s = timing.summary()
    #expect(s.completed == 0 && s.failed == 1 && s.windowCount == 0)
    #expect(s.meanMs == 0 && s.p99Ms == 0 && s.maxMs == 0)
  }

  @Test("Quantiles include outliers and the rolling window forgets evicted frames")
  func rollingWindow() {
    var timing = SessionTiming(capacity: 100)
    for ms in 1...100 { timing.record(UInt64(ms) * 1_000_000) }
    let s = timing.summary()
    #expect(s.p50Ms == 50 && s.p95Ms == 95 && s.p99Ms == 99 && s.maxMs == 100)
    #expect(s.meanMs == 50.5)
    for _ in 0..<100 { timing.record(2_000_000) }
    let next = timing.summary()
    #expect(next.completed == 200 && next.windowCount == 100)
    #expect(next.meanMs == 2 && next.p99Ms == 2 && next.maxMs == 2)
    #expect(s.maxMs == 100)  // snapshots remain independent
  }

  @Test("Profiling preserves real ORT recurrent outputs and reset behavior")
  func recurrentParity() throws {
    let model = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .appending(path: "Fixtures/tiny_stateful.onnx")
    let plan = SessionPlan(model: model, provider: nil, label: "CPU")
    let baseline = try OrtEngine(
      plans: [plan], device: "cpu", keepAlive: false, keepCPUWarm: false, profileSessions: false)
    let measured = try OrtEngine(
      plans: [plan], device: "cpu", keepAlive: false, keepCPUWarm: false, profileSessions: true)
    defer {
      baseline.close()
      measured.close()
    }
    let pairs = baseline.inputs.keys.filter { $0.hasPrefix("state_") }.sorted().map {
      (input: $0, output: "next_" + $0)
    }
    try baseline.loopState(pairs)
    try measured.loopState(pairs)
    for frame in 0..<9 {
      if frame == 5 {
        baseline.resetState()
        measured.resetState()
      }
      // Distinct image bytes each frame exercise both parity bindings.
      for engine in [baseline, measured] {
        for name in engine.hostInputs {
          let spec = engine.inputs[name]!
          engine.hostInput(name)!.initializeMemory(
            as: UInt8.self, repeating: 0, count: spec.byteCount)
          if name == "new_img" {
            engine.hostInput(name)!.initializeMemory(
              as: UInt8.self, repeating: UInt8(frame), count: spec.byteCount)
          }
        }
        try engine.run()
      }
      for name in baseline.hostOutputs {
        let count = baseline.outputs[name]!.byteCount
        #expect(
          Data(bytes: baseline.output(name)!, count: count)
            == Data(bytes: measured.output(name)!, count: count))
      }
      #expect(baseline.parity == measured.parity)
    }
    #expect(baseline.timingSummaries.isEmpty)
    let summary = try #require(measured.timingSummaries.first)
    #expect(summary.completed == 9 && summary.failed == 0 && summary.windowCount == 9)
    #expect(summary.maxMs > 0)
  }
}
