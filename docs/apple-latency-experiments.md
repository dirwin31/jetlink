# Apple inference latency experiments

These changes add measurement and an experimental CoreML output-copy bypass.
They do not establish a latency improvement. Native CoreML state, model graph
changes and device-specific partition selection remain separate work.

## Session measurements

Set `JETLINK_PROFILE_SESSIONS=1` in the server process environment (for iOS,
use the Xcode run scheme's environment variables). Each ORT session records
wall time around its bound execution. Logs from `ort-timing` at timing flush
and engine close include model/provider, successful and failed execution counts,
and mean/p50/p95/p99/max for the last 1,200 successful executions. A failure
increments the failure count without becoming a successful latency sample.
Recording uses bounded storage; sorting happens only when reporting.

The existing `lastGpuUs` protocol field measures whole-engine execution wall
time, including provider synchronization and copies. It is not GPU event time.
Session profiling is off by default and also includes synchronization/copies.
Neither metric includes the complete camera-to-result transport path.

## Experimental CoreML output backings

The released ORT binary cannot be modified by changing Swift. The accompanying
patch targets ONNX Runtime **v1.29.0** and requires rebuilding its CoreML-enabled
XCFramework for the desired Apple platform(s).

1. Check out ONNX Runtime v1.29.0 with its required submodules.
2. Run `python3 scripts/prepare_coreml_output_backings.py /path/to/onnxruntime`.
   The helper checks the original provider file's SHA-256 and refuses unknown
   source revisions or edits; repeating it on the patched source is harmless.
3. Build the XCFramework using ONNX Runtime's Apple build tooling, preserving
   CoreML support and the `onnxruntime` module/framework name. See the pinned
   runtime's `tools/ci_build/github/apple/build_and_assemble_apple_pods.py` and
   `tools/ci_build/github/apple/build_settings` for platform build configurations.
4. Set `JETLINK_ORT_XCFRAMEWORK=../vendor/onnxruntime.xcframework` with the path relative to `JetlinkKit` while
   SwiftPM evaluates `JetlinkKit/Package.swift` (including the Xcode process if
   building there). Without this variable the original pinned download is used.
5. Set `JETLINK_COREML_OUTPUT_BACKINGS=1` in the **server process** before its
   first inference. It is read once by the provider. Unset it to disable the
   experiment on the same custom binary. Restart between A/B runs.

The patch offers CoreML the current ORT destination on each prediction, so
alternating recurrent bindings remain distinct. Only static, nonempty,
non-scalar float32/float16/int32 outputs are candidates. Dynamic shapes and
integer conversions retain the existing path. CoreML feature constraints must
accept the supplied array. The copy is skipped only when the returned array
is the supplied object, its data pointer matches the ORT output, and its layout
is contiguous. CoreML may ignore a backing; the existing copy then remains.
Prediction failures are propagated without retrying a potentially stateful run.
This does not eliminate accelerator transfers or guarantee zero-copy execution.

## Validation before enabling by default

Use the same custom runtime with the flag off/on, identical model and replayed
inputs, and the same compute units and keep-alive settings. Check every output
including recurrent state over long sequences, both buffer parities, reset and
model reload. Check output-backing acceptance in a debugger/profiler: enabling
the flag alone does not prove a copy was removed. Compare sustained p50/p95/p99
and end-to-end latency after warm-up, then repeat at thermal steady state.
Keep profiling enabled in both runs. Report device, OS, model hash, providers,
warm-up exclusion and frame/drop counts; avoid comparing unrelated models.

Implemented validation: the JetlinkORT target builds; timing tests cover empty
windows, failure counts, quantiles and eviction. A real CPU ORT recurrent-model
test checks bit-for-bit parity over alternating buffers and reset with profiling
on/off. These tests do **not** validate the patched CoreML provider or its speed.
The runtime patch still requires a full ORT build and target-device replay.
