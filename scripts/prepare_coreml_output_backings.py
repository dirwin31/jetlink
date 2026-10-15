#!/usr/bin/env python3
"""Apply JetLink's opt-in experiment to an ONNX Runtime v1.29.0 checkout."""
import argparse
import hashlib
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('ort_source', type=Path)
args = parser.parse_args()
root = args.ort_source.resolve()
patch = Path(__file__).resolve().parents[1] / 'patches/onnxruntime/1.29.0-coreml-output-backings.patch'
source = root / 'onnxruntime/core/providers/coreml/model/model.mm'
if not source.is_file():
    parser.error(f'Not an ONNX Runtime source tree: {root}')
reverse = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=root, capture_output=True)
if reverse.returncode == 0:
    print('Output-backings patch already applied.')
    raise SystemExit(0)
expected = '50f51f8c9da3ff6a15356a6dc31993cf2dcd1d5079aa7fbb53aeade0e29f84c4'
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    parser.error('model.mm differs from pinned v1.29.0; refusing to overwrite or patch it.')
subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True)
subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
print('Applied. Rebuild the CoreML-enabled XCFramework before enabling JETLINK_COREML_OUTPUT_BACKINGS=1.')
