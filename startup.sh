#!/usr/bin/env bash
set -euo pipefail
echo "=== startup.sh: environment checks and suggested fixes ==="
echo

# require conda env active
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: conda env not active. Run: conda activate cloudspace"
  exit 1
fi
echo "Conda prefix: $CONDA_PREFIX"
echo

echo "1) Python / Torch quick check"
python - <<'PY'
import torch,sys
print("python:", sys.version.splitlines()[0])
print("torch:", getattr(torch,'__version__',None))
print("torch.version.cuda:", getattr(torch.version,'cuda',None))
print("cuda available:", torch.cuda.is_available())
try:
    t = torch.randn(2,2, device='cuda')
    print("CUDA tensor test ok:", t)
except Exception as e:
    print("CUDA test failed:", e)
PY
echo

echo "2) nvcc check (system)"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || true
else
  echo "nvcc not found on PATH"
fi
echo

echo "3) GLIBCXX listings (system vs conda libstdc++)"
echo "System libstdc++:"
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 2>/dev/null | grep GLIBCXX | tail -n 12 || true
echo
echo "Conda env libstdc++:"
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
  strings "$CONDA_PREFIX/lib/libstdc++.so.6" 2>/dev/null | grep GLIBCXX | tail -n 12 || true
else
  echo "No libstdc++ in conda env"
fi
echo

echo "4) Ensure runtime will prefer system libstdc++ (temporary export)"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

echo "5) Check which libstdc++.so the torch .so's will pick (ldd)"
python - <<'PY'
import torch, os, subprocess
lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda_cpp.so")
if os.path.exists(lib):
    print("ldd of:", lib)
    print(subprocess.check_output(["ldd", lib]).decode())
else:
    # fallback: print torch extension path
    print("Torch main file:", torch.__file__)
PY
echo
echo "== startup check complete =="
echo "If nvcc is missing or its version mismatches torch.version.cuda, you will need a matching dev toolkit (e.g. conda cudatoolkit-dev=<torch.cuda>) or install a PyTorch built for the system CUDA."
echo "Next steps (recommended):"
echo "  1) If torch.version.cuda is e.g. 12.1 then run: conda install -n cloudspace -c conda-forge cudatoolkit-dev=12.1 -y"
echo "  2) Then set: export CUDA_HOME=$CONDA_PREFIX; export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  3) Clone detectron2 (main), then: pip install -e . --no-build-isolation --no-use-pep517"
