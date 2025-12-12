import os
import subprocess
import torch

print("="*50)
print("🔍 PYTORCH + CUDA CHECK")
print("="*50)

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.rand(2, 2).cuda()
    print("CUDA tensor test successful:", x)

print("\n" + "="*50)
print("🔍 GLIBCXX CHECK (Conda vs System)")
print("="*50)

conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib/libstdc++.so.6")
system_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

def check_glibcxx(lib_path, name):
    try:
        out = subprocess.check_output(
            f"strings {lib_path} | grep GLIBCXX | tail -n 5",
            shell=True, text=True
        )
        print(f"\n{name} ({lib_path}):")
        print(out.strip())
    except Exception as e:
        print(f"{name}: not found ({e})")

check_glibcxx(conda_lib, "Conda libstdc++")
check_glibcxx(system_lib, "System libstdc++")

print("\n✅ Warmup test completed.")
