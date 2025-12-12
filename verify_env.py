import torch, torchvision, importlib, subprocess, sys, argparse, os

def check_fast():
    print("===== PyTorch & CUDA =====")
    print("Torch version:", torch.__version__)
    print("Torch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    print("\n===== TorchVision =====")
    print("TorchVision version:", torchvision.__version__)

    print("\n===== Detectron2 =====")
    try:
        detectron2 = importlib.import_module("detectron2")
        print("Detectron2 version:", detectron2.__version__)
    except Exception as e:
        print("❌ Detectron2 not available:", e)

    print("\n===== PaddleOCR =====")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang="en")
        print("PaddleOCR initialized successfully (eng models).")
    except Exception as e:
        print("❌ PaddleOCR issue:", e)

    print("\n===== ONNX Runtime =====")
    try:
        import onnxruntime as ort
        print("onnxruntime version:", ort.__version__)
        print("Execution Providers:", ort.get_available_providers())
    except Exception as e:
        print("❌ ONNX Runtime issue:", e)

    print("\n===== Unstructured (fast mode) =====")
    try:
        import unstructured
        from unstructured.partition.pdf import partition_pdf
        print("Unstructured version:", unstructured.__version__)
        # Quick dummy check (doesn't need internet if installed properly)
        elements = partition_pdf("verify_env.py", strategy="fast")
        print("partition_pdf dummy run successful (fast mode). Extracted:", len(elements), "elements")
    except Exception as e:
        print("❌ Unstructured issue:", e)


def check_hires():
    print("\n===== HI_RES PIPELINE TEST =====")

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    dummy_pdf = "dummy_test.pdf"
    # Create a simple 1-page PDF
    c = canvas.Canvas(dummy_pdf, pagesize=letter)
    c.drawString(100, 750, "This is a HI_RES test page with a TABLE below:")
    c.drawString(100, 700, "| Name | Value |")
    c.drawString(100, 680, "| Test | 123   |")
    c.save()

    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            dummy_pdf,
            strategy="hi_res",
            hi_res_model_name="detectron2",
            ocr_strategy="paddle",
            infer_table_structure=True,
        )
        print(f"HI_RES pipeline successful ✅ — Extracted {len(elements)} elements")
        os.remove(dummy_pdf)
    except Exception as e:
        print("❌ HI_RES pipeline issue:", e)
        if os.path.exists(dummy_pdf):
            os.remove(dummy_pdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify environment setup")
    parser.add_argument("--hires", action="store_true", help="Run full hi_res pipeline test")
    args = parser.parse_args()

    check_fast()

    if args.hires:
        check_hires()

    print("\n===== System Info =====")
    print("Python executable:", sys.executable)
    try:
        subprocess.run(["lsb_release", "-a"])
    except Exception:
        print("lsb_release not available")
