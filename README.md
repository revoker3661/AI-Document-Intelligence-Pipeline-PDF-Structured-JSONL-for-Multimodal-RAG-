# AI-Document-Intelligence-Pipeline-PDF-Structured-JSONL-for-Multimodal-RAG-
A robust, production-grade pipeline converting complex Medical PDFs into structured, RAG-ready JSONL datasets. Features smart table merging, multimodal extraction, and dynamic layout analysis using Detectron2 &amp; PaddleOCR.
Plaintext

Medical-RAG-Pipeline/
│
├── input_data/                  # (Auto-created) Yahan apni PDF books rakhein
│   ├── Batch_01/
│   │   ├── book1.pdf
│   │   └── book2.pdf
│   └── Batch_02/
│
├── output_data/                 # (Auto-created) Processed JSONL aur Images yahan aayenge
│   ├── book1/
│   │   ├── structured_output.jsonl
│   │   └── images/
│
├── logs/                        # (Auto-created) Execution logs yahan save honge
│
├── models/                      # (Optional) Agar manual model download karke rakhne ho
│
├── extract_batch.py             # 🧠 MAIN BRAIN: Orchestrator script (Plan 18)
├── validate_setup.py            # 🛠 TOOL: Environment checker (GPU/Libs)
├── warm_cache_models.py         # 📥 SETUP: Models ko pehli baar download karne ke liye
├── requirements_extra.txt       # 📋 LIST: Sabhi libraries ki list
├── install_detectron2_manual.sh # 🐚 SCRIPT: Detectron2 install helper
│
└── README.md                    # 📖 GUIDE: Jo hum niche likh rahe hain
2. Professional GitHub README.md
Is content ko copy karke README.md file bana lo. Maine "Models Download" aur "YouTube Video" wale section wese hi daale hain jaise tumne kaha tha.

Markdown

# 🏥 Medical Book Extraction Pipeline for RAG (Plan 18)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red)
![Status](https://img.shields.io/badge/Pipeline-Production%20Ready-success)
![GPU](https://img.shields.io/badge/GPU-Required-orange)

An advanced, industrial-grade pipeline designed to convert complex Medical Textbooks (PDFs) into structured **JSONL** format suitable for **RAG (Retrieval-Augmented Generation)** models. 

This project implements **"Plan 18" architecture**, featuring Smart Table Merging, Dynamic Header/Footer detection, and Multi-Model Cross-Validation (LayoutParser + Table Transformer + PaddleOCR).

---

## ⚡ Key Features

* **📄 Intelligent Layout Analysis:** Uses **Detectron2 (PubLayNet)** to segment pages into Text, Tables, Figures, and Lists.
* **📊 Smart Table Extraction:** Implements an **IoU-based Smart Merging** algorithm to reconstruct tables accurately using **Microsoft Table Transformer**.
* **🧠 Dynamic Filtering:** Automatically detects and removes Headers and Footers based on recurring patterns in the book.
* **👁️ High-Quality OCR:** Powered by **PaddleOCR (GPU)** for robust text extraction from images and non-selectable PDFs.
* **🛡️ Robustness:** Includes checkpointing (resume where you left off) and detailed logging.

---

## 🛠️ Prerequisites

Before starting, ensure your system meets these requirements:

1.  **OS:** Linux (Recommended) or Windows with WSL2.
2.  **GPU:** NVIDIA GPU with CUDA 12.1 support (Required for efficient processing).
3.  **Python:** Version 3.8 or higher.
4.  **System Libraries:** `libstdc++` (Modern version required for OCR).

---

## 🚀 Installation Guide (Step-by-Step)

Follow these steps strictly to set up the environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Medical-RAG-Pipeline.git](https://github.com/YOUR_USERNAME/Medical-RAG-Pipeline.git)
cd Medical-RAG-Pipeline
2. Create Virtual Environment
Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Core Dependencies
We have a curated requirements file. Install it using pip:

Bash

pip install -r requirements_extra.txt
4. Install Detectron2 (Crucial Step)
Detectron2 can be tricky. If the command above fails for Detectron2, use our manual script:

Bash

chmod +x install_detectron2_manual.sh
./install_detectron2_manual.sh
📥 Model Setup (First Time Run)
This pipeline uses large models (LayoutParser, Table Transformer, PaddleOCR). Instead of downloading them during runtime, run the warmer script first to download and cache them locally.

Run this command once:

Bash

python warm_cache_models.py
This script will download all necessary weights to ~/.torch/ and ~/.paddleocr/ directories.

✅ Validate Setup
Before running the main processor, run the validation tool to check if GPU, CUDA, and Libraries are linked correctly:

Bash

python validate_setup.py
If you see "✅ VALIDATION COMPLETE", you are ready to go!

▶️ How to Run
1. Prepare Input Data
Create a folder named input_data and add your PDF files inside batch folders:

Plaintext

input_data/
    Batch_01/
        Anatomy_Book.pdf
        Physiology.pdf
2. Start Processing
Run the main orchestrator script:

Bash

python extract_batch.py
3. Check Outputs
The script will generate an output_data folder.

JSONL: Contains the structured text and metadata.

Images: Contains extracted figures/charts cropped from the pages.

🎥 Video Tutorial
Click the image below to watch the complete step-by-step setup and demo video on YouTube.

(Note: The video demonstrates how to configure the paths and interprets the JSONL output.)

📄 Output Structure (JSONL)
Each line in the output file represents a single element (Text block, Table, or Image):

JSON

{
  "type": "Table",
  "page_number": 45,
  "coordinates": [100, 200, 500, 600],
  "confidence": 0.98,
  "text": "Full text content of the table...",
  "html_table": "<table><tr><td>Cell Data</td>...</table>"
}
📞 Contact & Support
For issues or contributions, please open an issue in this repository.

Maintainer: [Your Name]
