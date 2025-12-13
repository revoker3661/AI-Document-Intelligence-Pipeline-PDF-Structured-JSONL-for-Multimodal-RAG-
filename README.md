Markdown

# AI Document Intelligence Pipeline: PDF to Structured JSONL (Plan 18)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Pipeline](https://img.shields.io/badge/Plan--18-Robust-orange)

**An advanced, production-grade ETL pipeline designed to convert complex Medical Textbooks into high-fidelity, multimodal JSONL datasets suitable for RAG (Retrieval-Augmented Generation) and Large Multimodal Models (LMMs).**

---

## 🖼️ Project Visualization

![Project Thumbnail](assets/thumbnail.png)

*(This pipeline transforms raw PDFs into structured data, extracting text, tables, and images with semantic preservation.)*

---

## 📖 Project Overview

This project implements the **"Plan 18" Architecture**, a robust document intelligence system specifically engineered for the medical domain. Unlike standard PDF parsers that output messy text, this pipeline acts as a "Structural Surgeon" for documents.

It utilizes a multi-model approach (LayoutParser, Detectron2, PaddleOCR, and Table Transformers) to intelligently segment pages, reconstruct complex tables, and extract figures—all while filtering out noise like dynamic headers and footers. The final output is a strictly formatted **JSONL** file, ready to be indexed into Vector Databases (like Pinecone/Milvus) for high-precision RAG applications.

---

## ✨ Core Features

* **📄 Intelligent Layout Analysis:** Leverages **Detectron2 (PubLayNet)** to distinguish between Text, Titles, Lists, Tables, and Figures with high precision.
* **📊 Smart Table Reconstruction:** Implements a custom **IoU (Intersection over Union)** based merging algorithm combined with **Microsoft Table Transformer** to preserve complex table structures.
* **🧠 Dynamic Header/Footer Filtering:** Automatically learns the layout of a book to remove repetitive headers and footers, ensuring clean data ingestion.
* **👁️ GPU-Accelerated OCR:** Powered by **PaddleOCR** for extracting text from images and non-selectable PDFs.
* **🖼️ Multimodal Extraction:** Automatically crops and saves figures/charts as separate image files, linking them in the JSONL metadata for Multimodal RAG.
* **🛡️ Robust Error Handling:** Includes checkpointing (resume capability) and detailed logging to handle large batch processing without failure.

---

## 🛠️ Tech Stack & Architecture

This pipeline is built on a modular architecture to ensure scalability and maintainability.

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | Python 3.10+ | Main logic handling file I/O and pipeline flow. |
| **Layout Model** | Detectron2 (PubLayNet) | Detecting page segments (Text vs Table vs Image). |
| **OCR Engine** | PaddleOCR (GPU) | High-accuracy text extraction from image blocks. |
| **Table Structure** | Table Transformer | Deep learning model to recognize rows/columns. |
| **Data Processing** | NumPy / OpenCV | Image preprocessing and coordinate manipulation. |
| **Output Format** | JSONL | Line-delimited JSON for scalable streaming/indexing. |

---

## 🚀 Setup & Installation Guide

Follow these steps to set up the pipeline on your local machine or cloud server.

### Step 1: Prerequisites
Ensure you have the following installed:
* **OS:** Linux (Recommended) or Windows with WSL2.
* **GPU:** NVIDIA GPU with CUDA 12.x support (Critical for performance).
* **Git:** To clone the repository.

### Step 2: Clone the Repository

```bash
git clone [https://github.com/revoker3661/AI-Document-Intelligence-Pipeline-PDF-Structured-JSONL-for-Multimodal-RAG-.git](https://github.com/revoker3661/AI-Document-Intelligence-Pipeline-PDF-Structured-JSONL-for-Multimodal-RAG-.git)
Bash

cd AI-Document-Intelligence-Pipeline-PDF-Structured-JSONL-for-Multimodal-RAG-
Step 3: Create Virtual Environment
It is best practice to use a generic virtual environment to manage dependencies.

Bash

python -m venv venv
Activate the environment:

Windows:

Bash

venv\Scripts\activate
Linux/Mac:

Bash

source venv/bin/activate
Step 4: Install Dependencies
Install the core libraries from the requirements file.

Bash

pip install -r requirements_extra.txt
Step 5: Install Detectron2 (Manual)
Detectron2 requires a manual build to link correctly with CUDA. If the previous step failed for Detectron2, run:

Bash

chmod +x install_detectron2_manual.sh
Bash

./install_detectron2_manual.sh
Step 6: Initialize Models (Warm Cache)
Download the heavy model weights (LayoutParser, OCR, Transformers) to your local cache once before running.

Bash

python warm_cache_models.py
Step 7: Validate System
Run the validation script to check if GPU, CUDA, and libraries are linked correctly.

Bash

python validate_setup.py
▶️ Usage
1. Prepare Input Data
Create an input_data folder and place your PDFs inside batch folders.

Structure:

Plaintext

input_data/
└── Batch_01/
    ├── Harrison_Medicine.pdf
    └── Gray_Anatomy.pdf
2. Run the Extractor
Execute the main script to start processing.

Bash

python extract_batch.py
3. Check Outputs
The results will be saved in the output_data directory.

Structure:

Plaintext

output_data/
└── Harrison_Medicine/
    ├── structured_output.jsonl
    └── images/
        ├── page_10_figure_1.png
        └── page_15_figure_2.png
📄 Output Data Example
Each line in the structured_output.jsonl file represents a distinct element:

Table Example:

JSON

{
  "type": "Table",
  "page_number": 42,
  "confidence": 0.98,
  "text": "Table 1: Dosage guidelines...",
  "html_table": "<table><tr><td>Drug</td><td>Dose</td></tr>...</table>",
  "coordinates": [50, 100, 500, 400]
}
Figure Example:

JSON

{
  "type": "Figure",
  "page_number": 42,
  "image_path": "output_data/BookName/images/page_42_element_2_Figure.png",
  "ocr_text": "Figure 1.2: Diagram of the Heart"
}
🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.

Repository Link: https://github.com/revoker3661/AI-Document-Intelligence-Pipeline-PDF-Structured-JSONL-for-Multimodal-RAG-
