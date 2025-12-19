# ResonAI - Inference Application

This directory contains the core deployment code for **ResonAI**. This is the exact source code currently running on **Hugging Face Spaces**. It is designed to run as a standalone web application using **Gradio**, allowing users to perform real-time vocal pitch estimation either locally or in a cloud environment.

## File Structure

* **`app.py`**: The main Python script that launches the Gradio interface and handles real-time inference.
* **`vocal_tuna_best.pth`**: The pre-trained 1D-CNN model weights (Required for inference).
* **`requirements.txt`**: List of Python dependencies needed to run this application.

## Installation & Usage

Follow these steps to run the application on your local machine:

### Quick Start (Install & Run)
To install all dependencies and start the application immediately, open your terminal in this directory (`/app`) and run the following commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py
```
### Accessing the App : Once the server starts, the terminal will display a local URL. Open your browser and visit: http://127.0.0.1:7860

### Note: Ensure vocal_tuna_best.pth is present in this folder before running the commands above.
