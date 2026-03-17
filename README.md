# PDF Data Extraction and Processing

This project is a Python application that extracts and processes data from PDF files. It uses image processing techniques to correct the perspective of the images and OCR (Optical Character Recognition) to extract the data.

## Features

- Convert PDF files to images
- Correct the perspective of the images
- Extract data from the images using OCR
- Save the extracted data to an Excel file

## Requirements

- Python 3.8 or higher
- OpenCV
- PyTesseract
- Openpyxl
- pdf2image
- poppler-utils

## Installation

1. Clone the repository:
```bash
git clone https://github.com/martuico/pdf-excel
```

2. Install the requirements:
```bash
pip install opencv-python
pip install pytesseract
pip install openpyxl
pip install pdf2image
```
Download the [Poppler](https://poppler.freedesktop.org/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) libraries from here and paste them into the `assets` folder.


## Usage

1. Place your PDF files in the `input_pdf` directory and update the `COLUMNS` dictionary as your PDF.
2. Run the script:

```bash
python main.py
```

3. The processed data will be saved in the `output_excel` directory.
