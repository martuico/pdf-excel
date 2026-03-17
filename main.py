import os
import time
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from openpyxl import Workbook
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm

INPUT_DIR = "input_pdf"
OUTPUT_DIR = "output_excel"

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (better than fixed 150)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )

    # Morphological closing to connect text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def detect_rows(thresh):
    projection = np.sum(thresh, axis=1)

    rows = []
    start = None

    for i, val in enumerate(projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 12:
                rows.append((start, i))
            start = None

    return rows


def ocr_cell(cell, col_name):
    config = "--oem 3 --psm 6"

    if col_name in ["Debit", "Credit", "Balance"]:
        config += " -c tessedit_char_whitelist=0123456789.,"

    if col_name == "Date":
        config += " -c tessedit_char_whitelist=0123456789/"

    return pytesseract.image_to_string(cell, config=config).strip()


def detect_columns(thresh):
    vertical_proj = np.sum(thresh, axis=0)

    cols = []
    start = None

    for i, val in enumerate(vertical_proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 20:  # filter noise
                cols.append((start, i))
            start = None

    return cols


def extract_table(image):
    thresh = preprocess(image)
    rows = detect_rows(thresh)
    cols = detect_columns(thresh)

    data = []

    for y1, y2 in rows:
        row_dict = {}

        for col, (x1, x2) in enumerate(cols):
            cell = image[y1:y2, x1:x2]
            text = ocr_cell(cell, col)
            row_dict[col] = text

        # Skip junk rows
        if any(row_dict.values()) and "/" in row_dict["Date"]:
            data.append(row_dict)

    return data


def clean_number(text):
    return "".join(c for c in text if c.isdigit() or c == ".")


def process_pdf(pdf_path):
    info = pdfinfo_from_path(pdf_path)
    total_pages = info["Pages"]
    all_data = []

    total_rows = 0
    start_time = time.time()

    with tqdm(
        total=total_pages, desc=f"📄 {os.path.basename(pdf_path)}", unit="page"
    ) as pbar:
        for page_num in range(1, total_pages + 1):

            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
            page = pages[0]

            image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            table = extract_table(image)

            # Clean numbers
            for row in table:
                row["Debit"] = clean_number(row["Debit"])
                row["Credit"] = clean_number(row["Credit"])
                row["Balance"] = clean_number(row["Balance"])

            page_rows = len(table)
            total_rows += page_rows
            all_data.extend(table)

            elapsed = time.time() - start_time
            speed = total_rows / elapsed if elapsed > 0 else 0

            pbar.set_postfix(
                {"rows": total_rows, "page_rows": page_rows, "rows/sec": f"{speed:.1f}"}
            )

            pbar.update(1)

    return all_data


def save_to_excel(data):
    wb = Workbook()
    ws = wb.active

    headers = ["Date", "Journal", "Ref", "Description", "Debit", "Credit", "Balance"]

    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col).value = h

    for r, row in enumerate(data, 2):
        for c, h in enumerate(headers, 1):
            ws.cell(row=r, column=c).value = row[h]

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M") + ".xlsx"
    path = os.path.join(OUTPUT_DIR, filename)

    wb.save(path)
    print(f"Saved: {path}")


def check_dependencies():
    if not os.path.exists(INPUT_DIR):
        return False, "Missing input_pdf folder"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not shutil.which("tesseract"):
        return False, "Install tesseract: sudo apt install tesseract-ocr"

    if not shutil.which("pdftoppm"):
        return False, "Install poppler: sudo apt install poppler-utils"

    return True, ""


def main():
    ok, msg = check_dependencies()
    if not ok:
        print(msg)
        return

    pdfs = [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]

    if not pdfs:
        print("No PDFs found.")
        return

    all_data = []

    for pdf in tqdm(pdfs, desc="📁 PDFs", unit="file"):
        data = process_pdf(os.path.join(INPUT_DIR, pdf))
        all_data.extend(data)

    print(f"\n✅ Total rows extracted: {len(all_data)}")
    save_to_excel(all_data)


if __name__ == "__main__":
    import shutil

    main()
