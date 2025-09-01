import os
import subprocess
import argparse
import glob
import tempfile
from pathlib import Path

def check_dependencies():
    """Check for required tools"""
    missing = []
    for cmd, desc in [
        ("convert", "ImageMagick"),
        ("pdfunite", "Poppler-utils"),
        ("tesseract", "Tesseract OCR"),
        ("hocr-pdf", "hocr-pdf tool")
    ]:
        try:
            subprocess.run([cmd, "--version"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(desc)
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: sudo apt-get install imagemagick poppler-utils tesseract-ocr hocr-tools")
        return False
    return True

def create_pdf(image_dir, output_pdf):
    """Create PDF from processed images"""
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not images:
        print("No images found")
        return None

    with tempfile.TemporaryDirectory() as tempdir:
        # Convert each image to PDF
        pdf_files = []
        for img in images:
            pdf_path = os.path.join(tempdir, os.path.basename(img) + ".pdf")
            try:
                subprocess.run(["convert", img, pdf_path], check=True)
                pdf_files.append(pdf_path)
            except subprocess.CalledProcessError as e:
                print(f"Error converting {img}: {e}")

        if not pdf_files:
            return None

        # Combine PDFs
        try:
            subprocess.run(["pdfunite"] + pdf_files + [output_pdf], check=True)
            print(f"Created PDF: {output_pdf}")
            return output_pdf
        except subprocess.CalledProcessError as e:
            print(f"Error combining PDFs: {e}")
            return None

def add_ocr(input_pdf, output_pdf):
    """Add OCR layer to PDF"""
    try:
        # Try hocr-pdf method first
        subprocess.run([
            "hocr-pdf",
            os.path.dirname(input_pdf),
            output_pdf
        ], check=True)
        return output_pdf
    except:
        try:
            # Fallback to tesseract pdf output
            base = os.path.splitext(output_pdf)[0]
            subprocess.run([
                "tesseract",
                input_pdf,
                base,
                "pdf"
            ], check=True)
            return f"{base}.pdf"
        except subprocess.CalledProcessError as e:
            print(f"OCR failed: {e}")
            return None

def main(input_dir, output_dir):
    output_pdf = os.path.join(output_dir, "presentation.pdf")
    ocr_pdf = os.path.join(output_dir, "ocr_presentation.pdf")

    # Create PDF
    base_pdf = create_pdf(input_dir, output_pdf)
    if not base_pdf:
        return None

    # Add OCR
    final_pdf = add_ocr(base_pdf, ocr_pdf)
    if final_pdf:
        return final_pdf
    return base_pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create PDF and add OCR')
    parser.add_argument('--input', type=str, required=True, help='Folder with processed images')
    parser.add_argument('--output', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    if not check_dependencies():
        exit(1)

    result = main(args.input, args.output)
    if result:
        print(f"\nSUCCESS: Final PDF saved at {result}")
    else:
        print("\nERROR: PDF creation failed")