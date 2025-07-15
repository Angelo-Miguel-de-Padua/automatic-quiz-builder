from parsers.pdf_parser import PDFParser

parser = PDFParser()

pdf_path = "LISN09L Topic 7 (Revised).pdf"

try:
    pages = parser.parse_pdf(pdf_path)
    for i, page_text in enumerate(pages):
        print (f"\n Page {i + 1} ---\n")
        print(page_text)
except FileNotFoundError as e:
    print(f"Error:{e}")