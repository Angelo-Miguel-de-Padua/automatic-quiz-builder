from parsers.pdf_parser import PDFParser
from collections import defaultdict

parser = PDFParser()

pdf_path = "test_files/LISN09L Topic 7 (Revised).pdf"

try:
    content_blocks = parser.parse_pdf(pdf_path)

    grouped_pages = defaultdict(list)
    for block in content_blocks:
        grouped_pages[block.page_number].append(block)
    
    for page_num in sorted(grouped_pages.keys()):
        print(f"\n Page {page_num} ---\n")
        for block in grouped_pages[page_num]:
            print(block)
    
except FileNotFoundError as e:
    print(f"Error:{e}")