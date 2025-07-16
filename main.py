from parsers.pdf_parser import PDFParser
from collections import defaultdict
import textwrap
import re

def print_readable_blocks(content_blocks):
    grouped_pages = defaultdict(list)
    for block in content_blocks:
        grouped_pages[block.page_number].append(block)

    for page_num in sorted(grouped_pages.keys()):
        print(f"\n{'='*80}")
        print(f"PAGE {page_num}")
        print(f"{'='*80}")

        for i, block in enumerate(grouped_pages[page_num], 1):
            print(f"\n[Block {i}] {block.content_type.upper()}")
            print(f"Confidence: {block.confidence:.2%}")
            print(f"Source: {block.metadata.get('source', 'text')}")
            print("-" * 60)

            content = block.content.strip()
            if block.content_type == 'heading':
                print(f"📌 {content}")
            elif block.content_type == 'list':
                print("📋 LIST ITEMS:")
                for line in content.split('\n'):
                    if line.strip():
                        print(f"   • {line.strip()}")
            else:
                if block.metadata.get('source') == 'ocr':
                    lines = content.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('@'):
                            cleaned_lines.append(line)
                    content = ' '.join(cleaned_lines)

                    content = re.sub(r'\s+', ' ', content)
                    content = re.sub(r'(\w)\s+(\w)', r'\1 \2', content)
                
                wrapped_text = textwrap.fill(content, width=70, initial_indent="   ", subsequent_indent="   ")
                print(wrapped_text  )

            print()

parser = PDFParser()
pdf_path = "test_files/LISN09L Topic 7 (Revised).pdf"

try:
    content_blocks = parser.parse_pdf(pdf_path)

    print_readable_blocks(content_blocks)
    
except FileNotFoundError as e:
    print(f"Error:{e}")
except Exception as e:
    print(f"Unexpected error: {e}")