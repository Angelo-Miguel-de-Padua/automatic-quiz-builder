import fitz
import io
import os
import numpy as np
from PIL import Image
from .process_images import ImageProcessor, OCREngine
from .text_utils import TextCleaner

class PDFProcessor:
    def __init__(self, ocr_lang="en"):
        self.ocr_processor = OCREngine(ocr_lang)
        self.image_processor = ImageProcessor() 
        self.doc = None 

    def extract_from_pdf(self, pdf_path):
        self.doc = fitz.open(pdf_path)  
        try:
            for i in range(len(self.doc)):
                page = self.doc[i]
                txt = page.get_text("text").strip()
                has_text = bool(txt)
                has_images = len(page.get_images(full=True)) > 0

                combined_text = txt
                text_bbox_pairs = []

                if has_text:
                    print(f"[TEX] Page {i+1}: {len(txt)} chars (has image? {has_images})")
                    page_rect = page.rect
                    text_bbox_pairs.append({
                        'text': txt,
                        'bbox': [0, 0, int(page_rect.width), int(page_rect.height)]
                    })

                if has_images:
                    print(f"[IMG] Page {i+1} has {len(page.get_images(full=True))} images -> OCRing them")
                    image_pairs = self._process_page_images_with_ocr(page, i + 1)
                    combined_text += "\n" + " ".join([pair['text'] for pair in image_pairs])
                    text_bbox_pairs.extend(image_pairs)

                if has_text or has_images:
                    yield {
                        'page': i + 1,
                        'text': combined_text.strip(),
                        'source': 'text+image' if has_text and has_images else ('text' if has_text else 'ocr'),
                        'bbox': [0, 0, int(page.rect.width), int(page.rect.height)],
                        'text_bbox_pairs': text_bbox_pairs
                    }
                else:
                    print(f"[EMPTY] Page {i+1} has no text and no image")
        finally:
            self.doc.close()
            self.doc = None  

    def _process_page_images_with_ocr(self, page, page_num):
        image_pairs = []
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = self.doc.extract_image(xref)  
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                img_pil = self.image_processor.enhance_image(img_pil)
                img_pil = self.image_processor.resize_if_needed(img_pil)
                self.image_processor.save_debug_image(img_pil, f"{page_num}_{img_index}", output_dir="ocr_inputs/page_images")
                img_arr = np.array(img_pil)
                ocr_result = self.ocr_processor.process_image(img_arr, page_num)
                image_pairs.extend(ocr_result.get("text_bbox_pairs", []))
            except Exception as e:
                print(f"Failed to OCR image {img_index} on page {page_num}: {e}")
        return image_pairs
    
    def process_pdf(self, pdf_path):
        """Process PDF and return extracted data"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pages_data = list(self.extract_from_pdf(pdf_path))

        return {
            'pages_data': pages_data,
            'base_name': base_name,
            'document_type': 'pdf'
        }