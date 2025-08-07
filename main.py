import logging
from app.document_processing.pdf import PDFProcessor
from app.outputs.output_manager import OutputManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    pdf_file = "test_files/LISN09L Topic 7 (Revised).pdf"
    output_dir = "outputs"
    
    # Process PDF (extraction only)
    print("\nProcessing PDF...")
    extractor = PDFProcessor(ocr_lang="en")
    results = extractor.process_pdf(pdf_file)
    
    # Handle output saving (separate concern)
    print("Saving results...")
    output_manager = OutputManager(output_dir)
    
    # Save structured JSON
    json_path = output_manager.save_structured_json(
        results['pages_data'], 
        results['base_name']
    )
    
    # Format and save plain text  
    formatted_text = output_manager.format_plain_text(results['pages_data'])
    txt_path = output_manager.save_plain_text(
        formatted_text, 
        results['base_name']
    )
    
    print("\nFiles saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")
    
    return results

if __name__ == "__main__":
    main()