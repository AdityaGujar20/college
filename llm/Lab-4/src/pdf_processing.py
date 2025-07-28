import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    """Extract clean text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Clean up text
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        return text.strip()
    
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def prepare_context(text, max_length=1000):
    """Prepare context for QA model by truncating if necessary"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
