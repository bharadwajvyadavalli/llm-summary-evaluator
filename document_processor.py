"""
Document Processor for LLM Evaluation POC
"""

import os
import re
import PyPDF2
from pathlib import Path
import openai
import config

class DocumentProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    def process_pdfs(self, pdf_directory):
        """Process all PDFs in directory"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        documents = []
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            doc = self.process_single_pdf(str(pdf_file))
            if doc:
                documents.append(doc)
        
        return documents
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file"""
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return None
            
            # Extract abstract (golden record)
            abstract = self.extract_abstract(text)
            if not abstract:
                print(f"Warning: No abstract found in {os.path.basename(pdf_path)}")
                return None
            
            # Generate summary
            summary = self.generate_summary(text)
            
            return {
                'document_name': os.path.basename(pdf_path),
                'text': text,
                'abstract': abstract,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return None
    
    def extract_abstract(self, text):
        """Extract abstract from document text"""
        # Look for abstract section
        patterns = [
            r'(?i)abstract[\s:]*\n(.*?)(?=\n\s*(?:introduction|keywords|1\.))',
            r'(?i)abstract[\s:]*\n(.*?)(?=\n\s*[A-Z])',
            r'(?i)abstract[\s:]*(.*?)(?=\n\s*(?:introduction|keywords))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Validate length
                if 50 <= len(abstract) <= 2000:
                    return self.clean_text(abstract)
        
        return None
    
    def generate_summary(self, text):
        """Generate summary using LLM"""
        try:
            prompt = f"""
            Summarize this research document in 100-300 words. Focus on main findings and conclusions.
            
            Document:
            {text[:3000]}  # Limit for token usage
            
            Summary:
            """
            
            response = self.client.chat.completions.create(
                model=config.SUMMARY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Summary generation error: {e}")
            # Fallback: use first few sentences
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common artifacts
        text = re.sub(r'^(abstract|summary)[\s:]*', '', text, flags=re.IGNORECASE)
        return text.strip()