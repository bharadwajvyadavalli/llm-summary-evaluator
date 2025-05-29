"""
Simple PDF Processor - Works with any PDF documents
No abstracts required, no academic assumptions
"""

import os
import PyPDF2
from pathlib import Path
import openai
import config


class SimplePDFProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def process_pdfs(self, pdf_directory):
        """Process all PDFs in directory - simple and generic"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        documents = []

        print(f"📁 Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            print(f"🔄 Processing: {pdf_file.name}")
            doc = self.process_single_pdf(str(pdf_file))
            if doc:
                documents.append(doc)
                print(f"   ✅ Success: {len(doc['text']):,} characters extracted")
            else:
                print(f"   ❌ Failed to process")

        print(f"📊 Successfully processed: {len(documents)}/{len(pdf_files)} documents")
        return documents

    def process_single_pdf(self, pdf_path):
        """Process a single PDF - extract text and create basic summary"""
        try:
            # Extract all text from PDF
            text = self.extract_text_from_pdf(pdf_path)

            if not text or len(text) < 50:
                print(f"   ⚠️ Very little text extracted ({len(text)} chars)")
                return None

            # Create a simple summary of the document
            summary = self.create_simple_summary(text)

            return {
                'document_name': os.path.basename(pdf_path),
                'text': text,
                'summary': summary,
                'word_count': len(text.split()),
                'char_count': len(text)
            }

        except Exception as e:
            print(f"   ❌ Error processing {pdf_path}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path):
        """Simple PDF text extraction"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except Exception as e:
                        print(f"     ⚠️ Page {page_num} extraction failed: {e}")
                        continue

                # Basic text cleaning
                text = self.clean_text(text)
                return text

        except Exception as e:
            print(f"   ❌ PDF reading error: {e}")
            return None

    def create_simple_summary(self, text):
        """Create a simple summary of any document"""
        try:
            # Use first part of document for summary
            text_sample = text[:3000]  # Limit for API

            prompt = f"""
            Create a brief summary (2-3 sentences) of this document. 
            Focus on the main topics and key information.

            Document:
            {text_sample}

            Summary:
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"   ⚠️ Summary generation failed: {e}")
            # Fallback: use first few sentences
            sentences = text.split('. ')[:3]
            return '. '.join(sentences) + '.' if sentences else "Summary not available."

    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""

        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        return text.strip()