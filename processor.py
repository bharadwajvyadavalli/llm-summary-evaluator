"""PDF Processing and Summary Generation"""

import re
import PyPDF2
from pathlib import Path
import openai
import config

class PDFProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def process_directory(self, directory, training_mode=False):
        """Process all PDFs in directory"""
        pdf_files = list(Path(directory).glob("*.pdf"))
        documents = []

        for pdf_file in pdf_files:
            print(f"📄 Processing: {pdf_file.name}")

            if training_mode:
                # Generate multiple summaries for training
                doc_summaries = self.process_pdf(str(pdf_file), generate_multiple=True)
                if doc_summaries:
                    documents.extend(doc_summaries)
            else:
                # Generate single summary for inference
                doc = self.process_pdf(str(pdf_file), generate_multiple=False)
                if doc:
                    documents.append(doc)

        return documents

    def process_pdf(self, pdf_path, generate_multiple=False):
        """Process single PDF"""
        try:
            # Extract text
            text = self.extract_text(pdf_path)
            if not text:
                return None

            # Extract abstract
            abstract = self.extract_abstract(text)

            if generate_multiple:
                # For training: generate 3 different quality summaries
                documents = []
                for quality in ['high', 'medium', 'low']:
                    summary = self.generate_summary(text, abstract, quality)
                    documents.append({
                        'document': Path(pdf_path).name,
                        'text': text,
                        'abstract': abstract or "No abstract found",
                        'summary': summary,
                        'summary_quality': quality
                    })
                return documents
            else:
                # For inference: generate single high-quality summary
                summary = self.generate_summary(text, abstract, 'high')
                return {
                    'document': Path(pdf_path).name,
                    'text': text,
                    'abstract': abstract or "No abstract found",
                    'summary': summary
                }

        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return None

    def extract_text(self, pdf_path):
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
        """Extract abstract from text"""
        patterns = [
            r'(?i)abstract[\s:]*\n(.*?)(?=\n\s*(?:introduction|keywords|1\.))',
            r'(?i)abstract[\s:]*(.*?)(?=\n\s*[A-Z])'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                if 50 <= len(abstract) <= 2000:
                    return re.sub(r'\s+', ' ', abstract)

        return None

    def generate_summary(self, text, abstract, quality='high'):
        """Generate summary using LLM with specified quality level"""
        try:
            # Use abstract as context if available
            context = abstract if abstract else text[:1500]

            # Get appropriate prompt
            prompt = config.SUMMARY_PROMPTS.get(quality, config.SUMMARY_PROMPTS['high'])

            response = self.client.chat.completions.create(
                model=config.SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Document:\n{text[:3000]}\n\nContext:\n{context}"}
                ],
                max_tokens=config.MAX_SUMMARY_LENGTH,
                temperature=0.3 if quality == 'high' else 0.5 if quality == 'medium' else 0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Summary generation error: {e}")
            # Fallback to first few sentences
            sentences = text.split('. ')[:3]
            return '. '.join(sentences) + '.'