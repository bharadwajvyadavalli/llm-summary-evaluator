"""
Document Processor - PDF handling and summary generation
"""

import os
import re
import requests
import PyPDF2
from pathlib import Path
from typing import List, Dict
import openai
import config


class DocumentProcessor:
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def download_pdfs(self, source: str) -> List[str]:
        """Download PDFs from URLs or locate in directory"""
        output_dir = "pdfs"
        os.makedirs(output_dir, exist_ok=True)

        if source.startswith('http'):
            # Download from URLs
            urls = source.split(',')
            paths = []
            for url in urls:
                try:
                    resp = requests.get(url.strip(), timeout=config.PDF_TIMEOUT)
                    filename = f"{output_dir}/doc_{len(paths)}.pdf"
                    with open(filename, 'wb') as f:
                        f.write(resp.content)
                    paths.append(filename)
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
            return paths
        else:
            # Local directory
            return list(Path(source).glob("*.pdf"))

    def process_documents(self, pdf_paths: List[str]) -> List[Dict]:
        """Process multiple PDFs"""
        return [self.process_single_document(path) for path in pdf_paths]

    def process_single_document(self, pdf_path: str) -> Dict:
        """Process single PDF and generate summaries"""
        # Extract text
        text = self._extract_text(pdf_path)

        # Extract abstract as reference
        abstract = self._extract_abstract(text) or text[:500]

        # Generate summaries
        return {
            'name': os.path.basename(pdf_path),
            'text': text,
            'abstract': abstract,
            'summary_high': self._generate_summary(text, "high"),
            'summary_medium': self._generate_summary(text, "medium"),
            'summary_low': self._generate_summary(text, "low")
        }

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return '\n'.join(page.extract_text() for page in reader.pages)
        except:
            return ""

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section"""
        match = re.search(r'abstract[\s:]*\n(.*?)(?=\n\s*introduction)',
                          text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _generate_summary(self, text: str, level: str) -> str:
        """Generate summary at specified level"""
        level_config = config.SUMMARY_LEVELS[level]

        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"{level_config['prompt']}\n\n{text[:level_config['input_chars']]}"
                }],
                max_tokens=level_config['max_tokens'],
                temperature=config.TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except:
            # Fallback to simple extraction
            sentences = text.split('. ')
            if level == 'high':
                return '. '.join(sentences[:2]) + '.'
            elif level == 'medium':
                return '. '.join(sentences[:5]) + '.'
            else:
                return '. '.join(sentences[:20]) + '.'