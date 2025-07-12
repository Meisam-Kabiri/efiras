from document_readers.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureDocumentProcessor(DocumentProcessor):
    """Azure Document Intelligence processor for complex documents"""
    
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.processor_type = ProcessorType.AZURE_DI
        
        if not config.azure_endpoint or not config.azure_key:
            raise ValueError("Azure endpoint and key required for Azure processor")
    
    def is_available(self) -> bool:
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            return True
        except ImportError:
            return False
    
    def extract_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            client = DocumentAnalysisClient(
                endpoint=self.config.azure_endpoint,
                credential=AzureKeyCredential(self.config.azure_key)
            )
            
            with open(file_path, "rb") as f:
                poller = client.begin_analyze_document("prebuilt-read", f)
                result = poller.result()
            
            text = result.content
            page_texts = []
            tables = []
            
            for page in result.pages:
                page_text = ""
                for line in page.lines:
                    page_text += line.content + "\n"
                page_texts.append(page_text)
            
            if self.config.extract_tables and result.tables:
                for table in result.tables:
                    table_data = []
                    for cell in table.cells:
                        table_data.append({
                            'content': cell.content,
                            'row': cell.row_index,
                            'column': cell.column_index
                        })
                    tables.append(table_data)
            
            metadata = {
                'page_count': len(result.pages),
                'processor': self.processor_type.value,
                'confidence_scores': [page.confidence for page in result.pages],
                'tables_found': len(tables)
            }
            
            return {
                'text': text,
                'page_texts': page_texts,
                'tables': tables,
                'metadata': metadata,
                'extraction_method': 'ocr_and_text'
            }
            
        except Exception as e:
            logger.error(f"Azure Document Intelligence extraction failed: {e}")
            raise
    
    def extract_blocks(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract text blocks from PDF using Azure Document Intelligence to match PyMuPDF format"""
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            client = DocumentAnalysisClient(
                endpoint=self.config.azure_endpoint,
                credential=AzureKeyCredential(self.config.azure_key)
            )
            
            # Extract file metadata
            path = Path(file_path)
            filename = path.name
            filename_without_ext = path.stem
            extension = path.suffix
            
            with open(file_path, "rb") as f:
                poller = client.begin_analyze_document("prebuilt-read", f)
                result = poller.result()
            
            # Get page dimensions (use first page as reference)
            first_page = result.pages[0] if result.pages else None
            width = first_page.width if first_page else 612  # Default PDF width
            height = first_page.height if first_page else 792  # Default PDF height
            
            blocks = []
            
            for page_idx, page in enumerate(result.pages):
                # Convert Azure lines to blocks matching PyMuPDF format
                for line in page.lines:
                    # Azure gives us polygon points, convert to bbox [x0, y0, x1, y1]
                    if line.polygon:
                        x_coords = [point.x for point in line.polygon]
                        y_coords = [point.y for point in line.polygon]
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    else:
                        bbox = [0, 0, 0, 0]  # Fallback if no coordinates
                    
                    blocks.append({
                        'page': page_idx + 1,
                        'bbox': bbox,
                        'text': line.content
                    })
                
                # Optional: Also extract words as smaller blocks for more granular processing
                # Uncomment if you need word-level blocks
                # for word in page.words:
                #     if word.polygon:
                #         x_coords = [point.x for point in word.polygon]
                #         y_coords = [point.y for point in word.polygon]
                #         bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                #     else:
                #         bbox = [0, 0, 0, 0]
                #     
                #     blocks.append({
                #         'page': page_idx + 1,
                #         'bbox': bbox,
                #         'text': word.content
                #     })
            
            logger.info(f"Extracted {len(blocks)} blocks from {filename} using Azure Document Intelligence")
            
            # Match PyMuPDF output format exactly
            output = {
                "height": height,
                "width": width,
                "filename": filename,
                "filename_without_ext": filename_without_ext,
                "processor": "Azure_DI",
                "extension": extension,
                "pages": len(result.pages),
                "blocks": blocks,
            }
            
            # Save extracted blocks
            self._save_extracted_blocks(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Azure Document Intelligence block extraction failed: {e}")
            raise
    
    def _save_extracted_blocks(self, output: Dict[str, Any]) -> None:
        """Save extracted document blocks to JSON file"""
        saving_path = f"data_processed/{output['filename_without_ext']}_raw_blocks_azure.json"
        file_path = Path(saving_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(output, f, indent=4, ensure_ascii=False)

        logger.info(f"Azure blocks data saved to {file_path}")



#   When Azure Document Intelligence makes sense:
#   - Scanned/image-based PDFs requiring OCR
#   - Complex tables that need structured extraction
#   - Forms with specific field detection
#   - Multi-language documents
#   - Poor quality/handwritten documents

#   For regulatory documents (your use case):
#   - Most are text-based PDFs where PyMuPDF excels
#   - You already have good TOC extraction and chunking
#   - Your current pipeline seems to work well

#   Better Azure services for RAG:
#   1. Azure OpenAI Service - GPT models with better rate limits and compliance
#   2. Azure Cognitive Search - Advanced vector search with hybrid capabilities
#   3. Azure AI Search - Semantic search and ranking
#   4. Azure Container Instances - Host your RAG system reliably