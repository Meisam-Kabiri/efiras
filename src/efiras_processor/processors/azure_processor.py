from efiras_processor.processors.base import DocumentProcessor, ProcessorConfig, ProcessorType
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

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