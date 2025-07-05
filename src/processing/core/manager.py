from processing.processors.base import *
from processing.processors.pymupdf_processor import PyMuPDFProcessor
from processing.processors.pdfminer_processor import PDFMinerProcessor
from processing.processors.azure_processor import AzureDocumentProcessor
from processing.processors.unstructured_processor import UnstructuredProcessor

class DocumentProcessorManager:
    """Main manager class that orchestrates all processors"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.processors = self._initialize_processors()
        self.fallback_order = [
            ProcessorType.PYMUPDF,      # Fastest
            ProcessorType.UNSTRUCTURED,  # Good fallback
            ProcessorType.PDFMINER,     # Detailed analysis
            ProcessorType.AZURE_DI      # Most powerful but costs money
        ]
    
    def _initialize_processors(self) -> Dict[ProcessorType, DocumentProcessor]:
        processors = {}
        
        # Initialize PyMuPDF
        if PyMuPDFProcessor(self.config).is_available():
            processors[ProcessorType.PYMUPDF] = PyMuPDFProcessor(self.config)
        
        # Initialize Azure (only if credentials provided)
        if self.config.azure_endpoint and self.config.azure_key:
            if AzureDocumentProcessor(self.config).is_available():
                processors[ProcessorType.AZURE_DI] = AzureDocumentProcessor(self.config)
        
        # Initialize PDFMiner
        if PDFMinerProcessor(self.config).is_available():
            processors[ProcessorType.PDFMINER] = PDFMinerProcessor(self.config)
        
        # Initialize Unstructured
        if UnstructuredProcessor(self.config).is_available():
            processors[ProcessorType.UNSTRUCTURED] = UnstructuredProcessor(self.config)
        
        return processors
    
    def get_available_processors(self) -> List[ProcessorType]:
        return list(self.processors.keys())
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        preferred_processor: ProcessorType = ProcessorType.AUTO,
        fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Process document with specified processor or auto-selection
        """
        file_path = Path(file_path)
        
        if preferred_processor == ProcessorType.AUTO:
            processor_order = self._get_auto_processor_order(file_path)
        else:
            processor_order = [preferred_processor]
            if fallback:
                processor_order.extend([p for p in self.fallback_order if p != preferred_processor])
        
        last_error = None
        
        for processor_type in processor_order:
            if processor_type not in self.processors:
                continue
            
            try:
                logger.info(f"Attempting extraction with {processor_type.value}")
                processor = self.processors[processor_type]
                result = processor.extract_text(file_path)
                
                # Check quality
                quality_score = processor.get_quality_score(result)
                result['quality_score'] = quality_score
                result['processor_used'] = processor_type.value
                
                if quality_score >= 0.5:  # Acceptable quality threshold
                    logger.info(f"Successful extraction with {processor_type.value} (quality: {quality_score:.2f})")
                    return result
                elif not fallback:
                    return result
                else:
                    logger.warning(f"Low quality extraction ({quality_score:.2f}), trying fallback")
                    continue
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Processor {processor_type.value} failed: {e}")
                if not fallback:
                    raise
                continue
        
        raise Exception(f"All processors failed. Last error: {last_error}")
    
    def _get_auto_processor_order(self, file_path: Path) -> List[ProcessorType]:
        """Determine best processor order based on file characteristics"""
        file_size = file_path.stat().st_size
        
        # For small files, try fastest first
        if file_size < 1_000_000:  # < 1MB
            return [ProcessorType.PYMUPDF, ProcessorType.UNSTRUCTURED, ProcessorType.PDFMINER]
        
        # For larger files, might need more robust processing
        elif file_size > 10_000_000:  # > 10MB
            return [ProcessorType.AZURE_DI, ProcessorType.UNSTRUCTURED, ProcessorType.PYMUPDF]
        
        # Default order for medium files
        else:
            return self.fallback_order
    
 