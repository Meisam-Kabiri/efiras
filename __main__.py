# from src.doc_processor import *
import os, sys 
sys.path.append(os.path.abspath("src"))

from src.efiras_processor.core.manager import *




processor  = PyMuPDFProcessor(ProcessorConfig())
result = processor.extract_text("data/regulatory_documents/lu/Lux_cssf18_698eng.pdf")
print(result['text'][:5000])  # Print first 500 characters of extracted text


config = ProcessorConfig(
        chunk_size=1000,
        overlap=200,
        extract_tables=True,
        ocr_fallback=True,
        # azure_endpoint="https://your-resource.cognitiveservices.azure.com/",
        # azure_key="your-key-here"
    )
    
# Initialize manager
manager = DocumentProcessorManager(config)

print(f"Available processors: {[p.value for p in manager.get_available_processors()]}")

# Process a document
# try:
result = manager.process_document(
    "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf",
    preferred_processor=ProcessorType.AUTO,
    fallback=True
)

print(f"Extraction successful with {result['processor_used']}")
print(f"Quality score: {result['quality_score']:.2f}")
print(f"Text length: {len(result['text'])} characters")
print(f"Pages: {result['metadata']['page_count']}")

# Chunk the document
chunks = manager.chunk_document(result, "sample_regulation.pdf")
print(f"Created {len(chunks)} chunks")

    
# except Exception as e:
#     print(f"Processing failed: {e}")


