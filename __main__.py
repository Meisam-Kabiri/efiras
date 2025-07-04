# from src.doc_processor import *
import os, sys 
sys.path.append(os.path.abspath("src"))

from src.processing.core.manager import *
# from chunking.chunker_simple import RegulationChunker
from chunking.chunker import RegulationChunker
from src.cleaning.text_cleaner import RegulationCleaner
from chunking.hybrid_chunker import RegulatoryChunkingSystem
import json
from src.utils.useful_functions import convert_numpy_types





# processor  = PyMuPDFProcessor(ProcessorConfig())
# result = processor.extract_text("data/regulatory_documents/lu/Lux_cssf18_698eng.pdf")
# print(result['text'][:5000])  # Print first 500 characters of extracted text


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
result = manager.process_document(
    "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf",
    preferred_processor=ProcessorType.AUTO,
    fallback=True
)

print(f"Extraction successful with {result['processor_used']}")
print(f"Quality score: {result['quality_score']:.2f}")
print(f"Text length: {len(result['text'])} characters")
print(f"Pages: {result['metadata']['page_count']}")

cleaner = RegulationCleaner(result['text'])
cleaned_text = cleaner.clean()

# chunker = RegulationChunker(cleaned_text)
# chunks = chunker.chunk()

chunker = RegulatoryChunkingSystem()
chunks = chunker.process_document(cleaned_text)
print(f"Created {len(chunks)} chunks")
# print(f"First chunk: {chunks[0]['heading']} - {chunks[0]['content'][:1000]}...")

# example_usage.py
"""
Example usage of local and Azure RAG systems.
"""

from src.rag.rag_local import LocalRAG
from src.rag.langchain_rag import LangChainRAG
from src.rag.langgraph_rag import LangGraphRAG
# from rag_azure import AzureRAG

# Local RAG
# local_rag = LocalRAG()
# local_rag.embed_documents(chunks)
# print("Local RAG:\n", local_rag.generate_answer("What are the key components that an Investment Fund Manager (IFM) should include in their due diligence and ongoing monitoring process when delegating portfolio management, according to Circular CSSF 18/698?"))

# question = "What are the key components that an Investment Fund Manager (IFM) should include in their due diligence and ongoing monitoring process when delegating portfolio management, according to Circular CSSF 18/698?"
# langchain_rag = LangChainRAG()  # Use LangRag for local RAG
# langchain_rag.embed_documents(chunks)
# result = langchain_rag.generate_answer_with_sources(question)
# print(result['answer'])

question = "What are the key components that an Investment Fund Manager (IFM) should include in their due diligence and ongoing monitoring process when delegating portfolio management, according to Circular CSSF 18/698?"
question = """
A Luxembourg UCITS management company wants to delegate its risk management function to a UK-based entity that is authorized as both an investment firm and an AIFM. The UK entity proposes to use its proprietary risk management system, which has been approved by the UK FCA but not specifically validated by any EU authority. The Luxembourg management company also wants to delegate portfolio management for 30% of one UCITS fund to the same UK entity, while keeping the remaining 70% in-house. The UK entity would make investment decisions for its allocated portion but all trade execution would remain with the Luxembourg entity.
Is this dual delegation arrangement (risk management + partial portfolio management) to the same entity permissible under the CSSF framework?
"""
langgraph_rag = LangGraphRAG()  # Use LangRag for local RAG
langgraph_rag.embed_documents(chunks)
result = langgraph_rag.generate_answer(question)
print(result)




# # Azure RAG (requires Azure setup)
# az_rag = AzureRAG(
#     endpoint="https://<your-service>.search.windows.net",
#     index_name="financial-docs",
#     api_key="<your-key>"
# )
# print("Azure RAG:\n", az_rag.generate_answer("What are AMLD5 obligations?"))

# save the chunk in a File
# create the folder and file first if does not exist
os.makedirs("data/chunks", exist_ok=True)
# Save chunks to a JSON file
clean_chunks = convert_numpy_types(chunks)

with open("data/chunks/lux_cssf18_698eng_chunks.json", "w") as f:
    json.dump(clean_chunks, f, indent=2)



# Chunk the document
# chunks = manager.chunk_document(result, "sample_regulation.pdf")
# print(f"Created {len(chunks)} chunks")

    


