import sys
import os
from rag.embedding_service import EmbeddingService
from rag.unified_rag import RAGSystem

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem



input_pdf = "data/regulatory_documents/lu/Lux_cssf18_698eng.pdf"
output_dir = Path("data_processed")
output_dir.mkdir(exist_ok=True)

  
reader = PyMuPDFProcessor()


raw_blocks = reader.extract_blocks(input_pdf)

processor = block_processor(raw_blocks)
processed_blocks = processor.process_blocks()

chunker = RegulatoryChunkingSystem(processed_blocks)
chunks_doc = chunker.chunk_blocks()
embd_srv = EmbeddingService()
embd_dict = embd_srv.embed_all_chunks(chunks_doc)
vector_db = embd_dict['embeddings']

rag_system = RAGSystem()


# Step 1: Search for relevant documents
query = "What monitoring elements must IFM implement for central administration delegation?"
query_embeding = embd_srv.embed_text(query)
relevant_docs = rag_system.search(
    vector_db=vector_db,
    query= query,
    query_embedding=query_embeding,
    top_k=5,
    use_hybrid=True
)

# Step 2: Generate answer from relevant documents
answer = rag_system.answer_query(
    vector_db =vector_db,
    query=query, 
    query_embedding=query_embeding,  # Not needed for generation
    top_k=5
)
print(answer)


# embd_srv = EmbeddingService()
# embd_srv.embed_all_chunks(chunks_doc)
# embd_srv.get_provider_name()
# embd_srv.get_config()