from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import tempfile
import json
from pathlib import Path

# Add your src path
sys.path.append(os.path.abspath("src"))
from rag.search_service import SearchService
from rag.embedding_service import EmbeddingService
from rag.rag_generator import RAGGenerator
from document_readers.pymupdf_reader import PyMuPDFProcessor
from document_processing.block_processor import block_processor
from document_chunker.block_chunker import RegulatoryChunkingSystem
from fastapi.responses import StreamingResponse

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int

# Initialize the 3-service RAG system
print("Loading RAG system components...")
embedding_service = EmbeddingService()
search_service = SearchService(index_dir="indexes")
rag_generator = RAGGenerator()

# Load all existing embeddings and build/load indexes
print("Loading existing embeddings...")
emb_dir = Path("data_processed")
embd_file_list = list(emb_dir.glob("*embds_local_BAAI_bge-large-en-v1_5*.json"))
doc_list = []

for file in embd_file_list:
    with open(file, 'r') as f:
        doc_list.append(json.load(f))

print(f"Loaded {len(doc_list)} documents with embeddings")

# Setup search service
if not search_service.load_indexes():
    print("Building new search indexes...")
    search_service.build_indexes(doc_list)
    search_service.save_indexes()
else:
    print("Search indexes loaded successfully!")
    search_service.set_chunks(doc_list)

print("RAG system ready!")

@app.post("/query-stream")
async def query_documents_stream(request: QueryRequest):
    """Stream the response as it's generated"""
    
    def generate_response():
        try:
            # Get query embedding and search
            query_embedding = embedding_service.embed_text(request.question)
            relevant_chunks = search_service.search_documents(
                request.question, 
                query_embedding, 
                top_k=12
            )
            
            # Stream the answer generation
            for chunk in rag_generator.answer_query_stream(request.question, relevant_chunks):
                data = {
                    "type": "content",
                    "content": chunk
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send sources at the end - REPLACE [...] WITH YOUR ACTUAL SOURCES CODE
            # sources = []
            # for chunk in relevant_chunks[:5]:  # Top 5 sources
            #     sources.append({
            #         "filename": chunk.get("filename", "Unknown"),
            #         "page": chunk.get("page", "N/A"),
            #         "headers": chunk.get("headers", ""),
            #         "content_preview": chunk.get("content", "")[:200] + "..."
            #     })
            
            # final_data = {
            #     "type": "sources",
            #     "sources": sources
            # }
            # yield f"data: {json.dumps(final_data)}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error", 
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )
@app.post("/query-stream")
async def query_documents_stream(request: QueryRequest):
    def generate_response():
        try:
            query_embedding = embedding_service.embed_text(request.question)
            relevant_chunks = search_service.search_documents(
                request.question, query_embedding, top_k=12
            )
            
            # This will now work because answer_query_stream exists
            for chunk in rag_generator.answer_query_stream(request.question, relevant_chunks):
                data = {"type": "content", "content": chunk}
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send sources and completion
            sources = [...]  # your existing sources code
            final_data = {"type": "sources", "sources": sources}
            yield f"data: {json.dumps(final_data)}\n\n"
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.get("/")
def home():
    return {"message": "Financial RAG API is running with 3-service architecture"}

# @app.post("/upload", response_model=UploadResponse)
# async def upload_document(file: UploadFile = File(...)):
#     """Upload and process a document using the new 3-service architecture"""
    
#     if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
#         raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX, and TXT files are supported")
    
#     try:
#         # Create temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
#             content = await file.read()
#             temp_file.write(content)
#             temp_file_path = temp_file.name
        
#         print(f"Processing uploaded file: {file.filename}")
        
#         # Step 1: Extract text using PyMuPDF
#         processor = PyMuPDFProcessor()
#         raw_result = processor.process_document(temp_file_path)
        
#         # Step 2: Process blocks
#         block_proc = block_processor()
#         processed_data = block_proc.process_blocks(raw_result)
        
#         # Step 3: Create chunks
#         chunker = RegulatoryChunkingSystem(max_chunk_size=512)
#         chunked_blocks = chunker.chunk_blocks(processed_data)
        
#         chunks = chunked_blocks["chunks"]
#         print(f"Created {len(chunks)} chunks")
        
#         # Step 4: Generate embeddings using EmbeddingService
#         print("Generating embeddings...")
#         embedded_doc = {
#             "metadata": {
#                 "filename": file.filename,
#                 "pages": processed_data.get("pages", 0),
#                 "processor": "PyMuPDF"
#             },
#             "embeddings": []
#         }
        
#         for chunk in chunks:
#             # Generate embedding for this chunk
#             embedding = embedding_service.embed_text(chunk["text"])
            
#             embedded_chunk = {
#                 "content": chunk["text"],
#                 "embedding": embedding.tolist(),  # Convert numpy to list for JSON
#                 "id": chunk.get("chunk_id", len(embedded_doc["embeddings"])),
#                 "page": chunk.get("page", 1),
#                 "headers": chunk.get("headers", ""),
#                 "header_identifier": chunk.get("header_identifier", "")
#             }
#             embedded_doc["embeddings"].append(embedded_chunk)
        
#         # Step 5: Save embeddings
#         filename_stem = Path(file.filename).stem
#         embedding_path = emb_dir / f"{filename_stem}_embds_local_BAAI_bge-large-en-v1_5.json"
        
#         with open(embedding_path, 'w') as f:
#             json.dump(embedded_doc, f, indent=2)
        
#         # Step 6: Update search service with new document
#         global doc_list
#         doc_list.append(embedded_doc)
        
#         # Rebuild indexes to include new document
#         search_service.build_indexes(doc_list)
#         search_service.save_indexes()
        
#         # Clean up
#         os.unlink(temp_file_path)
        
#         return UploadResponse(
#             message=f"Document '{file.filename}' processed successfully",
#             filename=file.filename,
#             chunks_created=len(chunks)
#         )
        
#     except Exception as e:
#         if 'temp_file_path' in locals():
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass
        
#         print(f"Error processing file: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# @app.post("/query", response_model=QueryResponse)
# def query_documents(request: QueryRequest):
#     """Query documents using the 3-service architecture"""
#     if not request.question:
#         raise HTTPException(status_code=400, detail="No question provided")
    
#     try:
#         # Step 1: Get query embedding
#         query_embedding = embedding_service.embed_text(request.question)
        
#         # Step 2: Search for relevant chunks
#         relevant_chunks = search_service.search_documents(
#             request.question, 
#             query_embedding, 
#             top_k=12
#         )
        
#         # Step 3: Generate answer
#         answer = rag_generator.answer_query(request.question, relevant_chunks)
        
#         # Step 4: Format sources
#         sources = []
#         for chunk in relevant_chunks[:5]:  # Top 5 sources
#             sources.append({
#                 "filename": chunk.get("filename", "Unknown"),
#                 "page": chunk.get("page", "N/A"),
#                 "headers": chunk.get("headers", ""),
#                 "content_preview": chunk.get("content", "")[:200] + "..."
#             })
        
#         return QueryResponse(
#             question=request.question,
#             answer=answer,
#             sources=sources
#         )
        
#     except Exception as e:
#         print(f"Error querying documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/status")
def get_status():
    """Get system status"""
    return {
        "status": "running",
        "documents_loaded": len(doc_list),
        "total_chunks": sum(len(doc["embeddings"]) for doc in doc_list),
        "search_stats": search_service.get_stats(),
        "rag_config": rag_generator.get_config_info()
    }

@app.get("/documents")
def list_documents():
    """List all loaded documents"""
    docs_info = []
    for doc in doc_list:
        docs_info.append({
            "filename": doc["metadata"]["filename"],
            "pages": doc["metadata"].get("pages", 0),
            "chunks": len(doc["embeddings"])
        })
    return {"documents": docs_info}

if __name__ == "__main__":
    import uvicorn
    print("Starting modernized RAG API server...")
    uvicorn.run(app, host="localhost", port=8000)