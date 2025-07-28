from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os
import tempfile
import json
from pathlib import Path

# Add your src path
sys.path.append(os.path.abspath("src"))
from rag.rag_generator import UnifiedRAGSystem
from src.document_readers.base import DocumentProcessor, ProcessorConfig, ProcessorType
from src.document_processing.block_processor import block_processor
from src.document_chunker.block_chunker import RegulatoryChunkingSystem
from src.document_processing.manager import DocumentProcessorManager

app = FastAPI()

# ADD CORS IMMEDIATELY AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request models
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

# Initialize your RAG system
print("Loading RAG system...")
rag = UnifiedRAGSystem(use_local_embeddings=True, use_azure=False)
print("RAG system loaded!")

# Create output directory for processed files
output_dir = Path("data_processed")
output_dir.mkdir(exist_ok=True)

######################################################################
# temporary do not care about the upload jus use the exisign chunk for lu document
with open ("data_processed/Lux_cssf18_698eng_chunked_blocks.json", 'r') as f :
    chunked_blocks = json.load(f)

rag.add_documents(
    chunked_blocks,
    cache_path=str(output_dir),
    cache_file_name=f"embeddings"
)

######################################################################




@app.get("/")
def home():
    return {"message": "Financial RAG API is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document using your existing pipeline"""
    
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX, and TXT files are supported")
    
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"Processing uploaded file: {file.filename}")
        
        # Step 1: Configure document processor (same as your example)
        config = ProcessorConfig(
            chunk_size=2000,
            overlap=100,
            preserve_formatting=True,
            extract_tables=True,
            ocr_fallback=True
        )
        
        # Step 2: Process the document
        manager = DocumentProcessorManager(config)
        raw_result = manager.process_document(
            temp_file_path,
            preferred_processor="PYMUPDF",
            fallback=True
        )
        
        print(f"Processed {raw_result['pages']} pages using {raw_result['processor']}")
        
        # Step 3: Clean and structure the text
        processor = block_processor()
        processed_data = processor.process_blocks(raw_result)
        
        print(f"Extracted {len(processed_data['table_of_contents'])} TOC entries")
        print(f"Processed {len(processed_data['blocks'])} blocks")
        
        # Step 4: Create manageable chunks
        chunker = RegulatoryChunkingSystem(max_chunk_size=512)
        chunked_blocks = chunker.chunk_blocks(processed_data)
        
        print(f"Created {len(chunked_blocks)} chunks")
        
        # Step 5: Add to RAG system

        rag.add_documents(
            chunked_blocks,
            cache_path=str(output_dir),
            cache_file_name=f"{Path(file.filename).stem}_embeddings"
        )
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return UploadResponse(
            message=f"Document '{file.filename}' processed successfully",
            filename=file.filename,
            chunks_created=len(chunked_blocks)
        )
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Query the processed documents"""
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    try:
        result = rag.answer_with_sources(request.question, top_k=3)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.get("/status")
def get_status():
    """Get the current status of the RAG system"""
    return {
        "status": "running",
        "message": "RAG system is ready for document processing and queries"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG API server...")
    uvicorn.run(app, host="localhost", port=8000)