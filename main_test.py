# from src.doc_processor import *
import os, sys 
sys.path.append(os.path.abspath("src"))

# from src.document_readers.core.manager import *
# from chunking.chunker_simple import RegulationChunker
# from chunking.chunker import RegulationChunker
# from src.cleaning.text_cleaner import RegulationCleaner
from document_chunker.block_chunker import RegulatoryChunkingSystem
import json
from src.utils.useful_functions import convert_numpy_types



if __name__ == "__main__":
    # load the the list of blocks from a json File
    import json
    from pathlib import Path
    file_path = Path('src/cleaning/output_processed_text_pymupdf.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)
    for block in blocks:
        block["text"] = block["text"].replace("\n", " ")
    print(blocks[4]['text'])

    chunker = RegulatoryChunkingSystem()
    chunks = chunker.block_chunker(blocks)
    print(f"Created {len(chunks)} chunks")
    # Save chunks to a JSON file
    clean_chunks = [chunk['content'] for chunk in chunks]
    with open("data/chunks/lux_cssf18_698eng_chunks.json", "w") as f:
        json.dump(clean_chunks, f, indent=2)

        
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
    what is the size and equpiemet used in the aerolab environment?
    """
    langgraph_rag = LangGraphRAG()  # Use LangRag for local RAG
    langgraph_rag.embed_documents(chunks)
    result = langgraph_rag.generate_answer(question)
    print(result['answer'])

