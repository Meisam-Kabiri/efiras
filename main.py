# Process a document
"""
Complete EFIRAS Document Processing Pipeline Example
Shows how to use all processors together with fallback and validation
"""
# from src.doc_processor import *
import os, sys 
sys.path.append(os.path.abspath("src"))



import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Assuming the modules are in separate files
# from document_processor import DocumentProcessorManager, ProcessorConfig, ProcessorType
from src.efiras_processor.core.manager import *
from src.chunking.regulatory_chunker import RegulatoryChunker, ChunkValidator, DocumentChunk

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EFIRASDocumentPipeline:
    """Complete document processing pipeline for EFIRAS"""
    
    def __init__(self, 
                 azure_endpoint: str = None, 
                 azure_key: str = None,
                 output_dir: str = "processed_documents"):
        
        # Configuration
        self.config = ProcessorConfig(
            chunk_size=1000,
            overlap=200,
            preserve_formatting=True,
            extract_tables=True,
            ocr_fallback=True,
            azure_endpoint=azure_endpoint,
            azure_key=azure_key
        )
        
        # Initialize components
        self.processor_manager = DocumentProcessorManager(self.config)
        self.chunker = RegulatoryChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized EFIRAS pipeline with processors: {[p.value for p in self.processor_manager.get_available_processors()]}")
    
    def process_single_document(self, 
                              file_path: str,
                              preferred_processor: ProcessorType = ProcessorType.AUTO,
                              save_output: bool = True) -> Dict[str, Any]:
        """Process a single document end-to-end"""
        
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path.name}")
        
        try:
            # Step 1: Extract text
            logger.info("Step 1: Extracting text...")
            extracted_data = self.processor_manager.process_document(
                file_path=file_path,
                preferred_processor=preferred_processor,
                fallback=True
            )
            
            logger.info(f"✅ Extraction successful with {extracted_data['processor_used']} "
                       f"(quality: {extracted_data['quality_score']:.2f})")
            
            # Step 2: Create chunks
            logger.info("Step 2: Creating chunks...")
            chunks = self.chunker.chunk_document(extracted_data, file_path)
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Validate chunks
            logger.info("Step 3: Validating chunks...")
            validation = ChunkValidator.validate_chunk_set(chunks)
            
            if validation['overall_valid']:
                logger.info(f"✅ Validation passed ({validation['valid_chunks']}/{validation['total_chunks']} chunks valid)")
            else:
                logger.warning(f"⚠️ Validation issues ({validation['valid_chunks']}/{validation['total_chunks']} chunks valid)")
            
            # Step 4: Prepare results
            result = {
                'source_file': str(file_path),
                'extraction_info': {
                    'processor_used': extracted_data['processor_used'],
                    'quality_score': extracted_data['quality_score'],
                    'extraction_method': extracted_data['extraction_method'],
                    'page_count': extracted_data['metadata']['page_count']
                },
                'chunks': [self._chunk_to_dict(chunk) for chunk in chunks],
                'validation': validation,
                'summary': {
                    'total_chunks': len(chunks),
                    'total_characters': sum(len(chunk.content) for chunk in chunks),
                    'regulation_types': list(set(chunk.metadata.get('regulation_type') 
                                                for chunk in chunks 
                                                if chunk.metadata.get('regulation_type'))),
                    'jurisdictions': list(set(chunk.metadata.get('jurisdiction') 
                                            for chunk in chunks 
                                            if chunk.metadata.get('jurisdiction'))),
                    'page_coverage': validation['page_coverage']
                }
            }
            
            # Step 5: Save output
            if save_output:
                self._save_processing_result(result, file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {file_path.name}: {e}")
            raise
    
    def process_document_batch(self, 
                             file_paths: List[str],
                             preferred_processor: ProcessorType = ProcessorType.AUTO) -> Dict[str, Any]:
        """Process multiple documents in batch"""
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        results = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                result = self.process_single_document(
                    file_path=file_path,
                    preferred_processor=preferred_processor,
                    save_output=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append({'file': file_path, 'error': str(e)})
        
        # Create batch summary
        batch_result = {
            'processed_files': len(results),
            'failed_files': len(failed_files),
            'total_chunks': sum(r['summary']['total_chunks'] for r in results),
            'results': results,
            'failures': failed_files,
            'regulation_coverage': self._analyze_regulation_coverage(results),
            'processing_stats': self._calculate_processing_stats(results)
        }
        
        # Save batch summary
        self._save_batch_summary(batch_result)
        
        logger.info(f"✅ Batch processing complete: {len(results)}/{len(file_paths)} files processed")
        
        return batch_result
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary for JSON serialization"""
        return {
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'source_document': chunk.source_document,
            'page_numbers': chunk.page_numbers,
            'section_title': chunk.section_title,
            'confidence_score': chunk.confidence_score
        }
    
    def _save_processing_result(self, result: Dict[str, Any], source_file: Path):
        """Save processing result to JSON file"""
        output_file = self.output_dir / f"{source_file.stem}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved processing result to {output_file}")
    
    def _save_batch_summary(self, batch_result: Dict[str, Any]):
        """Save batch processing summary"""
        output_file = self.output_dir / "batch_summary.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved batch summary to {output_file}")
    
    def _analyze_regulation_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what regulations are covered across all documents"""
        regulation_counts = {}
        jurisdiction_counts = {}
        
        for result in results:
            for reg_type in result['summary']['regulation_types']:
                regulation_counts[reg_type] = regulation_counts.get(reg_type, 0) + 1
            
            for jurisdiction in result['summary']['jurisdictions']:
                jurisdiction_counts[jurisdiction] = jurisdiction_counts.get(jurisdiction, 0) + 1
        
        return {
            'regulations': regulation_counts,
            'jurisdictions': jurisdiction_counts,
            'unique_regulations': len(regulation_counts),
            'unique_jurisdictions': len(jurisdiction_counts)
        }
    
    def _calculate_processing_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        processors_used = {}
        quality_scores = []
        chunk_sizes = []
        
        for result in results:
            processor = result['extraction_info']['processor_used']
            processors_used[processor] = processors_used.get(processor, 0) + 1
            
            quality_scores.append(result['extraction_info']['quality_score'])
            
            for chunk in result['chunks']:
                chunk_sizes.append(len(chunk['content']))
        
        return {
            'processors_used': processors_used,
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'total_chunks_created': len(chunk_sizes)
        }
    
    def search_chunks(self, 
                     query: str, 
                     regulation_type: str = None,
                     jurisdiction: str = None) -> List[Dict[str, Any]]:
        """Search through processed chunks (basic text search)"""
        
        matching_chunks = []
        
        # Load all processed files
        for json_file in self.output_dir.glob("*_processed.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for chunk in data['chunks']:
                    # Text search
                    if query.lower() in chunk['content'].lower():
                        # Filter by regulation type if specified
                        if regulation_type and chunk['metadata'].get('regulation_type') != regulation_type:
                            continue
                        
                        # Filter by jurisdiction if specified
                        if jurisdiction and chunk['metadata'].get('jurisdiction') != jurisdiction:
                            continue
                        
                        matching_chunks.append({
                            'chunk_id': chunk['chunk_id'],
                            'content': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                            'source': chunk['source_document'],
                            'section': chunk['section_title'],
                            'regulation_type': chunk['metadata'].get('regulation_type'),
                            'jurisdiction': chunk['metadata'].get('jurisdiction'),
                            'confidence': chunk['confidence_score']
                        })
            
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
        
        return matching_chunks

def main():
    """Example usage of the complete pipeline"""
    
    # Initialize pipeline (with Azure credentials if available)
    pipeline = EFIRASDocumentPipeline(
        # azure_endpoint="https://your-resource.cognitiveservices.azure.com/",
        # azure_key="your-key-here",
        output_dir="efiras_processed"
    )
    
    # Example 1: Process single document
    print("=== Processing Single Document ===")
    try:
        result = pipeline.process_single_document(
            file_path="data/regulatory_documents/lu/Lux_cssf18_698eng.pdf",
            preferred_processor=ProcessorType.PYMUPDF
        )
        
        print(f"✅ Successfully processed {result['source_file']}")
        print(f"   Processor: {result['extraction_info']['processor_used']}")
        print(f"   Quality: {result['extraction_info']['quality_score']:.2f}")
        print(f"   Chunks: {result['summary']['total_chunks']}")
        print(f"   Regulations: {result['summary']['regulation_types']}")
        
    except Exception as e:
        print(f"❌ Single document processing failed: {e}")
    
    # Example 2: Process batch of documents
    print("\n=== Processing Document Batch ===")
    document_files = [
        "mifid2_directive.pdf",
        "basel_iii_regulation.pdf",
        "gdpr_regulation.pdf"
    ]
    
    # try:
    #     batch_result = pipeline.process_document_batch(
    #         file_paths=document_files,
    #         preferred_processor=ProcessorType.AUTO
    #     )
        
    #     print(f"✅ Batch processing complete:")
    #     print(f"   Processed: {batch_result['processed_files']}/{len(document_files)} files")
    #     print(f"   Total chunks: {batch_result['total_chunks']}")
    #     print(f"   Regulations found: {list(batch_result['regulation_coverage']['regulations'].keys())}")
        
    # except Exception as e:
    #     print(f"❌ Batch processing failed: {e}")
    
    # Example 3: Search processed documents
    print("\n=== Searching Processed Documents ===")
    try:
        results = pipeline.search_chunks(
            query="capital requirements",
            regulation_type="Basel III"
        )
        
        print(f"Found {len(results)} matching chunks:")
        for result in results[:3]:  # Show first 3
            print(f"   📄 {result['source']} - {result['section']}")
            print(f"      {result['content']}")
            print()
    
    except Exception as e:
        print(f"❌ Search failed: {e}")

if __name__ == "__main__":
    main()