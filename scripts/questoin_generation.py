import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("GPT_API_KEY")

# Set up OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

def generate_smart_questions(chunks):
   """
   Generate one question per chunk if the chunk contains meaningful content.
   Returns list of {chunk_id, question} pairs.
   """
   
   # Prepare the prompt with all chunks
   chunks_text = ""
   for chunk in chunks:
       chunks_text += f"ID: {chunk['chunk_id']}\nText: {chunk['text']}\n\n"
   prompt = f"""For each chunk below, generate high-quality questions ONLY if the text contains meaningful content that can be answered. Skip chunks that are just titles, headers, or too short/meaningless.

            IMPORTANT GUIDELINES:
            - Number of questions should be flexible based on content richness
            - Short chunks with basic info: 1-2 questions
            - Medium to  long chunks: 3-5 questions  
            - Each question must be meaningful and unique (no repetition)
            - Questions should cover different aspects: factual, procedural, analytical, compliance-related
            - Only generate questions that can actually be answered from the given text

            Return format for each valid chunk:
            ID: [chunk_id]
            Question: [your question]
            Question: [your question]
            (continue as needed based on content richness)

            Chunks:
            {chunks_text}

            Response:"""
   
   response = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": prompt}],
       max_tokens=2000,
       temperature=0.7
   )
   
   # Parse the response
   result = []
   lines = response.choices[0].message.content.strip().split('\n')
   
   current_id = None
   for line in lines:
       line = line.strip()
       if line.startswith('ID:'):
           current_id = line.replace('ID:', '').strip()
       elif line.startswith('Question:') and current_id:
           question = line.replace('Question:', '').strip()
           result.append({
               'chunk_id': current_id,
               'question': question
           })
           current_id = None
   
   return result

def process_chunks_in_batches(chunks, batch_size=15):
    all_results = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"Processing batch {batch_num}/{total_batches}: chunks {i+1} to {min(i+batch_size, len(chunks))}")
        
        try:
            batch_results = generate_smart_questions(batch)
            # Print each result from this batch
            for item in batch_results:
                print(f"Chunk {item['chunk_id']}: {item['question']}")  # ← CORRECT
            
            all_results.extend(batch_results)
            print(f"✅ Batch {batch_num} completed: {len(batch_results)} questions generated")
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            continue
    
    return all_results



def save_questions_to_file(questions_output, filename="generated_questions.json"):
   """Save questions to JSON file"""
   with open(filename, 'w', encoding='utf-8') as f:
       json.dump(questions_output, f, indent=2, ensure_ascii=False)
   print(f"✅ Questions saved to {filename}")

def save_questions_to_txt(questions_output, filename="generated_questions.txt"):
   """Save questions to readable text file"""
   with open(filename, 'w', encoding='utf-8') as f:
       for item in questions_output:
           f.write(f"Chunk ID: {item['chunk_id']}\n")
           f.write(f"Question: {item['question']}\n")
           f.write("-" * 50 + "\n")
   print(f"✅ Questions saved to {filename}")


# Usage example
import json
with open("data/data_processed/Lux_cssf18_698eng_chunks.json", 'r') as f:
    chunks = json.load(f)


# Generate questions
# chunks = chunks[30:32]
# questions_output = generate_smart_questions(chunks)

# # Save to both formats
# save_questions_to_file(questions_output, "questions.json")
# save_questions_to_txt(questions_output, "questions.txt")

# # Print to console too
# print("\nGenerated Questions:")
# for item in questions_output:
#    print(f"Chunk {item['chunk_id']}: {item['question']}")





# Usage for 300 chunks
print(f"Total chunks to process: {len(chunks)}")
# chunks = chunks["chunks"][20:30]
chunks = chunks["chunks"]
questions_output = process_chunks_in_batches(chunks, batch_size=10)
print(questions_output)

# Save results
save_questions_to_file(questions_output, "all_questions.json")
save_questions_to_txt(questions_output, "all_questions.txt")

print(f"\n🎉 Completed! Generated questions for {len(questions_output)} chunks")