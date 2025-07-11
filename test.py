import json
with open("data_processed/Lux_cssf18_698eng_embeddings_local.json", 'r') as f:
    embeddings = json.load(f)

# Find the 517 embedding
for emb in embeddings:
    if "517." in emb['content']:
        print("=== FOUND 517 IN EMBEDDINGS ===")
        print(f"Content length: {len(emb['content'])}")
        print(f"Content: {emb['content']}")
        print("=== END ===")
        break