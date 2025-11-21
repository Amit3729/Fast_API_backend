import os

# Define the file structure
structure = {
    "app/routers/upload.py": "",
    "app/routers/rag.py": "",
    "app/services/chunker.py": "",
    "app/services/embeddings.py": "", 
    "app/services/vector_store.py": "",
    "app/services/db.py": "",
    "app/services/redis_memory.py": "",
    "app/models/schemas.py": "",
    "app/utils/logger.py": "",
    "tests/test_chunker.py": "",
    "tests/test_booking_extractor.py": "",
    "main.py": "",
    ".env": ""
}

# Create directories and files
for filepath in structure:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        open(filepath, 'a').close()
        print(f"Created: {filepath}")
    else:
        print(f"Exists: {filepath}")

print("✅ Project structure ready!")