import uuid
from pinecone import Pinecone

# --- Hardcoded Credentials ---
PINECONE_API_KEY = "PINECONE_API_KEY"
PINECONE_INDEX_NAME = "email-agent-memory"

class PineconeMemory:
    def __init__(self):
        if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY_HERE":
            print("Warning: PINECONE_API_KEY is not set correctly.")
            
        self.index_name = PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

    def _ensure_index(self):
        """Checks if index exists, creates it using Pinecone's integrated model if not."""
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating index '{self.index_name}' with integrated embedding model...")
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "multilingual-e5-large", 
                    "field_map": {"text": "chunk_text"}
                }
            )

    def add_memory(self, user_id: str, text: str):
        """Adds memory using text-based upsert."""
        self._ensure_index()
        memory_id = str(uuid.uuid4())
        index = self.pc.Index(self.index_name)
        
        record = {
            "id": memory_id,
            "chunk_text": text,
            "text": text,       # <-- FIX: Re-added the 'text' key here
            "user_id": user_id,
        }
        
        try:
            index.upsert_records(
                namespace="default", 
                records=[record]
            )
            print(f"Memory saved for user {user_id}")
        except Exception as e:
            print(f"Failed to save memory: {e}")

    def search_memory(self, user_id: str, query: str, k: int = 2) -> list:
        """Searches memory using text query."""
        self._ensure_index()
        index = self.pc.Index(self.index_name)
        
        try:
            # FIX: 'filter' is now inside the 'query' dictionary
            resp = index.search_records(
                namespace="default",
                query={
                    "inputs": {"text": query}, 
                    "top_k": k,
                    "filter": {"user_id": {"$eq": user_id}}
                }
            )
            
            memories = []
            # Extract text (handling Pinecone's response object structure)
            result_data = resp.get('result', {}) if isinstance(resp, dict) else getattr(resp, 'result', {})
            hits = result_data.get('hits', []) if isinstance(result_data, dict) else getattr(result_data, 'hits', [])
            
            for hit in hits:
                fields = hit.get('fields', {}) if isinstance(hit, dict) else getattr(hit, 'fields', {})
                if 'chunk_text' in fields:
                    memories.append(fields['chunk_text'])
            
            return memories
            
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []
