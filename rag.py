#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) module for Circadia.

This provides the "factual memory" layer that complements LoRA fine-tuning:
- LoRA learns: style, preferences, patterns (compressed into weights)
- RAG retrieves: specific facts, conversations, details (stored verbatim)

Together they mimic biological memory:
- LoRA = long-term procedural memory (how to do things)
- RAG = episodic memory (what happened)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings


DEFAULT_MEMORY_DIR = Path("data/rag_memory")
DEFAULT_COLLECTION = "conversations"


class ConversationMemory:
    """Vector store for conversation retrieval."""
    
    def __init__(
        self, 
        persist_dir: Path = DEFAULT_MEMORY_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection (uses default embedding function)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Circadia conversation memory"}
        )
    
    def add_conversation(
        self, 
        messages: List[Dict[str, str]], 
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add a conversation to the memory store.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."} dicts
            conversation_id: Optional ID (auto-generated if not provided)
            metadata: Optional metadata (timestamp, tags, etc.)
        
        Returns:
            The conversation ID
        """
        # Format conversation as searchable text
        text = self._format_conversation(messages)
        
        # Generate ID if not provided
        if conversation_id is None:
            conversation_id = f"conv_{self.collection.count()}"
        
        # Prepare metadata
        meta = metadata or {}
        meta["num_turns"] = len(messages)
        meta["messages_json"] = json.dumps(messages)  # Store full conversation
        
        # Add to collection
        self.collection.add(
            documents=[text],
            ids=[conversation_id],
            metadatas=[meta],
        )
        
        return conversation_id
    
    def search(
        self, 
        query: str, 
        n_results: int = 3,
        min_relevance: float = 0.0,
    ) -> List[Dict]:
        """
        Search for relevant conversations.
        
        Args:
            query: The search query (usually the user's current message)
            n_results: Maximum number of results to return
            min_relevance: Minimum relevance score (0-1, higher = more relevant)
        
        Returns:
            List of conversation dicts with 'messages', 'relevance', and 'id'
        """
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )
        
        conversations = []
        for i, (doc_id, distance, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            # Convert distance to relevance (ChromaDB uses L2 distance by default)
            # Lower distance = more relevant, so we invert
            relevance = 1 / (1 + distance)
            
            if relevance < min_relevance:
                continue
            
            # Parse stored messages
            messages = json.loads(metadata.get("messages_json", "[]"))
            
            conversations.append({
                "id": doc_id,
                "messages": messages,
                "relevance": round(relevance, 3),
                "document": results["documents"][0][i],
            })
        
        return conversations
    
    def ingest_jsonl(self, jsonl_path: Path, clear_existing: bool = False) -> int:
        """
        Bulk ingest conversations from a JSONL file (like memories.jsonl).
        
        Args:
            jsonl_path: Path to JSONL file with {"messages": [...]} records
            clear_existing: If True, clear existing data first
        
        Returns:
            Number of conversations ingested
        """
        if clear_existing:
            # Delete and recreate collection
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "Circadia conversation memory"}
            )
        
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        self.add_conversation(obj["messages"])
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        return count
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for embedding/search."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    
    def count(self) -> int:
        """Return number of stored conversations."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all stored conversations."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Circadia conversation memory"}
        )


def format_context_prompt(
    user_query: str,
    retrieved: List[Dict],
    max_context_chars: int = 2000,
) -> str:
    """
    Format retrieved conversations as context for the LLM.
    
    Args:
        user_query: The user's current question
        retrieved: List of retrieved conversation dicts
        max_context_chars: Maximum characters of context to include
    
    Returns:
        Formatted prompt with context
    """
    if not retrieved:
        return user_query
    
    context_parts = []
    total_chars = 0
    
    for conv in retrieved:
        conv_text = f"[Previous conversation (relevance: {conv['relevance']})]\n"
        for msg in conv["messages"]:
            line = f"  {msg['role']}: {msg['content']}\n"
            conv_text += line
        
        if total_chars + len(conv_text) > max_context_chars:
            break
        
        context_parts.append(conv_text)
        total_chars += len(conv_text)
    
    if not context_parts:
        return user_query
    
    context = "\n".join(context_parts)
    
    return f"""Here are some relevant previous conversations for context:

{context}

Now, please respond to the following:
{user_query}"""


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG memory management")
    parser.add_argument("command", choices=["ingest", "search", "count", "clear"])
    parser.add_argument("--file", type=Path, help="JSONL file to ingest")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--n", type=int, default=3, help="Number of results")
    args = parser.parse_args()
    
    memory = ConversationMemory()
    
    if args.command == "ingest":
        if not args.file:
            print("Error: --file required for ingest")
            exit(1)
        count = memory.ingest_jsonl(args.file)
        print(f"Ingested {count} conversations from {args.file}")
        print(f"Total conversations in memory: {memory.count()}")
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query required for search")
            exit(1)
        results = memory.search(args.query, n_results=args.n)
        print(f"Found {len(results)} relevant conversations:\n")
        for r in results:
            print(f"[{r['id']}] (relevance: {r['relevance']})")
            for msg in r["messages"]:
                print(f"  {msg['role']}: {msg['content'][:100]}...")
            print()
    
    elif args.command == "count":
        print(f"Conversations in memory: {memory.count()}")
    
    elif args.command == "clear":
        memory.clear()
        print("Memory cleared")
