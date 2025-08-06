# app/rag_integration.py

import os
import logging
from typing import Optional, List, Dict, Union
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RAG_FAISS_PATH = os.path.join(os.getcwd(), "rag_data")
RAG_METADATA_PATH = os.path.join(RAG_FAISS_PATH, "metadata.pkl")
RAG_DEBUG_PATH = os.path.join(RAG_FAISS_PATH, "debug_info.json")
os.makedirs(RAG_FAISS_PATH, exist_ok=True)

# Initialize text splitter for large documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Initialize embeddings
embedding = None
vectorstore = None
metadata_store = {}
debug_info = {"initialization_attempts": 0, "last_error": None, "documents_added": 0}

def initialize_embeddings():
    """Initialize HuggingFace embeddings with error handling."""
    global embedding
    
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info("HuggingFace embeddings initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        debug_info["last_error"] = f"Embedding initialization failed: {str(e)}"
        return False

def initialize_vectorstore():
    """Initialize the FAISS vectorstore and metadata store with debugging."""
    global vectorstore, metadata_store, debug_info
    
    debug_info["initialization_attempts"] += 1
    
    if not initialize_embeddings():
        return False
    
    try:
        # Load metadata store first
        if os.path.exists(RAG_METADATA_PATH):
            with open(RAG_METADATA_PATH, 'rb') as f:
                metadata_store = pickle.load(f)
            logger.info(f"Metadata store loaded with {len(metadata_store)} entries")
        else:
            metadata_store = {}
            logger.info("New metadata store created")
        
        # Check if vectorstore files exist
        faiss_index_path = os.path.join(RAG_FAISS_PATH, "index.faiss")
        faiss_pkl_path = os.path.join(RAG_FAISS_PATH, "index.pkl")
        
        if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
            try:
                vectorstore = FAISS.load_local(
                    RAG_FAISS_PATH, 
                    embeddings=embedding, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Existing FAISS vectorstore loaded with {vectorstore.index.ntotal} vectors")
                
                # Add some sample data if vectorstore is empty
                if vectorstore.index.ntotal <= 1:  # Only has dummy document
                    add_sample_data()
                
            except Exception as load_error:
                logger.warning(f"Could not load existing vectorstore: {load_error}")
                vectorstore = create_new_vectorstore()
        else:
            logger.info("No existing vectorstore found, creating new one")
            vectorstore = create_new_vectorstore()
        
        # Save debug info
        debug_info["vectorstore_documents"] = vectorstore.index.ntotal if vectorstore else 0
        debug_info["last_initialization"] = datetime.now().isoformat()
        debug_info["last_error"] = None
        save_debug_info()
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to initialize vectorstore: {e}"
        logger.error(error_msg)
        debug_info["last_error"] = error_msg
        save_debug_info()
        return False

def create_new_vectorstore():
    """Create a new FAISS vectorstore with sample data."""
    global vectorstore
    
    try:
        # Create sample documents for initialization
        sample_docs = [
            Document(
                page_content="This is the RAG knowledge system for AI Video Chat Assistant.",
                metadata={"type": "system", "content_type": "initialization", "timestamp": datetime.now().isoformat()}
            ),
            Document(
                page_content="The system can analyze videos and answer questions about their content.",
                metadata={"type": "system", "content_type": "capability", "timestamp": datetime.now().isoformat()}
            )
        ]
        
        vectorstore = FAISS.from_documents(sample_docs, embedding)
        logger.info(f"New FAISS vectorstore created with {len(sample_docs)} sample documents")
        
        # Save immediately
        save_vectorstore()
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to create new vectorstore: {e}")
        raise e

def add_sample_data():
    """Add sample data to help with testing."""
    sample_entries = [
        {
            "text": "Video analysis example: A user uploaded a video showing a cat playing with a toy mouse. The video had good lighting and clear audio.",
            "content_type": "video_analysis",
            "session_id": "sample_session"
        },
        {
            "text": "User frequently asks about video quality, object detection, and scene analysis in uploaded content.",
            "content_type": "user_pattern",
            "session_id": "sample_session"
        },
        {
            "text": "The AI assistant can identify objects, analyze scenes, describe actions, and answer questions about video content using computer vision.",
            "content_type": "capability",
            "session_id": "global"
        }
    ]
    
    for entry in sample_entries:
        add_to_rag_vectorstore(
            text=entry["text"],
            session_id=entry["session_id"],
            content_type=entry["content_type"],
            source="sample_data"
        )
    
    logger.info(f"Added {len(sample_entries)} sample entries to vectorstore")

def save_vectorstore():
    """Save the vectorstore and metadata to disk."""
    try:
        if vectorstore is not None:
            vectorstore.save_local(RAG_FAISS_PATH)
            logger.debug("Vectorstore saved successfully")
        
        with open(RAG_METADATA_PATH, 'wb') as f:
            pickle.dump(metadata_store, f)
        
        debug_info["last_save"] = datetime.now().isoformat()
        save_debug_info()
        
        return True
    except Exception as e:
        error_msg = f"Failed to save vectorstore: {e}"
        logger.error(error_msg)
        debug_info["last_error"] = error_msg
        save_debug_info()
        return False

def save_debug_info():
    """Save debug information."""
    try:
        with open(RAG_DEBUG_PATH, 'w') as f:
            json.dump(debug_info, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save debug info: {e}")

def add_to_rag_vectorstore(
    text: str, 
    session_id: Optional[str] = None, 
    content_type: str = "general",
    source: str = "chat",
    chunk_text: bool = True
) -> bool:
    """Add text to the RAG vectorstore with enhanced metadata and debugging."""
    global debug_info
    
    if vectorstore is None:
        logger.error("Vectorstore not initialized")
        debug_info["last_error"] = "Add operation failed: vectorstore not initialized"
        return False
    
    if not text or not text.strip():
        logger.warning("Empty text provided, skipping")
        return False
    
    try:
        # Prepare metadata
        metadata = {
            "session_id": session_id or "global",
            "content_type": content_type,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "char_count": len(text)
        }
        
        # Split text into chunks if needed
        if chunk_text and len(text) > 500:
            chunks = text_splitter.split_text(text)
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        else:
            documents = [Document(page_content=text, metadata=metadata)]
        
        # Add to vectorstore
        vectorstore.add_documents(documents)
        
        # Update metadata store
        doc_id = f"{session_id}_{datetime.now().timestamp()}"
        metadata_store[doc_id] = {
            "metadata": metadata,
            "document_count": len(documents),
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        }
        
        # Update debug info
        debug_info["documents_added"] += len(documents)
        debug_info["total_documents"] = vectorstore.index.ntotal
        debug_info["last_add_operation"] = datetime.now().isoformat()
        
        # Save to disk
        save_vectorstore()
        
        logger.info(f"Successfully added {len(documents)} document(s) to RAG vectorstore")
        return True
        
    except Exception as e:
        error_msg = f"Failed to add to vectorstore: {e}"
        logger.error(error_msg)
        debug_info["last_error"] = error_msg
        save_debug_info()
        return False

def query_rag_vectorstore(
    query: str, 
    session_id: Optional[str] = None, 
    k: int = 5,
    content_type_filter: Optional[str] = None,
    similarity_threshold: float = 0.0
) -> List[Document]:
    """Query the RAG vectorstore with enhanced filtering and debugging."""
    global debug_info
    
    if vectorstore is None:
        logger.error("Vectorstore not initialized for query")
        debug_info["last_error"] = "Query failed: vectorstore not initialized"
        return []
    
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return []
    
    try:
        logger.info(f"Querying vectorstore with query: '{query[:50]}...' (total docs: {vectorstore.index.ntotal})")
        
        # First try a simple similarity search without filters
        all_results = vectorstore.similarity_search_with_score(query, k=k*2)
        
        if not all_results:
            logger.warning("No results found for query")
            debug_info["last_query_results"] = 0
            return []
        
        logger.info(f"Found {len(all_results)} initial results")
        
        # Apply filters manually since FAISS filtering can be unreliable
        filtered_results = []
        for doc, score in all_results:
            doc_metadata = doc.metadata
            
            # Apply session filter
            if session_id and doc_metadata.get("session_id") != session_id:
                continue
            
            # Apply content type filter
            if content_type_filter and doc_metadata.get("content_type") != content_type_filter:
                continue
            
            # Apply similarity threshold
            if score < similarity_threshold:
                continue
            
            filtered_results.append(doc)
            if len(filtered_results) >= k:
                break
        
        debug_info["last_query"] = query[:100]
        debug_info["last_query_results"] = len(filtered_results)
        debug_info["last_query_time"] = datetime.now().isoformat()
        save_debug_info()
        
        logger.info(f"Retrieved {len(filtered_results)} filtered documents for query")
        return filtered_results
        
    except Exception as e:
        error_msg = f"Failed to query vectorstore: {e}"
        logger.error(error_msg)
        debug_info["last_error"] = error_msg
        save_debug_info()
        return []

def get_vectorstore_stats() -> Dict:
    """Get comprehensive statistics about the vectorstore."""
    try:
        stats = {
            "status": "operational" if vectorstore is not None else "failed",
            "total_documents": vectorstore.index.ntotal if vectorstore else 0,
            "total_entries": len(metadata_store),
            "debug_info": debug_info.copy()
        }
        
        if metadata_store:
            # Count by session and content type
            session_counts = {}
            content_type_counts = {}
            
            for doc_id, data in metadata_store.items():
                metadata = data.get('metadata', {})
                session = metadata.get('session_id', 'unknown')
                content_type = metadata.get('content_type', 'unknown')
                
                session_counts[session] = session_counts.get(session, 0) + data.get('document_count', 1)
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + data.get('document_count', 1)
            
            stats.update({
                "sessions": len(session_counts),
                "session_breakdown": session_counts,
                "content_type_breakdown": content_type_counts,
            })
        
        stats.update({
            "vectorstore_path": RAG_FAISS_PATH,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "files_exist": {
                "index.faiss": os.path.exists(os.path.join(RAG_FAISS_PATH, "index.faiss")),
                "index.pkl": os.path.exists(os.path.join(RAG_FAISS_PATH, "index.pkl")),
                "metadata.pkl": os.path.exists(RAG_METADATA_PATH)
            }
        })
        
        return stats
        
    except Exception as e:
        return {"error": str(e), "debug_info": debug_info.copy()}

def debug_add_test_data():
    """Add test data for debugging purposes."""
    test_entries = [
        "Test entry 1: This is a sample video analysis about a cooking tutorial.",
        "Test entry 2: User asked about ingredients in the recipe video.",
        "Test entry 3: The AI identified tomatoes, onions, and garlic in the cooking video.",
        "Test entry 4: Analysis of a nature documentary showing wildlife behavior.",
        "Test entry 5: User inquiry about animal species identification in nature videos."
    ]
    
    success_count = 0
    for i, entry in enumerate(test_entries):
        if add_to_rag_vectorstore(
            text=entry,
            session_id=f"test_session_{i % 2}",
            content_type="test_data",
            source="debug"
        ):
            success_count += 1
    
    logger.info(f"Debug: Added {success_count}/{len(test_entries)} test entries")
    return success_count

def force_reinitialize():
    """Force reinitialize the vectorstore (useful for debugging)."""
    global vectorstore, metadata_store, debug_info
    
    logger.info("Force reinitializing RAG system...")
    
    # Clear current state
    vectorstore = None
    metadata_store = {}
    debug_info["force_reinit_count"] = debug_info.get("force_reinit_count", 0) + 1
    
    # Reinitialize
    success = initialize_vectorstore()
    
    if success:
        # Add test data
        debug_add_test_data()
        logger.info("Force reinitialization completed successfully")
    else:
        logger.error("Force reinitialization failed")
    
    return success

# Initialize vectorstore on module import
logger.info("Initializing RAG integration module...")
initialize_success = initialize_vectorstore()

if not initialize_success:
    logger.error("Failed to initialize RAG vectorstore. Attempting force reinitialization...")
    initialize_success = force_reinitialize()

if initialize_success:
    logger.info("RAG integration module loaded successfully")
else:
    logger.error("RAG integration module failed to load properly. Some features may not work.")

# Convenience functions remain the same...
def add_video_analysis(video_filename: str, analysis: str, session_id: str) -> bool:
    """Convenience function to add video analysis to RAG."""
    content = f"Video Analysis for '{video_filename}': {analysis}"
    return add_to_rag_vectorstore(
        text=content,
        session_id=session_id,
        content_type="video_analysis",
        source="video"
    )

def get_context_for_query(query: str, session_id: str) -> str:
    """Get formatted context for a query."""
    try:
        # Get session-specific context
        session_docs = query_rag_vectorstore(query, session_id, k=3)
        # Get global context  
        global_docs = query_rag_vectorstore(query, None, k=2)
        
        context_parts = []
        
        if session_docs:
            session_context = "\n".join([doc.page_content for doc in session_docs])
            context_parts.append(f"Session Context:\n{session_context}")
        
        if global_docs:
            global_context = "\n".join([doc.page_content for doc in global_docs])
            context_parts.append(f"Global Knowledge:\n{global_context}")
        
        if context_parts:
            return "\n---\n".join(context_parts) + "\n---\n"
        
        return ""
        
    except Exception as e:
        logger.error(f"Failed to get context for query: {e}")
        return ""