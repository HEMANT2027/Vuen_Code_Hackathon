# ---------------------------------------------
# AI Video Chat Assistant with RAG Integration
# ---------------------------------------------
from dotenv import load_dotenv
# ========== 0. ENVIRONMENT AND CONFIG ==========
load_dotenv()  # Load environment variables from .env file

# Set up keys and paths
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
TEMP_VIDEO_DIR = "temp_videos"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)  # Ensure temp dir exists

# ========== 1. CHROMA MEMORY SETUP ==========

# Setup ChromaDB client with Gemini embedding function
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)
    client = PersistentClient(path=CHROMA_DB_PATH)
    chat_history_collection = client.get_or_create_collection(
        name="video_chat_history",
        embedding_function=gemini_ef
    )
    print("✅ ChromaDB collection loaded successfully.")
except Exception as e:
    print(f"❌ FATAL: Error creating ChromaDB collection: {e}")
    chat_history_collection = None

# ========== 2. FASTAPI BACKEND SETUP ==========
app = FastAPI(title="AI Video Chat Assistant", description="Video analysis with RAG integration")

# API schema for chat response
class ChatResponse(BaseModel):
    response_text: str
    session_id: str

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ FATAL: API key configuration error: {e}")

# ========== 2A. RAG IMPORT HANDLING ==========

# Try importing RAG functions (gracefully degrade if unavailable)
try:
    from rag_integration import (
        add_to_rag_vectorstore, 
        query_rag_vectorstore, 
        get_vectorstore_stats,
        debug_add_test_data,
        force_reinitialize,
        get_context_for_query
    )
    RAG_AVAILABLE = True
    print("✅ RAG integration imported successfully")
except ImportError as e:
    print(f"⚠️ RAG integration not available: {e}")
    RAG_AVAILABLE = False
    # Dummy functions to avoid crashes
    def add_to_rag_vectorstore(*args, **kwargs): return True
    def query_rag_vectorstore(*args, **kwargs): return []
    def get_vectorstore_stats(): return {"error": "RAG not available"}
    def debug_add_test_data(): return 0
    def force_reinitialize(): return False
    def get_context_for_query(*args, **kwargs): return ""

# ========== 2B. UTILITY FUNCTIONS ==========

def extract_frames(video_path: str, fps: int = 1) -> List[Dict]:
    """Extract frames from video for analysis (returns Gemini-compatible format)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                frames.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffer).decode()
                    }
                })
        count += 1
    cap.release()
    return frames

def get_enhanced_context(prompt: str, session_id: str) -> str:
    """Fetch context from ChromaDB and (if available) RAG for grounding responses."""
    context_parts = []

    # Get session-specific history from ChromaDB
    try:
        if chat_history_collection:
            query_embedding = gemini_ef([prompt])[0]
            chroma_results = chat_history_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"session_id": session_id}
            )
            if chroma_results['documents'] and chroma_results['documents'][0]:
                recent_history = "\n".join(reversed(chroma_results['documents'][0]))
                context_parts.append(f"Recent conversation history:\n{recent_history}")
    except Exception as e:
        print(f"Warning: Could not query ChromaDB: {e}")

    # Get extra RAG context (if RAG is available)
    if RAG_AVAILABLE:
        try:
            rag_context = get_context_for_query(prompt, session_id)
            if rag_context:
                context_parts.append(f"Relevant knowledge from previous interactions:\n{rag_context}")
        except Exception as e:
            print(f"Warning: Could not query RAG vectorstore: {e}")

    return "\n".join(context_parts) + "\n---\n" if context_parts else ""

# ========== 2C. API ROUTES FOR CHAT AND RAG ==========

@app.post("/chat/video", response_model=ChatResponse)
async def chat_with_video(
    video_file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Handle video + prompt submission and return AI-generated response."""
    if not chat_history_collection or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Backend not initialized correctly.")

    session_id = session_id or str(uuid.uuid4())
    temp_video_path = os.path.join(TEMP_VIDEO_DIR, f"temp_{session_id}_{video_file.filename}")

    with open(temp_video_path, "wb") as f:
        f.write(await video_file.read())

    try:
        frames = extract_frames(temp_video_path, fps=1)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")

        context_history = get_enhanced_context(prompt, session_id)
        full_prompt = f"{context_history}Analyze this video and answer the user's question: {prompt}"
        content = [{"role": "user", "parts": [{"text": full_prompt}] + frames}]

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=content)
        ai_response = response.text

        # Save conversation to ChromaDB
        chat_history_collection.add(
            documents=[f"User: {prompt}\nAssistant: {ai_response}"],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )

        # Save context to RAG (if available)
        if RAG_AVAILABLE:
            try:
                add_to_rag_vectorstore(f"Video analysis for '{video_file.filename}': {ai_response}", session_id, "video_analysis", "video")
                add_to_rag_vectorstore(f"User asked about video '{video_file.filename}': {prompt}", session_id, "user_query", "video")
            except Exception as e:
                print(f"Warning: Could not store in RAG: {e}")

        return ChatResponse(response_text=ai_response, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text_only(prompt: str = Form(...), session_id: Optional[str] = Form(None)):
    """Handle follow-up text-only queries and return response."""
    if not chat_history_collection or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Backend not initialized correctly.")

    session_id = session_id or str(uuid.uuid4())
    context_history = get_enhanced_context(prompt, session_id)
    full_prompt = f"{context_history}User's follow-up question: {prompt}"

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=[{"role": "user", "parts": [{"text": full_prompt}]}])
        ai_response = response.text

        chat_history_collection.add(
            documents=[f"User: {prompt}\nAssistant: {ai_response}"],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )

        if RAG_AVAILABLE and len(ai_response) > 100:
            try:
                add_to_rag_vectorstore(f"Q: {prompt}\nA: {ai_response}", session_id, "ai_response", "chat")
            except Exception as e:
                print(f"Warning: Could not store in RAG: {e}")

        return ChatResponse(response_text=ai_response, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")

# Additional endpoints for RAG querying, stats, and debugging
@app.post("/rag/query")
async def query_rag_knowledge(query: str = Form(...), session_id: Optional[str] = Form(None), k: int = Form(5)):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        results = query_rag_vectorstore(query, session_id, k)
        return {
            "query": query,
            "session_id": session_id,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query error: {e}")

@app.get("/rag/stats")
async def get_rag_stats():
    if not RAG_AVAILABLE:
        return {"error": "RAG system not available", "rag_available": False}
    try:
        stats = get_vectorstore_stats()
        stats["rag_available"] = True
        return stats
    except Exception as e:
        return {"error": str(e), "rag_available": False}

@app.post("/rag/debug")
async def debug_rag_system():
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        test_count = debug_add_test_data()
        stats = get_vectorstore_stats()
        return {
            "test_data_added": test_count,
            "stats": stats,
            "message": "Debug operation completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {e}")

@app.post("/rag/reinitialize")
async def reinitialize_rag():
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        success = force_reinitialize()
        stats = get_vectorstore_stats()
        return {
            "success": success,
            "stats": stats,
            "message": "RAG system reinitialized" if success else "Reinitialization failed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reinitialize error: {e}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "chromadb_available": chat_history_collection is not None,
        "rag_available": RAG_AVAILABLE,
        "temp_dir_exists": os.path.exists(TEMP_VIDEO_DIR)
    }

# ========== 3. GRADIO FRONTEND ==========
# (No comments added here as it's already well-documented inline.)

# ========== 4. MAIN EXECUTION BLOCK ==========
if __name__ == "__main__":
    # Start FastAPI backend in separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    rag_status_msg = "with RAG Integration" if RAG_AVAILABLE else "(RAG Unavailable - Install: pip install langchain-community sentence-transformers faiss-cpu)"

    print("🚀 Starting AI Video Chat Assistant...")
    print(f"📊 FastAPI Backend: http://127.0.0.1:8000")
    print(f"🎨 Gradio Frontend: http://127.0.0.1:7860")
    print(f"🧠 RAG System: {rag_status_msg}")
    print("=" * 70)
    if not RAG_AVAILABLE:
        print("⚠️  WARNING: RAG system is not available!")
        print("   Install dependencies with: pip install langchain-community sentence-transformers faiss-cpu")
        print("   The application will work without RAG but with limited knowledge persistence.")
        print("=" * 70)

    # Launch Gradio Interface
    launch_gradio_ui()
