# File: app/main.py

import os
import io
import base64
import cv2
import httpx
import uuid
import threading
import uvicorn
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import gradio as gr
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from chromadb import PersistentClient
import numpy as np
from dotenv import load_dotenv

# Import RAG functions - with error handling
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
    # Define dummy functions
    def add_to_rag_vectorstore(*args, **kwargs): return True
    def query_rag_vectorstore(*args, **kwargs): return []
    def get_vectorstore_stats(): return {"error": "RAG not available"}
    def debug_add_test_data(): return 0
    def force_reinitialize(): return False
    def get_context_for_query(*args, **kwargs): return ""

# --- 0. Environment and Configuration ---
load_dotenv() # Load variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
TEMP_VIDEO_DIR = "temp_videos"

# Create temporary directory if it doesn't exist
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# --- 1. The Memory Core: ChromaDB Setup ---
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

# --- 2. The Robust Backend: FastAPI Setup ---
app = FastAPI(title="AI Video Chat Assistant", description="Video analysis with RAG integration")

# Pydantic data models for API communication
class ChatResponse(BaseModel):
    response_text: str
    session_id: str

# --- Workstream A: Backend and AI Logic ---

# Initialize the Gemini API client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ FATAL: API key configuration error: {e}")

def extract_frames(video_path: str, fps: int = 1) -> List[Dict]:
    """Extracts frames and returns them as a list of Gemini API-compatible parts."""
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
    """
    Enhanced context retrieval using both ChromaDB and RAG FAISS vectorstore.
    Combines conversation history with relevant knowledge from RAG.
    """
    context_parts = []
    
    # Get ChromaDB conversation history
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
    
    # Get RAG knowledge context (only if RAG is available)
    if RAG_AVAILABLE:
        try:
            rag_context = get_context_for_query(prompt, session_id)
            if rag_context:
                context_parts.append(f"Relevant knowledge from previous interactions:\n{rag_context}")
        except Exception as e:
            print(f"Warning: Could not query RAG vectorstore: {e}")
    
    # Combine contexts
    if context_parts:
        return "\n".join(context_parts) + "\n---\n"
    return ""

@app.post("/chat/video", response_model=ChatResponse)
async def chat_with_video(
    video_file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Processes a video and prompt, using ChromaDB for memory and RAG for enhanced knowledge."""
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

        # Get enhanced context from both ChromaDB and RAG
        context_history = get_enhanced_context(prompt, session_id)
        
        full_prompt = f"{context_history}Analyze this video and answer the user's question: {prompt}"
        content = [{"role": "user", "parts": [{"text": full_prompt}] + frames}]

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=content)
        ai_response = response.text

        # Store in ChromaDB for conversation history
        conversation_entry = f"User: {prompt}\nAssistant: {ai_response}"
        chat_history_collection.add(
            documents=[conversation_entry],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )
        
        # Store in RAG vectorstore for knowledge persistence (only if available)
        if RAG_AVAILABLE:
            try:
                video_analysis_text = f"Video analysis for '{video_file.filename}': {ai_response}"
                add_to_rag_vectorstore(video_analysis_text, session_id, "video_analysis", "video")
                
                user_context = f"User asked about video '{video_file.filename}': {prompt}"
                add_to_rag_vectorstore(user_context, session_id, "user_query", "video")
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
    """Handles text-only follow-up questions with enhanced RAG context."""
    if not chat_history_collection or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Backend not initialized correctly.")

    session_id = session_id or str(uuid.uuid4())

    # Get enhanced context from both ChromaDB and RAG
    context_history = get_enhanced_context(prompt, session_id)
    
    full_prompt = f"{context_history}User's follow-up question: {prompt}"
    content = [{"role": "user", "parts": [{"text": full_prompt}]}]

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=content)
        ai_response = response.text

        # Store in ChromaDB for conversation history
        conversation_entry = f"User: {prompt}\nAssistant: {ai_response}"
        chat_history_collection.add(
            documents=[conversation_entry],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )
        
        # Store important responses in RAG for future knowledge (only if available)
        if RAG_AVAILABLE and len(ai_response) > 100:
            try:
                add_to_rag_vectorstore(f"Q: {prompt}\nA: {ai_response}", session_id, "ai_response", "chat")
            except Exception as e:
                print(f"Warning: Could not store in RAG: {e}")
        
        return ChatResponse(response_text=ai_response, session_id=session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")

@app.post("/rag/query")
async def query_rag_knowledge(query: str = Form(...), session_id: Optional[str] = Form(None), k: int = Form(5)):
    """Direct RAG knowledge query endpoint for testing and debugging."""
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
    """Get RAG system statistics."""
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
    """Debug endpoint to add test data and reinitialize if needed."""
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
    """Force reinitialize the RAG system."""
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "chromadb_available": chat_history_collection is not None,
        "rag_available": RAG_AVAILABLE,
        "temp_dir_exists": os.path.exists(TEMP_VIDEO_DIR)
    }

# --- 3. The Interactive Frontend: Enhanced Gradio UI ---

def launch_gradio_ui():
    """Launches the enhanced Gradio interface with modern UI design and RAG integration."""

    def handle_chat_submission(user_input: str, history: List, session_id: str, video_path: str):
        """
        Main function to handle user submissions (text with or without video).
        It calls the appropriate FastAPI backend endpoint.
        """
        history = history or []
        
        if not user_input.strip():
            return history, gr.update(value=""), gr.update(interactive=True), gr.update(interactive=True)
        
        # Append user message to chat history immediately for better UX
        history.append({"role": "user", "content": user_input})
        status_message = "🤔 Analyzing with AI and retrieving relevant context..." if RAG_AVAILABLE else "🤔 Analyzing with AI..."
        history.append({"role": "assistant", "content": status_message})
        yield history, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False)

        output_message = ""
        try:
            with httpx.Client(timeout=180.0) as client:
                if video_path:
                    # User submitted a video with the prompt
                    with open(video_path, "rb") as f:
                        files = {'video_file': (os.path.basename(video_path), f, 'video/mp4')}
                        data = {'prompt': user_input, 'session_id': session_id}
                        response = client.post("https://vuen-code-hackathon.onrender.com/chat/video", files=files, data=data)
                else:
                    # User submitted a text-only follow-up
                    data = {'prompt': user_input, 'session_id': session_id}
                    response = client.post("https://vuen-code-hackathon.onrender.com/chat/text", data=data)

                response.raise_for_status()
                chat_response = response.json()
                output_message = chat_response['response_text']

        except httpx.HTTPStatusError as e:
            output_message = f"❌ Error: Failed to get response from server. Status {e.response.status_code}."
        except httpx.RequestError:
            output_message = f"❌ Error: Network request failed. Please check if the backend server is running."
        except Exception as e:
            output_message = f"❌ An unexpected error occurred: {str(e)}"

        history[-1]["content"] = output_message
        yield history, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def handle_video_upload(video_file, session_id):
        """Handles the video upload event and updates the preview."""
        if video_file:
            print(f"Video uploaded: {video_file.name} for session {session_id}")
            return (
                video_file.name,  # video_path_state
                gr.update(value=video_file.name),  # Update video player
                gr.update(interactive=True, placeholder="✨ Video loaded! Ask me anything..."),
                gr.update(visible=True, value=f"📹 **{os.path.basename(video_file.name)}** uploaded and ready for AI analysis!")
            )
        return (
            None, 
            gr.update(value=None), 
            gr.update(interactive=False, placeholder="📁 Upload a video to start..."),
            gr.update(visible=False)
        )

    def clear_session():
        """Clears all UI components and generates a new session ID."""
        new_session_id = str(uuid.uuid4())
        print(f"New session started: {new_session_id}")
        return (
            [],  # Clear chatbot
            None, # Clear video player
            None, # Clear video path state
            new_session_id, # Set new session ID
            gr.update(placeholder="📁 Upload a video to begin...", interactive=False), # Reset textbox
            gr.update(visible=False)  # Hide upload status
        )

    def query_knowledge_base(query, session_id):
        """Query the RAG knowledge base directly for testing."""
        if not RAG_AVAILABLE:
            return "❌ **RAG System Not Available**\n\nThe RAG (Retrieval-Augmented Generation) system is not properly initialized. Please check that you have installed the required dependencies:\n\n```bash\npip install langchain-community sentence-transformers faiss-cpu\n```"
        
        if not query.strip():
            return "Please enter a query to search the knowledge base."
        
        try:
            with httpx.Client(timeout=30.0) as client:
                data = {'query': query, 'session_id': session_id, 'k': 5}
                response = client.post("https://vuen-code-hackathon.onrender.com/rag/query", data=data)
                response.raise_for_status()
                results = response.json()
                
                if results['results']:
                    formatted_results = f"## Knowledge Base Results ({results['total_results']} found):\n\n"
                    for i, result in enumerate(results['results'], 1):
                        content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                        metadata = result['metadata']
                        content_type = metadata.get('content_type', 'unknown')
                        timestamp = metadata.get('timestamp', 'unknown')[:19] if metadata.get('timestamp') else 'unknown'
                        
                        formatted_results += f"**Result {i}** ({content_type}):\n"
                        formatted_results += f"{content}\n"
                        formatted_results += f"*Session: {metadata.get('session_id', 'unknown')}, Time: {timestamp}*\n\n"
                    return formatted_results
                else:
                    return "No relevant knowledge found in the database.\n\n**Troubleshooting:**\n- Try adding some content by chatting with videos\n- Use the 'Add Test Data' button below\n- Check if the RAG system is properly initialized"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                return "❌ **RAG Service Unavailable**\n\nThe RAG system is not properly initialized. Try using the 'Add Test Data' button to initialize the system."
            return f"❌ Server Error: {e.response.status_code}. The backend may not be running."
        except httpx.RequestError:
            return "❌ Connection Error: Cannot reach the backend server."
        except Exception as e:
            return f"❌ Error querying knowledge base: {str(e)}"

    def add_debug_data():
        """Add test data to the RAG system for debugging."""
        if not RAG_AVAILABLE:
            return "❌ **RAG System Not Available**\n\nCannot add test data because the RAG system is not initialized."
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post("https://vuen-code-hackathon.onrender.com/rag/debug")
                response.raise_for_status()
                result = response.json()
                
                return f"✅ **Debug Operation Completed**\n\nTest data added: {result['test_data_added']} entries\n\nTotal documents in system: {result['stats'].get('total_documents', 'unknown')}\n\nYou can now try searching the knowledge base!"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                return "❌ **RAG Service Unavailable**\n\nThe RAG system is not properly initialized."
            return f"❌ Failed to add debug data. Server error: {e.response.status_code}"
        except Exception as e:
            return f"❌ Failed to add debug data: {str(e)}"

    def get_system_stats():
        """Get RAG system statistics."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get("https://vuen-code-hackathon.onrender.com/rag/stats")
                response.raise_for_status()
                stats = response.json()
                
                if not stats.get("rag_available", False):
                    return "❌ **RAG System Not Available**\n\nThe RAG system is not properly initialized."
                
                formatted_stats = "## 📊 RAG System Statistics\n\n"
                formatted_stats += f"**Status:** {stats.get('status', 'unknown')}\n"
                formatted_stats += f"**Total Documents:** {stats.get('total_documents', 0)}\n"
                formatted_stats += f"**Total Entries:** {stats.get('total_entries', 0)}\n"
                formatted_stats += f"**Active Sessions:** {stats.get('sessions', 0)}\n\n"
                
                if stats.get('content_type_breakdown'):
                    formatted_stats += "**Content Types:**\n"
                    for content_type, count in stats['content_type_breakdown'].items():
                        formatted_stats += f"- {content_type}: {count}\n"
                
                if stats.get('debug_info'):
                    debug = stats['debug_info']
                    formatted_stats += f"\n**System Info:**\n"
                    formatted_stats += f"- Documents Added: {debug.get('documents_added', 0)}\n"
                    formatted_stats += f"- Last Operation: {debug.get('last_add_operation', 'None')[:19] if debug.get('last_add_operation') else 'None'}\n"
                
                return formatted_stats
        except Exception as e:
            return f"❌ Failed to get stats: {str(e)}"

    # Custom CSS for enhanced styling with RAG features
    custom_css = """
    .gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
    .main-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .upload-section { background: #ffffff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .chat-section { background: #ffffff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .rag-section { background: #f8f9ff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e1e5fe; }
    .status-box { background: #e0f7fa; border-left: 5px solid #00acc1; color: #006064; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    .send-button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
    .clear-button { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; }
    .rag-button { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important; }
    .upload-button { border: 2px dashed #667eea !important; color: #667eea !important; }
    .upload-button:hover { background: #e8eaf6 !important; }
    """

    rag_status = "with RAG Knowledge System" if RAG_AVAILABLE else "(RAG System Unavailable)"
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")), 
        css=custom_css,
        title=f"🎥 AI Video Chat Assistant {rag_status}"
    ) as demo:
        
        session_id_state = gr.State(value=str(uuid.uuid4()))
        video_path_state = gr.State()

        with gr.Row():
            header_html = f"""<div class="main-header">
                <h1>🎥 AI Video Chat Assistant</h1>
                <h3>Powered by Gemini Vision AI {rag_status}</h3>
                <p>Upload videos, chat intelligently, and leverage persistent knowledge retrieval!</p>
            </div>"""
            gr.HTML(header_html)
        
        with gr.Row():
            # Left column - Video and RAG
            with gr.Column(scale=1):
                with gr.Tab("📹 Video Hub", elem_classes="upload-section"):
                    video_player = gr.Video(label="Video Preview", height=300, show_label=False)
                    upload_button = gr.UploadButton("🎬 Click to Upload Video", file_types=["video"], file_count="single", elem_classes="upload-button")
                    upload_status = gr.Markdown(visible=False, elem_classes="status-box")
                    clear_button = gr.Button("🔄 New Conversation", variant="secondary", elem_classes="clear-button")
                
                with gr.Tab("🧠 Knowledge Base", elem_classes="rag-section"):
                    if RAG_AVAILABLE:
                        gr.HTML("<h4 style='color: #333; margin-bottom: 1rem;'>Query & Manage RAG Knowledge</h4>")
                    else:
                        gr.HTML("<h4 style='color: #d32f2f; margin-bottom: 1rem;'>⚠️ RAG Knowledge (Unavailable)</h4>")
                    
                    with gr.Row():
                        rag_query_input = gr.Textbox(placeholder="Search knowledge base...", label="Query", scale=3, interactive=RAG_AVAILABLE)
                        rag_query_button = gr.Button("🔍 Search", elem_classes="rag-button", scale=1, interactive=RAG_AVAILABLE)
                    
                    rag_results = gr.Markdown(
                        label="Results", 
                        value="Enter a query to search the knowledge base." if RAG_AVAILABLE else "❌ RAG system not available. Install dependencies: `pip install langchain-community sentence-transformers faiss-cpu`"
                    )
                    
                    gr.HTML("<h5 style='color: #666; margin-top: 1.5rem;'>Debug Tools:</h5>")
                    with gr.Row():
                        debug_button = gr.Button("🎯 Add Test Data", variant="secondary", scale=1, interactive=RAG_AVAILABLE)
                        stats_button = gr.Button("📊 Show Stats", variant="secondary", scale=1, interactive=RAG_AVAILABLE)

            # Right column - Chat
            with gr.Column(scale=2, elem_classes="chat-section"):
                gr.HTML("<h3 style='text-align: center; color: #333; margin-bottom: 1rem;'>💬 Intelligent Chat Interface</h3>")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    show_label=False,
                    avatar_images=("🧑‍💻", "🤖"),
                    type="messages" 
                )
                
                with gr.Row():
                    prompt_box = gr.Textbox(placeholder="📁 Upload a video to begin...", interactive=False, scale=4, container=False, show_label=False)
                    send_button = gr.Button("Send 🚀", scale=1, variant="primary", elem_classes="send-button")

        # Event handlers
        upload_button.upload(
            fn=handle_video_upload,
            inputs=[upload_button, session_id_state],
            outputs=[video_path_state, video_player, prompt_box, upload_status]
        )

        submit_listeners = [prompt_box.submit, send_button.click]
        for listener in submit_listeners:
            listener(
                fn=handle_chat_submission,
                inputs=[prompt_box, chatbot, session_id_state, video_path_state],
                outputs=[chatbot, prompt_box, send_button, upload_button]
            ).then(
                fn=lambda: None, 
                outputs=video_path_state
            )
        
        clear_button.click(
            fn=clear_session,
            outputs=[chatbot, video_player, video_path_state, session_id_state, prompt_box, upload_status]
        )
        
        rag_query_button.click(
            fn=query_knowledge_base,
            inputs=[rag_query_input, session_id_state],
            outputs=rag_results
        )
        
        debug_button.click(
            fn=add_debug_data,
            outputs=rag_results
        )
        
        stats_button.click(
            fn=get_system_stats,
            outputs=rag_results
        )
        
        demo.queue().launch(
            server_name="0.0.0.0", 
            server_port=8000,
            show_error=True,
            # inbrowser=True,
            # share=True
        )

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    rag_status_msg = "with RAG Integration" if RAG_AVAILABLE else "(RAG Unavailable - Install: pip install langchain-community sentence-transformers faiss-cpu)"
    
    print("🚀 Starting AI Video Chat Assistant...")
    print(f"📊 FastAPI Backend: https://vuen-code-hackathon.onrender.com")
    print(f"🎨 Gradio Frontend: http://127.0.0.1:7860")
    print(f"🧠 RAG System: {rag_status_msg}")
    print("=" * 70)
    
    if not RAG_AVAILABLE:
        print("⚠️  WARNING: RAG system is not available!")
        print("   Install dependencies with: pip install langchain-community sentence-transformers faiss-cpu")
        print("   The application will work without RAG but with limited knowledge persistence.")
        print("=" * 70)

    launch_gradio_ui()
