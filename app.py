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
# from app.rag_integration import add_to_rag_vectorstore

# --- 0. Environment and Configuration ---
load_dotenv()  # Load variables from .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
TEMP_VIDEO_DIR = "temp_videos"

# Create temporary directory if it doesn't exist
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# --- 1. The Memory Core: ChromaDB Setup ---
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    # Initialize embedding function for vector similarity search
    gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)
    client = PersistentClient(path=CHROMA_DB_PATH)
    
    # Create collection to store chat history with embeddings
    chat_history_collection = client.get_or_create_collection(
        name="video_chat_history",
        embedding_function=gemini_ef
    )
    print("ChromaDB collection loaded successfully.")
except Exception as e:
    print(f"FATAL: Error creating ChromaDB collection: {e}")
    chat_history_collection = None

# --- 2. The Robust Backend: FastAPI Setup ---
app = FastAPI()

# Response model for API endpoints
class ChatResponse(BaseModel):
    response_text: str
    session_id: str

# --- Workstream A: Backend and AI Logic ---

# Initialize the Gemini API client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"FATAL: API key configuration error: {e}")


def extract_frames(video_path: str, fps: int = 1) -> List[Dict]:
    """Extracts frames from video at specified FPS and converts to base64 for Gemini."""
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
        
        # Extract frame at specified intervals
        if count % frame_interval == 0:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                # Convert frame to Gemini-compatible format
                frames.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffer).decode()
                    }
                })
        count += 1
    cap.release()
    return frames

@app.post("/chat/video", response_model=ChatResponse)
async def chat_with_video(
    video_file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Main endpoint for video + text chat using Gemini Vision and ChromaDB memory."""
    if not chat_history_collection or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Backend not initialized correctly.")

    session_id = session_id or str(uuid.uuid4())
    temp_video_path = os.path.join(TEMP_VIDEO_DIR, f"temp_{session_id}_{video_file.filename}")

    # Save uploaded video temporarily
    with open(temp_video_path, "wb") as f:
        f.write(await video_file.read())

    try:
        # Extract frames for Gemini processing
        frames = extract_frames(temp_video_path, fps=1)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")

        # Retrieve relevant conversation history using vector search
        context_history = ""
        try:
            query_embedding = gemini_ef([prompt])[0]
            results = chat_history_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"session_id": session_id}
            )
            if results['documents'] and results['documents'][0]:
                context_history = "\n".join(reversed(results['documents'][0]))
                context_history = f"Relevant past conversation:\n{context_history}\n---\n"
        except Exception as e:
            print(f"Warning: Could not query ChromaDB: {e}")

        # Prepare content for Gemini with context + frames
        full_prompt = f"{context_history}Given the video context, answer the user's question: {prompt}"
        content = [{"role": "user", "parts": [{"text": full_prompt}] + frames}]

        # Generate response using Gemini Vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=content)
        ai_response = response.text

        # Store conversation in vector database
        chat_history_collection.add(
            documents=[f"User: {prompt}\nAssistant: {ai_response}"],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )
        return ChatResponse(response_text=ai_response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        # Clean up temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text_only(prompt: str = Form(...), session_id: Optional[str] = Form(None)):
    """Endpoint for text-only follow-up questions with memory context."""
    if not chat_history_collection or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Backend not initialized correctly.")

    session_id = session_id or str(uuid.uuid4())

    # Retrieve conversation context from ChromaDB
    context_history = ""
    try:
        query_embedding = gemini_ef([prompt])[0]
        results = chat_history_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"session_id": session_id}
        )
        if results['documents'] and results['documents'][0]:
            context_history = "\n".join(reversed(results['documents'][0]))
            context_history = f"Here is the summary of our conversation so far:\n{context_history}\n---\n"
    except Exception as e:
        print(f"Warning: Could not query ChromaDB for text chat: {e}")

    # Generate response with conversation context
    full_prompt = f"{context_history}User's follow-up question: {prompt}"
    content = [{"role": "user", "parts": [{"text": full_prompt}]}]

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(contents=content)
        ai_response = response.text

        # Store new conversation turn
        chat_history_collection.add(
            documents=[f"User: {prompt}\nAssistant: {ai_response}"],
            metadatas=[{"session_id": session_id}],
            ids=[str(uuid.uuid4())]
        )
        return ChatResponse(response_text=ai_response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")

# --- 3. The Interactive Frontend: Enhanced Gradio UI ---

def launch_gradio_ui():
    """Launches the enhanced Gradio interface with modern UI design."""

    def handle_chat_submission(user_input: str, history: List, session_id: str, video_path: str):
        """
        Handles user chat submissions and calls appropriate FastAPI endpoint.
        Routes to video or text endpoint based on whether video is uploaded.
        """
        history = history or []
        
        if not user_input.strip():
            return history, gr.update(value=""), gr.update(interactive=True), gr.update(interactive=True)
        
        # Show user message and thinking indicator immediately
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": "🤔 Thinking..."})
        yield history, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False)

        output_message = ""
        try:
            with httpx.Client(timeout=180.0) as client:
                if video_path:
                    # Process video + text request
                    with open(video_path, "rb") as f:
                        files = {'video_file': (os.path.basename(video_path), f, 'video/mp4')}
                        data = {'prompt': user_input, 'session_id': session_id}
                        response = client.post("http://127.0.0.1:8000/chat/video", files=files, data=data)
                else:
                    # Process text-only follow-up
                    data = {'prompt': user_input, 'session_id': session_id}
                    response = client.post("http://127.0.0.1:8000/chat/text", data=data)

                response.raise_for_status()
                chat_response = response.json()
                output_message = chat_response['response_text']

        except httpx.HTTPStatusError as e:
            output_message = f"❌ Error: Failed to get response from server. Status {e.response.status_code}."
        except httpx.RequestError:
            output_message = f"❌ Error: Network request failed. Please check if the backend server is running."
        except Exception as e:
            output_message = f"❌ An unexpected error occurred: {str(e)}"

        # Update chat with AI response
        history[-1]["content"] = output_message
        yield history, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def handle_video_upload(video_file, session_id):
        """Processes video upload and updates UI components."""
        if video_file:
            print(f"Video uploaded: {video_file.name} for session {session_id}")
            return (
                video_file.name,  # Store video path
                gr.update(value=video_file.name),  # Update video player
                gr.update(interactive=True, placeholder="✨ Video loaded! Ask me anything..."),
                gr.update(visible=True, value=f"📹 **{os.path.basename(video_file.name)}** uploaded!")
            )
        return (
            None, 
            gr.update(value=None), 
            gr.update(interactive=False, placeholder="📁 Upload a video to start..."),
            gr.update(visible=False)
        )

    def clear_session():
        """Resets all UI components and generates new session ID."""
        new_session_id = str(uuid.uuid4())
        print(f"New session started: {new_session_id}")
        return (
            [],  # Clear chat history
            None, # Clear video player
            None, # Clear video path
            new_session_id, # New session ID
            gr.update(placeholder="📁 Upload a video to begin...", interactive=False),
            gr.update(visible=False)  # Hide upload status
        )

    # Custom CSS for enhanced styling
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    .main-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .upload-section { background: #ffffff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .chat-section { background: #ffffff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    .status-box { background: #e0f7fa; border-left: 5px solid #00acc1; color: #006064; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    .send-button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
    .clear-button { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; }
    .upload-button { border: 2px dashed #667eea !important; color: #667eea !important; }
    .upload-button:hover { background: #e8eaf6 !important; }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")), 
        css=custom_css,
        title="🎥 AI Video Chat Assistant"
    ) as demo:
        
        # State variables for session management
        session_id_state = gr.State(value=str(uuid.uuid4()))
        video_path_state = gr.State()

        # Header section
        with gr.Row():
            gr.HTML("""<div class="main-header"><h1>🎥 AI Video Chat Assistant</h1><h3>Powered by Gemini Vision AI</h3><p>Upload a video and have an intelligent conversation about its content!</p></div>""")
        
        with gr.Row():
            # Left column: Video upload and controls
            with gr.Column(scale=1, elem_classes="upload-section"):
                gr.HTML("<h3 style='text-align: center; color: #333; margin-bottom: 1rem;'>📹 Video Hub</h3>")
                video_player = gr.Video(label="Video Preview", height=300, show_label=False)
                upload_button = gr.UploadButton("🎬 Click to Upload Video", file_types=["video"], file_count="single", elem_classes="upload-button")
                upload_status = gr.Markdown(visible=False, elem_classes="status-box")
                clear_button = gr.Button("🔄 New Conversation", variant="secondary", elem_classes="clear-button")

            # Right column: Chat interface
            with gr.Column(scale=2, elem_classes="chat-section"):
                gr.HTML("<h3 style='text-align: center; color: #333; margin-bottom: 1rem;'>💬 Chat Interface</h3>")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=False,
                    avatar_images=("🧑‍💻", "🤖"),
                    type="messages" 
                )
                
                # Input row with textbox and send button
                with gr.Row():
                    prompt_box = gr.Textbox(placeholder="📁 Upload a video to begin...", interactive=False, scale=4, container=False, show_label=False)
                    send_button = gr.Button("Send 🚀", scale=1, variant="primary", elem_classes="send-button")

        # Event handlers
        upload_button.upload(
            fn=handle_video_upload,
            inputs=[upload_button, session_id_state],
            outputs=[video_path_state, video_player, prompt_box, upload_status]
        )

        # Handle both textbox submit and button click
        submit_listeners = [prompt_box.submit, send_button.click]
        for listener in submit_listeners:
            listener(
                fn=handle_chat_submission,
                inputs=[prompt_box, chatbot, session_id_state, video_path_state],
                outputs=[chatbot, prompt_box, send_button, upload_button]
            ).then(
                fn=lambda: None,  # Clear video path after processing
                outputs=video_path_state
            )
        
        # Clear session handler
        clear_button.click(
            fn=clear_session,
            outputs=[chatbot, video_player, video_path_state, session_id_state, prompt_box, upload_status]
        )
        
        # Launch Gradio interface
        demo.queue().launch(
            server_name="127.0.0.1", 
            server_port=7860,
            show_error=True,
            inbrowser=True
        )

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    def run_fastapi():
        """Runs FastAPI server in background thread."""
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # Start FastAPI server in daemon thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()

    # Print startup information
    print("🚀 Starting AI Video Chat Assistant...")
    print("📊 FastAPI Backend: http://127.0.0.1:8000")
    print("🎨 Gradio Frontend: http://127.0.0.1:7860")
    print("="*50)

    # Launch Gradio UI (runs in main thread)
    launch_gradio_ui()