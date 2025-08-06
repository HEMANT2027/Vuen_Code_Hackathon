# debug_rag.py - Run this script to test and debug your RAG system

import sys
import os
import json
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.getcwd(), 'app'))

def test_rag_system():
    """Test the RAG system functionality."""
    print("🧠 Testing RAG System...")
    print("=" * 50)
    
    try:
        # Import the RAG module
        from rag_integration import (
            vectorstore, 
            debug_add_test_data, 
            query_rag_vectorstore, 
            get_vectorstore_stats,
            add_to_rag_vectorstore,
            force_reinitialize
        )
        
        print("✅ RAG module imported successfully")
        
        # Check vectorstore status
        if vectorstore is None:
            print("❌ Vectorstore is None - attempting force reinitialization...")
            if force_reinitialize():
                print("✅ Force reinitialization successful")
            else:
                print("❌ Force reinitialization failed")
                return False
        else:
            print(f"✅ Vectorstore loaded with {vectorstore.index.ntotal} documents")
        
        # Get and display stats
        print("\n📊 Vectorstore Statistics:")
        stats = get_vectorstore_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Add test data
        print("\n➕ Adding test data...")
        test_count = debug_add_test_data()
        print(f"✅ Added {test_count} test entries")
        
        # Test queries
        print("\n🔍 Testing queries...")
        test_queries = [
            "cooking tutorial",
            "video analysis", 
            "nature documentary",
            "recipe ingredients",
            "animal species"
        ]
        
        for query in test_queries:
            results = query_rag_vectorstore(query, k=3)
            print(f"  Query: '{query}' -> {len(results)} results")
            for i, doc in enumerate(results[:2]):  # Show first 2 results
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"    {i+1}: {preview}")
        
        print("\n✅ RAG system test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import RAG module: {e}")
        print("💡 Make sure you have installed: pip install langchain-community sentence-transformers faiss-cpu")
        return False
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing RAG dependencies...")
    
    dependencies = [
        "langchain-community",
        "sentence-transformers", 
        "faiss-cpu",
        "pickle5"  # For Python < 3.8 compatibility
    ]
    
    import subprocess
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")

def create_sample_data():
    """Create comprehensive sample data for testing."""
    print("\n🎯 Creating comprehensive sample data...")
    
    try:
        from rag_integration import add_to_rag_vectorstore
        
        sample_data = [
            {
                "text": "Video Analysis: A cooking tutorial showing how to make pasta. The chef demonstrates boiling water, adding salt, and cooking spaghetti for 8-10 minutes. The video has clear audio and good lighting.",
                "content_type": "video_analysis",
                "session_id": "cooking_session_1"
            },
            {
                "text": "User Question: What ingredients do I need for the pasta recipe? The user is asking about the specific ingredients shown in the cooking video.",
                "content_type": "user_query", 
                "session_id": "cooking_session_1"
            },
            {
                "text": "AI Response: Based on the video analysis, the pasta recipe requires: spaghetti noodles, water, salt, olive oil, garlic, tomatoes, and fresh basil. The chef also uses parmesan cheese for garnish.",
                "content_type": "ai_response",
                "session_id": "cooking_session_1"
            },
            {
                "text": "Video Analysis: Nature documentary featuring African wildlife. Shows lions hunting zebras in the savanna. Excellent cinematography with drone footage and close-up shots of animal behavior.",
                "content_type": "video_analysis",
                "session_id": "nature_session_1"
            },
            {
                "text": "Video Analysis: Educational content about machine learning concepts. The instructor explains neural networks using whiteboard diagrams and code examples in Python.",
                "content_type": "video_analysis",
                "session_id": "ml_session_1"
            },
            {
                "text": "System Capability: The AI can identify objects, people, animals, text, and activities in videos. It can also analyze video quality, lighting, audio, and provide detailed scene descriptions.",
                "content_type": "capability",
                "session_id": "global"
            },
            {
                "text": "User Pattern: Users frequently ask about identifying objects in videos, understanding video content, and getting summaries of long videos.",
                "content_type": "user_pattern", 
                "session_id": "global"
            }
        ]
        
        success_count = 0
        for entry in sample_data:
            if add_to_rag_vectorstore(
                text=entry["text"],
                session_id=entry["session_id"],
                content_type=entry["content_type"],
                source="sample"
            ):
                success_count += 1
        
        print(f"✅ Created {success_count}/{len(sample_data)} sample entries")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def interactive_query_test():
    """Interactive query testing."""
    print("\n🎮 Interactive Query Test")
    print("Type queries to test the RAG system. Type 'quit' to exit.")
    print("-" * 50)
    
    try:
        from rag_integration import query_rag_vectorstore, get_vectorstore_stats
        
        while True:
            query = input("\n🔍 Enter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"Searching for: '{query}'...")
            results = query_rag_vectorstore(query, k=5)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. Content: {doc.page_content[:150]}...")
                    print(f"   Metadata: {doc.metadata}")
            else:
                print("No results found.")
                
                # Show stats for debugging
                stats = get_vectorstore_stats()
                print(f"Total documents in store: {stats.get('total_documents', 0)}")
    
    except KeyboardInterrupt:
        print("\n👋 Exiting interactive test...")
    except Exception as e:
        print(f"❌ Error in interactive test: {e}")

if __name__ == "__main__":
    print("🎥 AI Video Chat RAG System Debug Tool")
    print("=" * 50)
    
    # Check if dependencies need to be installed
    try:
        import langchain_community
        import sentence_transformers  
        import faiss
        print("✅ All dependencies are available")
    except ImportError:
        print("⚠️  Missing dependencies detected")
        install_deps = input("Install missing dependencies? (y/n): ").lower().startswith('y')
        if install_deps:
            install_dependencies()
        else:
            print("❌ Cannot proceed without dependencies")
            sys.exit(1)
    
    # Main test sequence
    success = test_rag_system()
    
    if success:
        # Create more comprehensive sample data
        create_sample_data()
        
        # Offer interactive testing
        interactive_test = input("\n🎮 Run interactive query test? (y/n): ").lower().startswith('y')
        if interactive_test:
            interactive_query_test()
    
    print("\n🏁 Debug session completed!")
    print("📝 Check the 'rag_data/debug_info.json' file for detailed logs.")