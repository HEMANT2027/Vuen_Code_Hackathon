# Import necessary modules from LangChain community
from langchain_community.vectorstores import FAISS  # For vector store operations
from langchain_community.embeddings import HuggingFaceEmbeddings  # For generating text embeddings

# Initialize the embedding model using Hugging Face transformers
# Using a lightweight model suitable for CPU usage
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Pretrained sentence embedding model
    model_kwargs={'device': 'cpu'}  # Force CPU usage (no GPU required)
)

# Load an existing FAISS vector store from local disk
# `rag_data` is the directory where the index is stored
# `allow_dangerous_deserialization=True` is needed when loading pickled data in some environments
vectorstore = FAISS.load_local(
    "rag_data",
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

# Define the user query for semantic search
query = "What did the user ask about the video?"

# Perform a similarity search on the vector store to get top 3 matching documents
results = vectorstore.similarity_search(query, k=3)

# Print the top 3 results retrieved from the similarity search
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:\n{doc.page_content}")
