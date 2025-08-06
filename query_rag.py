from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.load_local("rag_data", embeddings=embedding, allow_dangerous_deserialization=True)

query = "What did the user ask about the video?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:\n{doc.page_content}")
