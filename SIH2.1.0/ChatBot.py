import os
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGChatBot:
    def __init__(self, documents_path, openai_api_key=None):
        """Initialize the RAG-powered chatbot"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.documents_path = documents_path
        self.qa_chain = None
        self.setup_rag()

    def setup_rag(self):
        """Set up the RAG pipeline"""
        # Load documents
        loader = DirectoryLoader(self.documents_path, glob="**/*.txt")
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Create QA chain
        self.qa_chain = VectorDBQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=self.openai_api_key),
            chain_type="stuff",
            vectorstore=vectorstore
        )

    def ask(self, question: str) -> str:
        """Ask a question to the RAG-powered chatbot"""
        if not self.qa_chain:
            raise ValueError("RAG pipeline not initialized. Call setup_rag() first.")
        
        response = self.qa_chain.run(question)
        return response

# Example usage
if __name__ == "__main__":
    # Initialize chatbot with path to your documents
    chatbot = RAGChatBot("path/to/your/documents")
    
    # Ask questions
    question = "What is RAG in AI?"
    answer = chatbot.ask(question)
    print(f"Q: {question}\nA: {answer}")
