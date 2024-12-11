from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json
import os
import random
import re

# Set your HuggingFace token
HUGGINGFACE_TOKEN = "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"  # Replace with your token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

class RAGChatbot:
    def __init__(self):
        try:
            # Load the JSON data
            with open("app/Database.json", 'r', encoding='utf-8') as file:
                self.intents_data = json.load(file)
            print("Successfully loaded Database.json")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Successfully initialized embeddings")
            
            # Initialize vector store
            self.initialize_knowledge_base()
            print("Successfully initialized knowledge base")
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def clean_response(self, response):
        """Clean HTML tags and format links properly"""
        # Replace <a> tags with proper formatting
        response = re.sub(r'<a target="_blank" href="([^"]+)">', r'\1 ', response)
        response = response.replace('</a>', '')
        
        # Remove any remaining HTML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        return response.strip()

    def initialize_knowledge_base(self):
        try:
            # Create embeddings for all patterns
            all_patterns = []
            pattern_to_intent = {}
            
            for intent in self.intents_data['intents']:
                for pattern in intent['patterns']:
                    all_patterns.append(pattern)
                    pattern_to_intent[pattern] = intent
            
            # Initialize Chroma
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory="data/vectorstore"
            )
            
            # Add patterns to vector store
            self.vectorstore.add_texts(
                texts=all_patterns,
                metadatas=[{'intent': pattern_to_intent[pattern]['tag']} for pattern in all_patterns]
            )
            
        except Exception as e:
            print(f"Error in knowledge base initialization: {e}")
            raise

    def get_response(self, query: str):
        try:
            print(f"\nProcessing query: {query}")
            query = query.lower().strip()
            
            # Direct pattern matching
            for intent in self.intents_data['intents']:
                patterns = [p.lower().strip() for p in intent['patterns']]
                if query in patterns:
                    response = random.choice(intent['responses'])
                    response = self.clean_response(response)
                    print(f"Direct match found for intent: {intent['tag']}")
                    return response, [intent['tag']]
            
            # Similarity search if no direct match
            results = self.vectorstore.similarity_search_with_score(query, k=1)
            
            if results:
                doc, score = results[0]
                intent_tag = doc.metadata.get('intent')
                print(f"Found similar intent: {intent_tag} with score: {score}")
                
                # Find matching intent
                for intent in self.intents_data['intents']:
                    if intent['tag'] == intent_tag:
                        response = random.choice(intent['responses'])
                        response = self.clean_response(response)
                        return response, [intent_tag]
            
            return "I'm not sure how to respond to that. Could you please rephrase your question?", []
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I apologize, but I encountered an error. Please try again.", []
