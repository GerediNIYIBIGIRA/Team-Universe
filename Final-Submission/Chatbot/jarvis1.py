

import os
import requests
from bs4 import BeautifulSoup
import json
import argparse
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv

# Langchain and AI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
_ = load_dotenv(find_dotenv())

class RAGVoiceAssistant:
    def __init__(self, config):
        # Initialize voice and speech recognition
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', config['voice'])
        self.engine.setProperty('volume', config['volume'])
        self.engine.setProperty('rate', config['rate'])
        
        self.recognizer = sr.Recognizer()
        
        # Initialize LLM and RAG components
        self.llm = ChatOpenAI(
            temperature=config['temperature'], 
            model=config['model'], 
            api_key=config['api_key']
        )
        
        # Initialize session and conversation history
        self.session_id = config['session_id']
        self.store = {}
        
        # Prepare RAG components
        self.vector_store = self.prepare_rag_documents()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an AI assistant For Team Universe, skilled in {ability}. Use retrieved context to provide precise answers."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self.runnable = self.prompt | self.llm

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Manage session history"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def scrape_websites(self, urls):
        """Scrape content from given URLs"""
        scraped_data = {}
        for idx, url in enumerate(urls, start=1):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = soup.title.string if soup.title else f"Document {idx}"
                content = soup.get_text(separator='\n', strip=True)
                
                scraped_data[url] = {"title": title, "content": content}
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                scraped_data[url] = {"title": f"Error {idx}", "content": f"Failed to scrape: {e}"}
        
        return scraped_data

    def prepare_rag_documents(self, json_file_path='formatted_data.json'):
        """Prepare documents for RAG"""
        # URLs to scrape
        links = [
            "https://neveragainrwanda.org/youth-unemployment-and-perplexing-access-to-finance-in-rwanda/",
            "https://statistics.gov.rw/publication/2119",
            "https://www.statistics.gov.rw/publication/2138"
        ]
        
        # Scrape websites
        scraped_data = self.scrape_websites(links)
        
        # Load JSON data
        try:
            with open(json_file_path, "r") as file:
                json_data = json.load(file)
        except FileNotFoundError:
            json_data = {}
        
        # Combine scraped data with existing JSON data
        combined_data = {
            **json_data, 
            **{f"scraped_{idx}": data["content"] for idx, data in enumerate(scraped_data.values(), start=len(json_data) + 1)}
        }
        
        # Prepare documents for vectorization
        documents = [
            Document(page_content=str(entry), metadata={"id": idx}) 
            for idx, entry in combined_data.items()
        ]
        
        # Initialize embeddings and text splitter
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        
        # Split and vectorize documents
        split_documents = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(split_documents, embeddings)
        
        return vector_store

    def retrieve_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        return self.vector_store.similarity_search(query, k=k)

    def generate_response(self, ability, query):
        """Generate response using RAG approach"""
        # Retrieve context
        context_docs = self.retrieve_context(query)
        context = "\n".join([doc.page_content for doc in context_docs])
        
        # Create message history runnable
        with_message_history = RunnableWithMessageHistory(
            self.runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        # Invoke with enhanced context
        completions = with_message_history.invoke(
            {"ability": ability, "input": f"Context: {context}\n\nQuery: {query}"},
            config={"configurable": {"session_id": self.session_id}}
        )
        
        return completions.content

    def speak(self, text):
        """Text-to-speech output"""
        print(f"Jarvis: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Speech recognition input"""
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source, phrase_time_limit=5)
            print("Processing...")
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def run(self, ability="Psychology"):
        """Main interaction loop"""
        self.speak("Hello, I am Team Universe AI Assistant, I am being developed in NISR Big Data Hackathon. I can help you with all the information regarding to unemployment in Rwanda, And on top of that I can give you some job recommendaion and career guidance, How can I help you?")
        
        while True:
            prompt = self.listen()
            
            if prompt is None:
                self.speak("I'm sorry, I didn't understand that.")
                continue
            
            if prompt.lower() == "thank you for your help":
                self.speak("Goodbye!")
                break
            
            try:
                response = self.generate_response(ability, prompt)
                
                # Split response into sentences for speech
                sentences = response.split(".")
                for sentence in sentences:
                    if sentence.strip():
                        self.speak(sentence.strip() + ".")
            
            except Exception as e:
                self.speak(f"I encountered an error: {str(e)}")

def main():
    # Configure parser
    parser = argparse.ArgumentParser(description="RAG Voice Assistant")
    parser.add_argument("--ability", type=str, default="Unemployment", help="Assistant's primary ability")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    parser.add_argument("--voice", type=str, default="com.apple.eloquence.en-US.Grandpa", help="TTS voice")
    parser.add_argument("--volume", type=float, default=1.0, help="TTS volume")
    parser.add_argument("--rate", type=int, default=200, help="TTS speech rate")
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        'api_key': os.getenv("OPENAI_API_KEY"),
        'model': args.model,
        'temperature': args.temperature,
        'voice': args.voice,
        'volume': args.volume,
        'rate': args.rate,
        'session_id': 'default_session'
    }
    
    # Initialize and run assistant
    assistant = RAGVoiceAssistant(config)
    assistant.run(ability=args.ability)

if __name__ == "__main__":
    main()