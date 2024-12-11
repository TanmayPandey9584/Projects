from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from .embeddings import RAGChatbot

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create required directories if they don't exist
static_dir = os.path.join(os.path.dirname(__file__), "static")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Initialize RAG chatbot
chatbot = RAGChatbot()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    response, sources = chatbot.get_response(chat_request.message)
    return ChatResponse(response=response, sources=sources) 