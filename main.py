from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
import pdf_loader
import requests
import os


app = Flask(__name__)

raw_file_content = requests.get("https://www.helmerinc.com/sites/default/files/2026-01/Manual%20-%20GX%20Refrigerator%20IFU%20360414-E.pdf").content

file_path = './manual.pdf'

if not os.path.exists(file_path):
    with open("manual.pdf", "wb") as f:
        f.write(raw_file_content)

file_text = pdf_loader.get_pdf_text(file_path)
chunks = pdf_loader.split_text_into_chunks(file_text)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=8)

documents = []

for i in tqdm(range(len(chunks))):
    document = Document(
        page_content=chunks[i],
        metadata={"source":"pdf"},
        id=i
    )
    documents.append(document)

persist_dir = "chroma_db"
vector_store = Chroma.from_documents(documents, embeddings)
app.logger.info("Vector Store Set Up")

llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        temperature=1.0,  # Gemini 3.0+ defaults to 1.0
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

@app.route("/")
def base():
    return jsonify({
        "message": "It fucking works"
    })
        
@app.route("/api/<user_prompt>")
def run_prompt(user_prompt):
    retrieved_docs = vector_store.similarity_search(user_prompt)
    
    context = ""

    for i in range(len(retrieved_docs)):
        context += f"Context {i}: \n " + retrieved_docs[i].page_content + "\n_____________________\n"

    messages = [
        (
            "system",
            "You are a helpful assistant that teaches users how to use their GX Refrigerator.",
        ),
        ("human", context + "\n_____________________\n" + user_prompt),
    ]
    
    ai_msg = llm.invoke(messages)
    
    return jsonify({
        "response": ai_msg.content[0]["text"]
    })


if __name__ == '__main__':
    app.run(debug=True)



