from os import path
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from fastapi import FastAPI

app = FastAPI()

while True:

    try:
        filepath = input("Enter the path to the PDF file: ")
        loader = PyPDFLoader(filepath)
        doc = loader.load()
        if doc:
            print(f"Successfully loaded {len(doc)} pages from the PDF.")
            break
    except:
        print("Invalid input. Please provide a valid file path.")
        
@app.post("/ingest")
def ingest_pdf():
    with open(filepath, 'wb') as f:
        f.write


