import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from git import Repo
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings        # Updated import
from langchain.schema import Document
from openai import OpenAI
from dotenv import load_dotenv
import logging
from pinecone import Pinecone, ServerlessSpec
import requests 
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    callback_url: str

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "codebase-rag"

pinecone_instance = Pinecone(   
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)


if PINECONE_INDEX_NAME not in pinecone_instance.list_indexes().names():
    pinecone_instance.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',    
            region='us-east-1'
        )
    )


pinecone_index = pinecone_instance.Index(PINECONE_INDEX_NAME)

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
# Supported extensions and ignored directories
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                        '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
               '__pycache__', '.next', '.vscode', 'vendor'}

REPO_URL = "https://github.com/CoderAgent/SecureAgent"

def clone_repository(repo_url):
    repo_name = repo_url.rstrip('/').split("/")[-1]  # Extracting repository name from URL
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, repo_name)
    logger.info(f"Cloning repository {repo_url} into {repo_path}")
    Repo.clone_from(repo_url, repo_path)
    return repo_path

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    files_content = []

    try:
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        logger.error(f"Error reading repository: {str(e)}")

    return files_content

def get_huggingface_embeddings(text):
    """Generate embeddings for the given text using HuggingFace model."""
    return embedding_model.encode(text).tolist()

def perform_rag(query: str) -> str:
    """Performs Retrieval-Augmented Generation based on the query."""
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding,
        top_k=3,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    # Extract contexts
    contexts = [item['metadata']['content'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.

Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
"""

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    response = llm_response.choices[0].message.content
    return response 

@app.on_event("startup")
def startup_event():
    logger.info("Starting up and setting up Pinecone index.")
    repo_path = clone_repository(REPO_URL)
    file_content = get_main_files_content(repo_path)

    documents = []
    for file in file_content:
        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file['name'], "content": file['content']}
        )
        documents.append(doc)

    vectors = []
    for doc in documents:
        embedding = hf_embeddings.embed_query(doc.page_content)
        vectors.append((doc.metadata['source'], embedding, doc.metadata))

    logger.info(f"Upserting {len(vectors)} vectors to Pinecone.")
    pinecone_index.upsert(vectors=vectors, namespace="https://github.com/CoderAgent/SecureAgent")

    shutil.rmtree(os.path.dirname(repo_path))
    logger.info("Startup setup completed.")


def process_rag_task(query: str, callback_url: str, task_id: str):
    """Background task to perform RAG and send the result via webhook."""
    try:
        logger.info(f"Processing RAG task {task_id}")
        response = perform_rag(query)
        logger.info(f"RAG task {task_id} completed. Sending webhook.")
        
        # Prepare the payload
        payload = {
            "task_id": task_id,
            "response": response
        }
        
        # Send the webhook callback
        headers = {"Content-Type": "application/json"}
        webhook_response = requests.post(callback_url, json=payload, headers=headers)
        
        if webhook_response.status_code != 200:
            logger.error(f"Failed to send webhook for task {task_id}. Status code: {webhook_response.status_code}")
    except Exception as e:
        logger.error(f"Error processing RAG task {task_id}: {str(e)}")

        payload = {
            "task_id": task_id,
            "error": str(e)
        }
        try:
            requests.post(callback_url, json=payload, headers=headers)
        except Exception as webhook_error:
            logger.error(f"Failed to send error webhook for task {task_id}: {str(webhook_error)}")

@app.post("/perform_rag")
async def perform_rag_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Endpoint to initiate RAG processing."""
    try:
        task_id = str(uuid.uuid4())
        
        background_tasks.add_task(process_rag_task, request.query, request.callback_url, task_id)
        
        logger.info(f"RAG task {task_id} initiated.")
        return {"task_id": task_id, "message": "RAG task initiated. You will receive a callback upon completion."}
    except Exception as e:
        logger.error(f"Error initiating RAG task: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate RAG task.")