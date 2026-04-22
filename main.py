import os
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pypdf2
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mfc_knowledge")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
DATA_DIR = Path("/app/data")

# Initialize FastAPI app
app = FastAPI(title="MFC RAG System", description="Question-Answering system for MFC knowledge base")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for LangChain components
embeddings = None
llm = None
vector_store = None
qdrant_client = None


def initialize_components():
    """Initialize embeddings, LLM, and vector store"""
    global embeddings, llm, vector_store, qdrant_client
    
    try:
        # Initialize Ollama embeddings (using mxbai-embed-large via Ollama)
        logger.info(f"Initializing embeddings with Ollama at {OLLAMA_BASE_URL}")
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url=OLLAMA_BASE_URL
        )
        
        # Initialize Ollama LLM
        logger.info(f"Initializing LLM with Qwen2.5:7b-instruct at {OLLAMA_BASE_URL}")
        llm = OllamaLLM(
            model="qwen2.5:7b-instruct",
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            max_tokens=1024
        )
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not collection_exists:
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            # Get embedding dimension by embedding a test sentence
            test_embedding = embeddings.embed_query("test")
            vector_size = len(test_embedding)
            logger.info(f"Vector size: {vector_size}")
            
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from various file formats"""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif suffix == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Convert markdown to plain text (basic approach)
                return markdown.markdown(content, extensions=['extra'])
        
        elif suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return ""
    
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""


def chunk_document(text: str) -> List[str]:
    """Split document into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = []
    context_used: bool = True


@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest documents from uploaded files or from /data directory.
    Files are chunked, embedded, and stored in Qdrant.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    ingested_count = 0
    errors = []
    
    for file in files:
        try:
            logger.info(f"Processing file: {file.filename}")
            
            # Save file temporarily
            temp_path = DATA_DIR / file.filename
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract text
            text = extract_text_from_file(temp_path)
            
            if not text:
                errors.append(f"No text extracted from {file.filename}")
                continue
            
            # Chunk the document
            chunks = chunk_document(text)
            logger.info(f"Created {len(chunks)} chunks from {file.filename}")
            
            # Add chunks to vector store
            for i, chunk in enumerate(chunks):
                vector_store.add_texts(
                    texts=[chunk],
                    metadatas=[{
                        "source": file.filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }]
                )
            
            ingested_count += len(chunks)
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            errors.append(f"Error with {file.filename}: {str(e)}")
    
    return {
        "message": f"Ingestion complete",
        "chunks_added": ingested_count,
        "errors": errors if errors else None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on the knowledge base.
    Uses RAG: retrieves top-K relevant chunks and generates answer with LLM.
    """
    if not vector_store or not llm:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        question = request.question
        logger.info(f"Received question: {question}")
        
        # Perform similarity search using cosine similarity
        docs_with_scores = vector_store.similarity_search_with_score(question, k=TOP_K)
        
        if not docs_with_scores:
            return AnswerResponse(
                answer="Нет данных в базе знаний для ответа на этот вопрос.",
                sources=[],
                context_used=False
            )
        
        # Extract context and sources
        context_chunks = []
        sources = []
        
        for doc, score in docs_with_scores:
            context_chunks.append(doc.page_content)
            source_info = doc.metadata.get("source", "unknown")
            if source_info not in sources:
                sources.append(source_info)
            logger.info(f"Chunk score: {score}, source: {source_info}")
        
        # Build context string
        context = "\n\n---\n\n".join(context_chunks)
        
        # Create prompt with system instruction
        system_instruction = "Ты — помощник сотрудника МФЦ. Отвечай кратко, по делу, только на основе предоставленных регламентов."
        
        prompt = f"""{system_instruction}

Используя только этот контекст:
{context}

Ответь на вопрос: {question}

Если ответа нет в предоставленном контексте, скажи 'Нет данных'.
"""
        
        logger.info("Generating answer with LLM...")
        
        # Generate answer using LLM
        answer = llm.invoke(prompt)
        
        logger.info(f"Answer generated: {answer[:100]}...")
        
        return AnswerResponse(
            answer=answer.strip(),
            sources=sources,
            context_used=True
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_url": OLLAMA_BASE_URL,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": str(collection_info.status)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
