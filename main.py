from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import PyPDF2
import io
import os
from openai import OpenAI
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import concurrent.futures
import fitz  # PyMuPDF
import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from models import PDFDocument, PDFDocumentCreate, PDFDocumentResponse
from database import get_db, engine, Base
import requests.adapters
from urllib3.util.retry import Retry
import requests
from utils.rag import RAGProcessor
import logging
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity  # Add this import
from agents import latex_to_markdown
from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Configure requests with retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# Simple text processing helpers
def truncate_text(text: str, max_chars: int = 512000) -> str:
    """Truncate text to maximum character length while preserving word boundaries"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit('. ', 1)[0] + '.'

# Create the application
app = FastAPI()

# Create all tables
Base.metadata.create_all(bind=engine)

origins = [
    "http://localhost:8001",
    "http://localhost:3001",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost",
]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# OpenAI client with timeout and environment API key
client = OpenAI(
    api_key=OPENAI_API_KEY, 
    timeout=120.0,  # Increase timeout to 120 seconds
    max_retries=3
)

# Initialize RAG processor after OpenAI client
rag_processor = RAGProcessor(OPENAI_API_KEY)

# Retry decorator for OpenAI calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_with_openai(text_chunk):
    try:
        if not text_chunk:
            return "Error: Empty text chunk"
            
        text_chunk = truncate_text(text_chunk)
        logger.info(f"Processing text chunk of length: {len(text_chunk)}")
                
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a scientific paper analyzer specializing in error detection and correction. 
                    Analyze the text for:
                    1. Calculation Errors: Check mathematical formulas, units, statistical calculations
                    2. Logical Errors: Identify flaws in reasoning, contradictions, unsupported conclusions
                    3. Methodological Errors: Examine research design, sampling, controls, procedures
                    4. Data Interpretation Errors: Look for misinterpretation of results, statistical errors
                    5. Formatting Issues: Check citations, structure, presentation
                    
                    For each error found:
                    - Clearly describe the error
                    - Explain why it's problematic
                    - Provide a specific solution or correction
                    - Use clear headings for each type of error"""
                },
                {
                    "role": "user",
                    "content": text_chunk
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Analysis error: {str(e)}"

# Parallel text extraction
def extract_text_from_page(page):
    return page.extract_text()

def validate_pdf(file_path):
    """Validate PDF file and return number of pages if valid"""
    try:
        # Try with PyMuPDF first
        doc = fitz.open(file_path)
        page_count = doc.page_count
        doc.close()
        return True, page_count
    except Exception as e:
        try:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                if len(pdf.pages) > 0:
                    return True, len(pdf.pages)
        except Exception as e:
            return False, str(e)
    return False, "Invalid PDF format"

def extract_text_safely(file_path):
    """Extract text from PDF using multiple methods"""
    text_chunks = []
    try:
        # Try PyMuPDF first
        doc = fitz.open(file_path)
        for page in doc:
            try:
                text = page.get_text()
                text_chunks.append(text)
            except Exception as e:
                text_chunks.append(f"[Error extracting page {page.number + 1}]")
        doc.close()
    except Exception as e:
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_chunks.append(text)
                        else:
                            text_chunks.append(f"[Empty page {i + 1}]")
                    except Exception as page_error:
                        text_chunks.append(f"[Error extracting page {i + 1}]")
        except Exception as pdf_error:
            raise Exception(f"Failed to extract text: {str(pdf_error)}")
    
    if not text_chunks:
        raise Exception("No text could be extracted from the PDF")
    
    return "\n".join(text_chunks)


def process_text_for_rag(text: str, document_metadata: dict) -> tuple[list[dict], dict]:
    """Process text through RAG pipeline"""
    try:
        chunks = rag_processor.create_chunks_with_metadata(text, document_metadata)
        stored_data = rag_processor.create_embeddings(chunks)
        return chunks, stored_data
    except Exception as e:
        logger.error(f"RAG processing error: {str(e)}")
        raise Exception(f"Failed to process text: {str(e)}")

async def analyze_chunks(chunks: list[dict], stored_data: dict) -> str:
    """Analyze most relevant chunks and combine results"""
    queries = [
        "Find calculation errors, mathematical mistakes, and unit conversion issues",
        "Identify logical fallacies and reasoning errors in arguments and conclusions",
        "Detect methodological issues in research design, sampling, and procedures",
        "Find errors in data interpretation, statistical analysis, and results presentation",
        "Check for formatting issues, citation errors, and structural problems"
    ]
    
    all_results = []
    for query in queries:
        relevant_texts = rag_processor.get_relevant_chunks(stored_data, query)
        for text in relevant_texts:
            analysis = await analyze_with_openai(text)
            if analysis and not analysis.startswith("Analysis error"):
                all_results.append(analysis)
    
    # Combine and structure results
    combined_analysis = "\n\n".join(all_results)
    return combined_analysis

@app.get("/")
async def root():
    return {"message": "AI Error Detector"}

@app.post("/api/pdfs/", response_model=PDFDocumentResponse)
async def create_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Create directory if it doesn't exist
        os.makedirs("media/pdfs", exist_ok=True)
        
        # Save file
        file_path = f"media/pdfs/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()

        # Create database entry
        db_pdf = PDFDocument(
            file_path=file_path,
            title=file.filename,
            processed=True
        )
        db.add(db_pdf)
        db.commit()
        db.refresh(db_pdf)

        return PDFDocumentResponse.from_orm(db_pdf)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/pdfs/", response_model=list[PDFDocumentResponse])
async def get_pdfs(db: Session = Depends(get_db)):
    try:
        pdfs = db.query(PDFDocument).all()
        response = [PDFDocumentResponse.from_orm(pdf) for pdf in pdfs]
        return JSONResponse(
            content=[{
                "id": pdf.id,
                "file_path": pdf.file_path,
                "title": pdf.title,
                "processed": pdf.processed,
                "created_at": pdf.created_at.isoformat(),
                "updated_at": pdf.updated_at.isoformat()
            } for pdf in pdfs],
            headers={
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdfs/{pdf_id}/check_paper_stream")
async def check_paper_stream(pdf_id: int, db: Session = Depends(get_db)):
    async def event_stream():
        try:
            document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
            if not document:
                yield f"data: {json.dumps({'error': 'PDF not found'})}\n\n"
                return

            if not os.path.exists(document.file_path):
                yield f"data: {json.dumps({'error': 'PDF file not found on server'})}\n\n"
                return

            yield "data: **Validating PDF...**\n\n"
            yield "data: \n\n"
            
            # Validate PDF
            is_valid, page_count_or_error = validate_pdf(document.file_path)
            if not is_valid:
                yield f"data: {json.dumps({'error': f'Invalid PDF file: {page_count_or_error}'})}\n\n"
                return

            yield "data: **Processing PDF...**\n\n"
            yield "data: \n\n"
            
            try:
                extracted_text = extract_text_safely(document.file_path)
                if not extracted_text.strip():
                    raise Exception("No valid text content extracted from PDF")
                
                # Check text length and process accordingly
                text_length = len(extracted_text)
                logger.info(f"Extracted text length: {text_length} characters")
                
                if text_length > 512000:
                    yield "data: **Text exceeds 128K, processing in chunks...**\n\n"
                    document_metadata = {
                        "title": document.title,
                        "pdf_id": document.id,
                        "file_path": document.file_path
                    }
                    chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
                    content_to_analyze = await analyze_chunks(chunks, vectorstore)
                else:
                    yield "data: **Processing complete text...**\n\n"
                    content_to_analyze = extracted_text
                yield "data: \n\n"
                
                print(content_to_analyze)
                yield "data: **Analyzing content...**\n\n"
                yield "data: \n\n"
                
                # Analyze with streaming response
                response = client.chat.completions.create(
                    model="o1-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": "I have a scientific paper that I would like to analyze these papers and identify any errors and give me the solution for each error. These errors could be in calculations, logic, methodology, data interpretation, or even formatting. This is the full paper: " + extracted_text
                        #     "content": """Analyze this scientific paper for errors and provide solutions.
                        #     For each error found:
                        #     1. ERROR TYPE: [Calculation/Logic/Methodology/Data/Formatting]
                        #     2. DESCRIPTION: Clear description of the error
                        #     3. IMPACT: Why this error is significant
                        #     4. SOLUTION: Specific steps to correct the error
                            
                        #     Format each error analysis as a separate section with clear headings."""
                        # },
                        # {
                        #     "role": "user",
                        #     "content": content_to_analyze
                        # }
                        }
                    ],
                    stream=True,
                )
                print(response)
                buffer = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        buffer += chunk.choices[0].delta.content
                        if any(delimiter in buffer for delimiter in ['.', '!', '?', '\n']):
                            parts = buffer.split('\n')
                            for i, part in enumerate(parts[:-1]):
                                if part.strip():
                                    converted_text = latex_to_markdown(part)
                                    yield f"data: {converted_text}\n\n"
                            buffer = parts[-1]
                        
                    await asyncio.sleep(0)
                
                # Convert any remaining text in buffer
                if buffer.strip():
                    converted_text = latex_to_markdown(buffer)
                    yield f"data: {converted_text}\n\n"
                    
                yield "data: [$Analysis Done.$]\n\n"

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                yield f"data: {json.dumps({'error': f'Processing failed: {str(e)}'})}\n\n"
                return

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )

async def analyze_with_orchestrator(text: str, metadata: dict) -> dict:
    orchestrator = Orchestrator(client)
    return await orchestrator.analyze_paper(text, metadata)

@app.get("/api/pdfs/{pdf_id}/check_paper_fully")
async def check_paper_stream_orchestrator(pdf_id: int, db: Session = Depends(get_db)):
    async def event_stream():
        try:
            document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
            if not document:
                yield f"data: {json.dumps({'error': 'PDF not found'})}\n\n"
                return

            if not os.path.exists(document.file_path):
                yield f"data: {json.dumps({'error': 'PDF file not found on server'})}\n\n"
                return

            yield "data: **Validating PDF...**\n\n"
            yield "data: \n\n"
            
            # Validate PDF
            is_valid, page_count_or_error = validate_pdf(document.file_path)
            if not is_valid:
                yield f"data: {json.dumps({'error': f'Invalid PDF file: {page_count_or_error}'})}\n\n"
                return

            yield "data: **Processing PDF...**\n\n"
            yield "data: \n\n"
            
            try:
                extracted_text = extract_text_safely(document.file_path)
                if not extracted_text.strip():
                    raise Exception("No valid text content extracted from PDF")
                
                # Check text length and process accordingly
                text_length = len(extracted_text)
                logger.info(f"Extracted text length: {text_length} characters")
                
                if text_length > 512000:
                    yield "data: **Text exceeds 128K, processing in chunks...**\n\n"
                    document_metadata = {
                        "title": document.title,
                        "pdf_id": document.id,
                        "file_path": document.file_path
                    }
                    chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
                    content_to_analyze = await analyze_chunks(chunks, vectorstore)
                else:
                    yield "data: **Processing complete text...**\n\n"
                    content_to_analyze = extracted_text
                yield "data: \n\n"
                
                print(content_to_analyze)
                yield "data: **Analyzing content...**\n\n"
                yield "data: \n\n"
                
                metadata = {
                    "title": document.title,
                    "pdf_id": document.id,
                    "file_path": document.file_path
                }

                yield "data: **Analyzing with specialized reviewers...**\n\n"
                
                analysis_result = await analyze_with_orchestrator(content_to_analyze, metadata)
                
                # Stream the results
                for agent_result in analysis_result["analysis"]:
                    yield f"data: **{agent_result['type'].upper()} ANALYSIS**\n\n"
                    yield f"data: {agent_result['findings']}\n\n"
                
                yield "data: **SUMMARY**\n\n"
                yield f"data: {json.dumps(analysis_result['summary'], indent=2)}\n\n"
                yield "data: [$Analysis Done.$]\n\n"

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                yield f"data: {json.dumps({'error': f'Processing failed: {str(e)}'})}\n\n"
                return

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )

@app.get("/api/pdfs/{pdf_id}/check_paper")
async def check_paper(pdf_id: int, db: Session = Depends(get_db)):
    try:
        document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="PDF not found")

        if not os.path.exists(document.file_path):
            raise HTTPException(status_code=404, detail="PDF file not found on server")
        
        # Validate PDF
        is_valid, page_count_or_error = validate_pdf(document.file_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {page_count_or_error}")

        # Extract text
        extracted_text = extract_text_safely(document.file_path)
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No valid text content extracted from PDF")
        
        # Process text based on length
        text_length = len(extracted_text)
        logger.info(f"Extracted text length: {text_length} characters")
        
        if text_length > 512000:
            document_metadata = {
                "title": document.title,
                "pdf_id": document.id,
                "file_path": document.file_path
            }
            chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
            content_to_analyze = await analyze_chunks(chunks, vectorstore)
        else:
            content_to_analyze = extracted_text

        # Analyze with orchestrator
        metadata = {
            "title": document.title,
            "pdf_id": document.id,
            "file_path": document.file_path
        }
        
        analysis_result = await analyze_with_orchestrator(content_to_analyze, metadata)
        
        return JSONResponse(
            content={
                "status": "success",
                "metadata": metadata,
                "analysis": analysis_result["analysis"]
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error in check_paper: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdfs/{pdf_id}/get_summary")
async def get_summary(pdf_id: int, db: Session = Depends(get_db)):
    document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="PDF not found")
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="PDF file not found on server")
    
    # Validate PDF
    is_valid, page_count_or_error = validate_pdf(document.file_path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {page_count_or_error}")

    # Extract text
    extracted_text = extract_text_safely(document.file_path)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No valid text content extracted from PDF")
    
    # Process text based on length
    text_length = len(extracted_text)
    logger.info(f"Extracted text length: {text_length} characters")
    
    if text_length > 512000:
        document_metadata = {
            "title": document.title,
            "pdf_id": document.id,
            "file_path": document.file_path
        }
        chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
        content_to_analyze = await analyze_chunks(chunks, vectorstore)
    else:
        content_to_analyze = extracted_text

    summary = {
        "authors": [],
        "paper_link": "",
        "homepage_link": "",
        "title": "",
        "summary": ""
    }
    # Extract authors and links using a separate OpenAI call
    link_response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user",
                "content": """Extract and generate the following from the text:
                1. Author names (as comma-separated list)
                2. Paper URL/DOI link
                3. Author/Institution homepage link
                4. Title of the paper
                5. Three-level summary of the paper:
                   - For children (simple, basic concepts)
                   - For college students (moderate technical detail)
                   - For PhD students (full technical depth)
                
                Format your response as JSON:
                {
                    "authors": "author1, author2, ...",
                    "paper_link": "url or doi",
                    "homepage_link": "url",
                    "title": "title of the paper",
                    "summary": {
                        "child": "simple explanation...",
                        "college": "moderate technical explanation...",
                        "phd": "detailed technical explanation..."
                    }
                }
                
                For summaries:
                - Child: Use simple words, avoid technical terms
                - College: Include key concepts and methods
                - PhD: Include technical details, methods, and implications
                
                If any item is not found, use empty string."""
            },
            {"role": "user", "content": content_to_analyze}
        ]
    )
    try:
        content = link_response.choices[0].message.content
        logger.info(f"Link extraction response content: {content}")
        
        # Remove ```json and ``` markers if present
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        
        if content:
            try:
                link_data = json.loads(content)
                logger.info(f"Parsed link data: {link_data}")
                summary["authors"] = [
                    author.strip() 
                    for author in link_data.get("authors", "").split(",")
                    if author.strip()
                ]
                summary["paper_link"] = link_data.get("paper_link", "")
                summary["homepage_link"] = link_data.get("homepage_link", "")
                summary["summary"] = link_data.get("summary", {
                    "child": "",
                    "college": "",
                    "phd": ""
                })
                summary["title"] = link_data.get("title", "")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                summary["authors"] = []
                summary["paper_link"] = ""
                summary["homepage_link"] = ""
                summary["title"] = ""
                summary["summary"] = {
                    "child": "",
                    "college": "",
                    "phd": ""
                }
        else:
            raise ValueError("Empty content in link extraction response")
    except Exception as e:
        logger.error(f"Failed to parse link extraction response: {str(e)}")
        summary["authors"] = []
        summary["paper_link"] = ""
        summary["homepage_link"] = ""
        summary["title"] = ""
        summary["summary"] = {
            "child": "",
            "college": "",
            "phd": ""
        }
    
    return JSONResponse(
            content={
                "status": "success",
                "summary": summary
            },
            status_code=200
        )
    
            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)