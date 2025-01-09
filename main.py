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

from models import PDFDocument, PDFDocumentCreate, PDFDocumentResponse
from database import get_db, engine, Base

# Create the application
app = FastAPI()

# Create all tables
Base.metadata.create_all(bind=engine)

origins = [
    "http://localhost:3002",
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

# OpenAI client
client = OpenAI(api_key="sk-proj....")

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


@app.get("/api/pdfs/{pdf_id}/check_paper")
async def check_paper(pdf_id: int, db: Session = Depends(get_db)):
    try:
        document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()

        if not document:
            return JSONResponse(
                content={'error': 'PDF not found'},
                status_code=404
            )

        pdf = PdfDocument()
        pdf.LoadFromFile(document.file_path)
        
        extract_options = PdfTextExtractOptions()
        extracted_text = ""
        
        for i in range(pdf.Pages.Count):
            page = pdf.Pages.get_Item(i)
            text_extractor = PdfTextExtractor(page)
            text = text_extractor.ExtractText(extract_options)
            extracted_text += text

        # Clean up the text
        cleaned_text = extracted_text.replace(
            "Evaluation Warning : The document was created with Spire.PDF for Python.",
            ""
        )

        # Analyze the paper using OpenAI
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I have a scientific paper that I would like to analyze these papers and identify any errors and give me the solution for each error. These errors could be in calculations, logic, methodology, data interpretation, or even formatting. This is the full paper: " + cleaned_text
                    )
                }
            ]
        )

        analysis_results = response.choices[0].message.content

        return JSONResponse(
            content={
                'message': 'Paper analysis completed',
                'analysis': analysis_results
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={'error': str(e)},
            status_code=400
        )

@app.get("/api/pdfs/{pdf_id}/check_paper_stream")
async def check_paper_stream(pdf_id: int, db: Session = Depends(get_db)):
    async def event_stream():
        try:
            # Get document from database
            document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
            if not document:
                yield f"data: {json.dumps({'error': 'PDF not found'})}\n\n"
                return

            if not os.path.exists(document.file_path):
                yield f"data: {json.dumps({'error': 'PDF file not found on server'})}\n\n"
                return

            yield "data: **Processing PDF...**\n\n"
            yield "data:\n\n"
            # Extract text using PyPDF2
            with open(document.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                extracted_text = ""
                
                for page in pdf_reader.pages:
                    extracted_text += page.extract_text()

            yield "data: **Analyzing content...**\n\n"
            yield "data:\n\n"
            
            response = client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "I have a scientific paper that I would like to analyze these papers and identify any errors and give me the solution for each error. "
                            "These errors could be in calculations, logic, methodology, data interpretation, or even formatting. "
                            "This is the full paper: " + extracted_text
                        )
                    }
                ],
                stream=True
            )

            buffer = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    buffer += chunk.choices[0].delta.content
                    # Check if we have complete sentences or paragraphs
                    if any(delimiter in buffer for delimiter in ['.', '!', '?', '\n']):
                        # Split by sentences while preserving line breaks
                        parts = buffer.split('\n')
                        for i, part in enumerate(parts[:-1]):  # Process all but the last part
                            if part.strip():  # Only send non-empty lines
                                yield f"data: {part}\n\n"
                        buffer = parts[-1]  # Keep the last part in buffer
                    
                await asyncio.sleep(0)
            
            # Send any remaining content in the buffer
            if buffer.strip():
                yield f"data: {buffer}\n\n"
                
            yield "data: [$Analysis Done.$]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 