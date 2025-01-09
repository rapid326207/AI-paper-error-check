from sqlalchemy import Column, Integer, String, Boolean, DateTime
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os
from database import Base

class PDFDocument(Base):
    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String)
    title = Column(String)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists("media/pdfs"):
            os.makedirs("media/pdfs", exist_ok=True)

class PDFDocumentCreate(BaseModel):
    file_path: str
    title: str
    processed: bool = False

class PDFDocumentResponse(BaseModel):
    id: int
    file_path: str
    title: str
    processed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 