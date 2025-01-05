from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action
import PyPDF2
import io
from .models import PDFDocument
from .serializers import PDFDocumentSerializer
from openai import OpenAI
from spire.pdf import PdfDocument, PdfTextExtractOptions, PdfTextExtractor
import os

class PDFDocumentViewSet(viewsets.ModelViewSet):
    queryset = PDFDocument.objects.all()
    serializer_class = PDFDocumentSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        try:
            pdf_file = request.FILES.get('file')
            if not pdf_file:
                return Response({'error': 'No PDF file provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Process PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()

            # Save file and create document
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid():
                document = serializer.save()
                document.processed = True
                document.save()

                return Response({
                    'message': 'PDF processed successfully',
                    'text_content': text_content,
                    'document_id': document.id
                }, status=status.HTTP_201_CREATED)

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def check_paper(self, request, pk=None):
        try:
            document = self.get_object()
            if not document.file:
                return Response({'error': 'No PDF file found'}, status=status.HTTP_404_NOT_FOUND)

            # Initialize OpenAI client
            client = OpenAI(api_key="...")
            
            # Extract text from PDF
            pdf = PdfDocument()
            file_path = document.file.path
            
            if not os.path.exists(file_path):
                return Response({'error': 'PDF file not found on server'}, 
                              status=status.HTTP_404_NOT_FOUND)

            # Extract text using Spire.PDF
            extract_options = PdfTextExtractOptions()
            extracted_text = ""
            pdf.LoadFromFile(file_path)
            
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

            return Response({
                'message': 'Paper analysis completed',
                'analysis': analysis_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
