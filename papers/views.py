from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action
from .models import Paper
from .serializers import PaperSerializer
from .services import (
    validate_pdf, extract_text_safely, process_text_for_rag,
    analyze_chunks, analyze_with_orchestrator, generate_paper_summary
)


class PaperViewSet(viewsets.ModelViewSet):
    queryset = Paper.objects.all()
    serializer_class = PaperSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        try:
            print(request)
            pdf_file = request.FILES.get('file')
            if not pdf_file:
                return Response({'error': 'No PDF file provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Save file and create document
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid():
                document = serializer.save()
                document.processed = False
                document.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def check_paper(self, request, pk=None):
        document = self.get_object()
        try:
            # Validate PDF
            is_valid, error = validate_pdf(document.file.path)
            if not is_valid:
                return Response(
                    {"error": f"Invalid PDF file: {error}"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract and analyze text
            extracted_text = extract_text_safely(document.file.path)
            text_length = len(extracted_text)
            document_metadata = {
                "title": document.title,
                "paper_id": document.id,
                "file_path": document.file.path
            }
            if text_length > 512000:
          
                chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
                content_to_analyze = analyze_chunks(chunks, vectorstore)
            else:
                content_to_analyze = extracted_text

            # Generate summary
            analysis_result = analyze_with_orchestrator(content_to_analyze, document_metadata)

            return Response({
                "status": "success",
                "analysis": analysis_result['analysis'],
                "summary": analysis_result['summary'],
                "metadata": analysis_result['metadata']
            })

        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def get_summary(self, request, pk=None):
        document = self.get_object()
        try:
            # Validate PDF
            is_valid, error = validate_pdf(document.file.path)
            if not is_valid:
                return Response(
                    {"error": f"Invalid PDF file: {error}"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract and analyze text
            extracted_text = extract_text_safely(document.file.path)
            text_length = len(extracted_text)
            document_metadata = {
                "title": document.title,
                "paper_id": document.id,
                "file_path": document.file.path
            }

            if text_length > 512000:
                chunks, vectorstore = process_text_for_rag(extracted_text, document_metadata)
                content_to_analyze = analyze_chunks(chunks, vectorstore)
            else:
                content_to_analyze = extracted_text

            # Generate summary
            summary = generate_paper_summary(content_to_analyze, document_metadata)
            return Response({
                "status": "success",
                "summary": summary
            })

        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
