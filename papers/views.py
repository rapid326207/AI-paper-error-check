from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action
from .models import Paper, PaperSummary, PaperAnalysis
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

    @action(detail=False, methods=['get'])
    def get_analyzed_results(self, request, pk=None):
        try:
            page = int(request.query_params.get('page', 1))
            page_size = request.query_params.get('page_size', 3)
            start_idx = (page - 1) * page_size

            papers = Paper.objects.filter(has_summary=True, has_analysis=True).order_by('-created_at')
            total_papers = papers.count()
            total_pages = (total_papers + page_size - 1) // page_size

            paginated_papers = papers[start_idx:start_idx + page_size]
            results = []

            for paper in paginated_papers:
                results.append({
                    'id': paper.id,
                    'title': paper.title,
                    'paperSummary': PaperSummary.objects.filter(paper=paper).latest('generated_at').summary_data,
                    'paperAnalysis': PaperAnalysis.objects.filter(paper=paper).latest('analyzed_at').analysis_data,
                })

            return Response({
                'status': 'success',
                'data': results,
                'pagination': {
                    'currentPage': page,
                    'totalPages': total_pages,
                    'totalItems': total_papers,
                    'itemsPerPage': page_size
                }
            })
        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def get_statistics(self,request,pk=None):
        try:
            papers = Paper.objects.all()
            total_papers = papers.count()
            total_analyses = PaperAnalysis.objects.count()
            analysis_results = PaperAnalysis.objects.all()
            error_counts = {
                'math_errors': 0,
                'methdology_errors': 0,
                'logical_framework_errors': 0,
                'data_analysis_errors': 0,
                'technical_presentation_errors': 0,
                'research_quality_errors': 0,
                'total_errors': 0
            }

            # Sum up errors from all analyses
            for analysis in analysis_results:                
                error_counts['math_errors'] += analysis.math_errors
                error_counts['methdology_errors'] += analysis.methdology_errors
                error_counts['logical_framework_errors'] += analysis.logical_framework_errors
                error_counts['data_analysis_errors'] += analysis.data_analysis_errors
                error_counts['technical_presentation_errors'] += analysis.technical_presentation_errors
                error_counts['research_quality_errors'] += analysis.research_quality_errors
                
            error_counts['total_errors'] = error_counts['math_errors'] + error_counts['methdology_errors'] + error_counts['logical_framework_errors'] + error_counts['data_analysis_errors'] + error_counts['technical_presentation_errors'] + error_counts['research_quality_errors']

            return Response({
                'status': 'success',
                'data': {
                    'totalPapers': total_papers,
                    'totalAnalyses': total_analyses,
                    'errorStatistics': {
                        'mathErrors': error_counts['math_errors'],
                        'methdologyErrors': error_counts['methdology_errors'],
                        'logicalFrameworkErrors': error_counts['logical_framework_errors'],
                        'dataAnalysisErrors': error_counts['data_analysis_errors'],
                        'technicalPresentationErrors': error_counts['technical_presentation_errors'],
                        'researchQualityErrors': error_counts['research_quality_errors'],
                        'totalErrors': error_counts['total_errors']
                    }
                }
            })

        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
    @action(detail=False, methods=['get'])
    def convert_analysis(self, request, pk=None): 
        analysises = PaperAnalysis.objects.all()
        for analysis in analysises:
            try:
                analysis_data = analysis.analysis_data
                # Get error counts from each category in analysis
                analysis.math_errors = analysis_data['analysis'][0]['counts']  # Mathematical
                analysis.methdology_errors = analysis_data['analysis'][1]['counts']  # Methodological
                if len(analysis_data['analysis']) > 2:
                    analysis.logical_framework_errors = analysis_data['analysis'][2]['counts']  # Logical
                if len(analysis_data['analysis']) > 3:
                    analysis.data_analysis_errors = analysis_data['analysis'][3]['counts']  # Data Analysis
                if len(analysis_data['analysis']) > 4:
                   analysis.technical_presentation_errors = analysis_data['analysis'][4]['counts']  # Technical
                if len(analysis_data['analysis']) > 5:
                   analysis.research_quality_errors = analysis_data['analysis'][5]['counts']  # Research Quality
                
                # Calculate total errors
                analysis.total_errors = (
                    analysis.math_errors +
                    analysis.methdology_errors + 
                    analysis.logical_framework_errors +
                    analysis.data_analysis_errors +
                    analysis.technical_presentation_errors +
                    analysis.research_quality_errors
                )
                
                analysis.save()
                
            except Exception as e:
                return Response(f"Error processing analysis {analysis.id}: {str(e)}")
        
        return Response({"status": "success", "message": "Analysis conversion completed"})