from rest_framework import viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action
from .models import Paper, PaperSummary, PaperAnalysis
from .serializers import PaperSerializer
from .services import (
    validate_pdf, extract_text_safely, process_text_for_rag, openai_api_calculate_cost,
    analyze_chunks, analyze_with_orchestrator, generate_paper_summary, generate_analysis_prompt, generate_summary_prompt
)
from .scrape import process_arxiv_paper

import tiktoken
from django.conf import settings
import arxivscraper
import os

class PaperViewSet(viewsets.ModelViewSet):
    queryset = Paper.objects.all()
    serializer_class = PaperSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        try:
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
            document_metadata = {
                "title": document.title,
                "paper_id": document.id,
                "file_path": document.file.path if document.file else None
            }
            content_to_analyze = ""
            if not document.has_analysis:
                is_valid, error = validate_pdf(document.file.path)
                if not is_valid:
                    return Response(
                        {"error": f"Invalid PDF file: {error}"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                # Extract and analyze text
                extracted_text = extract_text_safely(document.file.path)
                text_length = len(extracted_text)
                
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
            document_metadata = {
                "title": document.title,
                "paper_id": document.id,
                "file_path": document.file.path if document.file else None
            }
            content_to_analyze = ""
            if not document.has_summary:
                is_valid, error = validate_pdf(document.file.path)
                if not is_valid:
                    return Response(
                        {"error": f"Invalid PDF file: {error}"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Extract and analyze text
                extracted_text = extract_text_safely(document.file.path)
                text_length = len(extracted_text)

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
            sort_by = request.query_params.get('sort_by', 'total_errors')
            order = request.query_params.get('order', 'desc')
            start_idx = (page - 1) * page_size

            allowed_sort_fields = [  
                'total_errors',
                'math_errors',  
                'methodology_errors',  
                'logical_framework_errors',  
                'data_analysis_errors',  
                'technical_presentation_errors',  
                'research_quality_errors',  
            ]  
            
            if sort_by not in allowed_sort_fields:  
                sort_by = 'total_errors'  # Fallback to default if invalid  
                
            if order == 'asc':  
                order_prefix = ''  
            else:  
                order_prefix = '-' 
            order_by_field = f'{order_prefix}analysis__{sort_by}'
            papers = Paper.objects.select_related('analysis').filter(has_summary=True, has_analysis=True).order_by(order_by_field, '-created_at')  
            total_papers = papers.count()
            total_pages = (total_papers + page_size - 1) // page_size

            paginated_papers = papers[start_idx:start_idx + page_size]
            results = []

            for paper in paginated_papers:
                results.append({
                    'id': paper.id,
                    'title': paper.title,
                    'input_tokens': paper.input_tokens,
                    'output_tokens': paper.output_tokens,
                    'total_cost': paper.total_cost,
                    'paperSummary': PaperSummary.objects.filter(paper=paper).latest('generated_at').summary_data,
                    'paperAnalysis': PaperAnalysis.objects.filter(paper=paper).latest('analyzed_at').analysis_data,
                    'result_id': PaperAnalysis.objects.filter(paper=paper).latest('analyzed_at').id,
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
            papers = Paper.objects.filter(has_summary=True, has_analysis=True)
            total_papers = papers.count()
            total_analyses = PaperAnalysis.objects.count()
            analysis_results = PaperAnalysis.objects.all()
            error_counts = {
                'math_errors': 0,
                'methodology_errors': 0,
                'logical_framework_errors': 0,
                'data_analysis_errors': 0,
                'technical_presentation_errors': 0,
                'research_quality_errors': 0,
                'total_errors': 0
            }

            # Sum up errors from all analyses
            for analysis in analysis_results:                
                error_counts['math_errors'] += analysis.math_errors
                error_counts['methodology_errors'] += analysis.methodology_errors
                error_counts['logical_framework_errors'] += analysis.logical_framework_errors
                error_counts['data_analysis_errors'] += analysis.data_analysis_errors
                error_counts['technical_presentation_errors'] += analysis.technical_presentation_errors
                error_counts['research_quality_errors'] += analysis.research_quality_errors
                
            error_counts['total_errors'] = error_counts['math_errors'] + error_counts['methodology_errors'] + error_counts['logical_framework_errors'] + error_counts['data_analysis_errors'] + error_counts['technical_presentation_errors'] + error_counts['research_quality_errors']

            return Response({
                'status': 'success',
                'data': {
                    'totalPapers': total_papers,
                    'totalAnalyses': total_analyses,
                    'errorStatistics': {
                        'mathErrors': error_counts['math_errors'],
                        'methdologyErrors': error_counts['methodology_errors'],
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
                # analysis_data = {'analysis': [{'type': 'Methodological Issues', 'findings': [{'error': 'Lack of empirical evidence supporting the central claim', 'explanation': 'The paper proposes that the forbidden fruit was cannabis but does not provide empirical evidence, such as historical texts, linguistic analysis, or archaeological findings, to substantiate this claim.', 'solution': 'Include empirical research and evidence from credible historical, linguistic, or archaeological sources to support the interpretation.', 'location': "Throughout Sections 6 and 7 ('Author's Interpretation' and 'Mental and Logical Evidence Supporting the Theory of the Forbidden Fruit as Cannabis')", 'severity': 'high'}], 'counts': 1}, {'type': 'Logical Framework', 'findings': [{'error': 'Speculative arguments presented as conclusions', 'explanation': 'The author presents speculative interpretations without sufficient evidence, leading to conclusions that are not fully justified or logically derived from the premises.', 'solution': 'Frame the interpretations as hypotheses and ensure that conclusions are logically derived from well-substantiated arguments.', 'location': 'Sections 6 and 7', 'severity': 'high'}, {'error': 'Lack of engagement with existing scholarship', 'explanation': 'The paper does not critically engage with existing interpretations or scholarly work on the topic, missing an opportunity to position the argument within the broader academic discourse.', 'solution': 'Review and reference existing scholarly work on the topic, and explain how this new interpretation fits within or challenges current understandings.', 'location': 'Throughout the paper', 'severity': 'medium'}], 'counts': 2}, {'type': 'Technical Presentation', 'findings': [{'error': 'Citation inaccuracies and inconsistencies', 'explanation': 'Some references are improperly formatted, lack publication details, or are inconsistently presented, which affects the credibility and traceability of sources.', 'solution': 'Revise the references to ensure they are complete, accurately formatted, and consistent with a standard citation style (e.g., APA, MLA, Chicago).', 'location': 'References section and in-text citations throughout the paper', 'severity': 'medium'}, {'error': 'Writing clarity and grammatical issues', 'explanation': "Certain sentences are unclear or contain grammatical errors, which impede the reader's understanding and affect the professional tone of the paper.", 'solution': 'Proofread the paper carefully, or consider professional editing services to improve clarity, grammar, and overall readability.', 'location': 'Various sections throughout the paper', 'severity': 'low'}, {'error': 'Formatting inconsistencies', 'explanation': 'The paper exhibits inconsistencies in formatting, such as irregular headings, spacing, and alignment, which can distract the reader and detract from the professional presentation.', 'solution': 'Ensure consistent formatting throughout the document, adhering to guidelines provided by a specific style manual or journal requirements.', 'location': 'Throughout the paper', 'severity': 'low'}], 'counts': 3}, {'type': 'Research Quality', 'findings': [{'error': 'Potential for confirmation bias', 'explanation': 'The author presents evidence that primarily supports their hypothesis without adequately considering or addressing counterarguments or alternative interpretations.', 'solution': 'Acknowledge and address alternative perspectives and potential criticisms to strengthen the argument and demonstrate scholarly rigor.', 'location': 'Throughout Sections 6 and 7', 'severity': 'medium'}, {'error': 'Ethical considerations in interpretation', 'explanation': 'Reinterpreting religious texts in a way that significantly deviates from traditional understandings may require careful ethical consideration to respect religious sensitivities.', 'solution': 'Approach the reinterpretation with cultural and religious sensitivity, possibly including discussions on the implications and acknowledging the diversity of beliefs.', 'location': 'Sections 6, 7, and the Conclusion', 'severity': 'medium'}], 'counts': 2}, {'type': 'Logical Framework', 'findings': [{'error': 'Overgeneralization and unsupported causal links', 'explanation': 'The paper makes broad generalizations and asserts causal relationships without sufficient evidence, potentially leading to logical fallacies.', 'solution': 'Provide evidence for causal claims and avoid overgeneralizations by qualifying statements and acknowledging limitations.', 'location': "Section 7 ('Mental and Logical Evidence Supporting the Theory of the Forbidden Fruit as Cannabis')", 'severity': 'medium'}], 'counts': 1}], 'summary': {'total_errors': 9, 'major_concerns': ['Lack of empirical evidence to support the central claim', 'Speculative arguments presented without sufficient logical support', 'Citation inaccuracies and inconsistencies affecting credibility'], 'improvement_priority': ['First, incorporate empirical evidence from credible sources to substantiate the central hypothesis.', 'Second, strengthen the logical framework by ensuring conclusions are logically derived from well-supported premises.', 'Third, revise citations and references for accuracy and consistency with academic standards.', 'Fourth, address potential biases by engaging with existing scholarship and alternative viewpoints.', 'Fifth, improve writing clarity, grammar, and formatting to enhance readability and professional presentation.'], 'overall_assessment': "The paper presents an original and thought-provoking interpretation of the forbidden fruit as cannabis. However, significant methodological shortcomings and a lack of empirical evidence undermine the credibility of the argument. Strengthening the logical framework and adhering to academic standards in research and presentation are necessary to enhance the paper's academic quality.", 'quality_score': 4}, 'metadata': {'title': 'TheMotherofallsinsARTICLE.edited.June24 (1).docx.pdf', 'paper_id': 16, 'file_path': '/home/devai/Ai-check-backend/media/papers/TheMotherofallsinsARTICLE.edited.June24_1.docx.pdf'}}
                # Get error counts from each category in analysis
                analysis.math_errors = analysis_data['analysis'][0]['counts']  # Mathematical
                analysis.methodology_errors = analysis_data['analysis'][1]['counts']  # Methodological
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
                    analysis.methodology_errors + 
                    analysis.logical_framework_errors +
                    analysis.data_analysis_errors +
                    analysis.technical_presentation_errors +
                    analysis.research_quality_errors
                )
                
                analysis.save()
                
            except Exception as e:
                return Response(f"Error processing analysis {analysis.id}: {str(e)}")
        
        return Response({"status": "success", "message": "Analysis conversion completed"})
    
    @action(detail=False, methods=['get'])
    def calculate_cost(self, request, pk=None):
        papers = Paper.objects.select_related('analysis', 'summaries').all()
        encoding = tiktoken.encoding_for_model('gpt-4')
        for paper in papers:
            full_path = (os.path.join(settings.MEDIA_ROOT, str(paper.file)))
            is_valid, error = validate_pdf(full_path)
            if not is_valid:
                return Response(
                    {"error": f"Invalid PDF file: {error}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract and analyze text
            extracted_text = extract_text_safely(full_path)
            summary_prompt = generate_analysis_prompt(extracted_text)
            analysis_prompt = generate_analysis_prompt(extracted_text)
            prompt_tokens = len(encoding.encode(summary_prompt + analysis_prompt))
            completion_tokens = len(encoding.encode(str(paper.analysis.analysis_data))) + len(encoding.encode(str(paper.summaries.summary_data)))
            total_cost = openai_api_calculate_cost(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, "o1-preview")
            paper.input_tokens = prompt_tokens
            paper.output_tokens = completion_tokens
            paper.total_cost = total_cost
            paper.save()

        return Response({
            'status': 'success',
        })

    @action(detail=False, methods=['get'])
    def scrape_papers(self, request, pk=None):
        category = request.query_params.get('category', "physics:nucl-ex")
        date_from = request.query_params.get('date_from', "2025-01-20")
        date_until = request.query_params.get('date_until', "2025-01-25")
        scraper = arxivscraper.Scraper(category = category, date_from = date_from, date_until = date_until)
        results = scraper.scrape()                    
        return Response({
            'status': 'success',
            'data': results
        })
    
    @action(detail=False, methods=['get'])
    def process_paper(self, request, pk=None):
        paper_id = request.query_params.get('id', "2101.00001")
        process_arxiv_paper(paper_id)
        return Response({
            'status': 'success',
        })

