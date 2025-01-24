from openai import OpenAI
import os
import fitz
import PyPDF2
import json
from typing import Dict, List, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from papers.utils.rag import RAGProcessor
from celery import shared_task
from papers.models import Paper, PaperAnalysis, PaperSummary
from django.utils import timezone
from docx2python import docx2python
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize clients
OPENAI_API_KEY = "sk-proj...."

client = OpenAI(
    api_key=OPENAI_API_KEY, 
    timeout=240.0,
    max_retries=3
)

rag_processor = RAGProcessor(OPENAI_API_KEY)


def truncate_text(text: str, max_chars: int = 512000) -> str:
    """Truncate text to maximum character length while preserving word boundaries"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit('. ', 1)[0] + '.'


def clean_text(self, text):
    """Clean text by escaping special characters for JSON"""
    if isinstance(text, str):
        # Replace backslashes with double backslashes
        text = text.replace('\\', '\\\\')
        # Escape other special characters
        text = json.dumps(text)[1:-1]
    return text
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@shared_task()
def analyze_with_openai(text_chunk):
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

@shared_task()
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

@shared_task()
def extract_text_safely(file_path):
    """Extract text from PDF using multiple methods"""
    text_chunks = []
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        # Try PyMuPDF first
        if file_ext == '.pdf':
            doc = fitz.open(file_path)
            for page in doc:
                try:
                    text = page.get_text()
                    text_chunks.append(text)
                except Exception as e:
                    text_chunks.append(f"[Error extracting page {page.number + 1}]")
            doc.close()
            
        # Handle DOCX files    
        elif file_ext == '.docx':
            with docx2python(file_path) as docx_content:
                text_chunks.append(docx_content.text)
                
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return ' '.join(text_chunks)
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

@shared_task()
def process_text_for_rag(text: str, document_metadata: dict) -> tuple[list[dict], dict]:
    """Process text through RAG pipeline"""
    try:
        chunks = rag_processor.create_chunks_with_metadata(text, document_metadata)
        stored_data = rag_processor.create_embeddings(chunks)
        return chunks, stored_data
    except Exception as e:
        logger.error(f"RAG processing error: {str(e)}")
        raise Exception(f"Failed to process text: {str(e)}")

@shared_task()
def analyze_chunks(chunks: list[dict], stored_data: dict) -> str:
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
            analysis = analyze_with_openai(text)
            if analysis and not analysis.startswith("Analysis error"):
                all_results.append(analysis)
    
    return "\n\n".join(all_results)

@shared_task()
def analyze_paper_comprehensive(text: str):
    o1_response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {
                "role": "user",
                "content": """Act as an expert academic research reviewer. Analyze the following paper comprehensively and provide me a structured analysis in JSON format.

                1. First provide me detailed analysis of following error categories:
                    Focus on these key error categories:

                    1). Mathematical and Calculation Analysis:
                    - Mathematical equations and formulas
                    - Statistical analyses and computations
                    - Data processing and numerical results
                    - Unit conversions and measurements

                    2). Methodological Issues:
                    - Research design and methodology
                    - Sampling procedures and sample size
                    - Data collection methods
                    - Control measures
                    - Variable operationalization

                    3). Logical Framework:
                    - Theoretical foundations
                    - Argument structure and flow
                    - Causal relationships
                    - Hypothesis formulation
                    - Conclusions validity

                    4). Data Analysis:
                    - Statistical test appropriateness
                    - Results interpretation
                    - Data visualization
                    - Significance testing
                    - Effect size reporting

                    5). Technical Presentation:
                    - Figure and table accuracy
                    - Formatting consistency
                    - Citation accuracy
                    - Writing clarity
                    - Structural organization

                    6). Research Quality:
                    - Internal/external validity
                    - Reliability measures
                    - Methodological bias
                    - Ethical considerations
                    - Replicability

                    For each error found, provide:
                    - Clear error description
                    - Severity rating (high/medium/low)
                    - Specific text location
                    - Improvement recommendation
                    - Reference to academic standards

                2. Then provide me a comprehensive summary including:
                    - Total error count
                    - Major concerns
                    - Improvement priorities
                    - Overall quality assessment
                    - Quality score (1-10)

                Must provide all 6 error categories.(if the error count is zero) Return BOTH sections in this exact JSON structure:
                {
                    "analysis": [
                        {
                            "type": "category_name",
                            "findings": [
                                {
                                    "error": "specific error title",
                                    "explanation": "detailed description",
                                    "solution": "recommended fix",
                                    "location": "where in the paper",
                                    "severity": "high/medium/low"
                                }
                            ],
                            "counts": "number_of_errors"
                        }
                    ],
                    "summary": {
                        "total_errors": "total number of errors found",
                        "major_concerns": [
                            "list of most critical issues requiring immediate attention"
                        ],
                        "improvement_priority": [
                            "prioritized list of what should be fixed first, second, etc."
                        ],
                        "overall_assessment": "brief evaluation of paper quality",
                        "quality_score": "numerical score 1-10"
                    }
                }

                Analyze this paper: """ + text
            },
     
        ]
    )
    o1_response_content = o1_response.choices[0].message.content
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user", 
                "content": f"""
                Given the following data, format it with the given response format: {o1_response_content}
                """
            }
        ],
        response_format={"type":"json_object"},
    )
    
    return response.choices[0].message.content

@shared_task()
def analyze_with_orchestrator(text: str, metadata: dict) -> Dict:
    try:
        # Get the paper object
        paper = Paper.objects.select_related('summaries').get(id=metadata.get('paper_id'))
        encoding = tiktoken.encoding_for_model('gpt-4')
        # Check if analysis already exists
        if paper.has_analysis:
            existing_analysis = PaperAnalysis.objects.filter(paper=paper).latest('analyzed_at')
            return existing_analysis.analysis_data
        
        # Perform the analysis
        analysis_result = analyze_paper_comprehensive(text)
        analysis_result["metadata"] = metadata
        # Save the analysis result
        paper_analysis = PaperAnalysis.objects.create(
            paper = paper,
            analysis_data = analysis_result,
            total_errors = analysis_result["summary"]["total_errors"],
            math_errors = analysis_result["analysis"][0]["counts"],
            methodology_errors = analysis_result["analysis"][1]["counts"],
            logical_framework_errors = analysis_result["analysis"][2]["counts"],
            data_analysis_errors = analysis_result["analysis"][3]["counts"],
            technical_presentation_errors = analysis_result["analysis"][4]["counts"],
            research_quality_errors = analysis_result["analysis"][5]["counts"],
            analyzed_at=timezone.now()
        )
        
        # Update paper status
        paper.has_analysis = True
        summary_prompt = generate_analysis_prompt(text)
        analysis_prompt = generate_analysis_prompt(text)
        prompt_tokens = len(encoding.encode(summary_prompt + analysis_prompt))
        completion_tokens = len(encoding.encode(str(analysis_result))) + len(encoding.encode(str(paper.summaries.summary_data)))
        total_cost = openai_api_calculate_cost(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, "o1-preview")
        paper.input_tokens = prompt_tokens
        paper.output_tokens = completion_tokens
        paper.total_cost = total_cost
        paper.save()
        
        return analysis_result
    except Paper.DoesNotExist:
        logger.error(f"Paper with ID {metadata.get('paper_id')} not found")
        raise Exception("Paper not found")
    except Exception as e:
        logger.error(f"Orchestrator error: {str(e)}")
        raise Exception(str(e))

@shared_task()
def generate_paper_summary(content: str, metadata: dict):
    try:
        # Get the paper object
        paper = Paper.objects.get(id=metadata.get('paper_id'))
        
        # Check if summary already exists
        if paper.has_summary:
            existing_summary = PaperSummary.objects.filter(paper=paper).latest('generated_at')
            return existing_summary.summary_data
            
        # Generate the summary
        o1_response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": """As a scientific paper analysis expert, extract and analyze the following with high precision:

                    1. METADATA EXTRACTION:
                    - Author names with affiliations (comma-separated)
                    - Paper DOI/URL (verify format)
                    - Author/Institution links (including ORCID if available)
                    - Full paper title
                    - Publication date and journal/conference
                    - Keywords and research areas

                    2. HIERARCHICAL SUMMARY ANALYSIS:
                    Create detailed but concise summaries (maximum 250 words each) that capture all key aspects:
                    
                    a) Basic Level (General Audience):
                        Explain the research, its findings, and real-world impact using everyday language 
                        and clear examples that anyone can understand. Focus on practical implications 
                        and benefits to society.
                    
                    b) Intermediate Level (Advanced Undergrad/Graduate):
                        Present the research objectives, methodology, and results using appropriate 
                        technical terminology while maintaining clarity. Include key statistical 
                        findings and theoretical framework references.
                    
                    c) Expert Level (Specialist/PhD):
                        Provide a technical analysis covering methodology, implementation details, 
                        statistical significance, theoretical implications, and research gaps. 
                        Include critical evaluation and future research directions.

                    3. TECHNICAL METRICS:
                    - Sample size and methodology rigor
                    - Statistical methods used
                    - Validation approaches
                    - Technical limitations

                    Guidelines for summaries:
                    1. Basic: Use analogies and real-world examples, avoid jargon
                    2. Intermediate: Balance technical accuracy with accessibility
                    3. Expert: Maintain technical rigor while being concise
                    4. Each summary must not exceed 250 words
                    5. Include essential information without redundancy
                    6. Use proper scientific writing style appropriate for each level

                    Ensure:
                    1. Summaries are clear and well-structured
                    2. Technical accuracy is maintained
                    3. Word limit is strictly observed
                    4. Key findings are highlighted
                    5. Empty fields use null instead of empty string

                    Return only as a JSON with the following format :
                    {
                        "metadata": {
                            "authors": "["author1 (affiliation)", "author2 (affiliation)", ...]",
                            "paper_link": "doi/url",
                            "institution_links": ["url1", "url2"],
                            "title": "complete title",
                            "publication_info": {
                                "date": "YYYY-MM-DD",
                                "journal": "name",
                                "keywords": ["keyword1", "keyword2"]
                            }
                        },
                        "summary": {
                            "child": "Clear explanation in everyday language (max 250 words)",
                            "college": "Technical summary with key findings (max 250 words)",
                            "phd": "Detailed technical analysis (max 250 words)"
                        },
                        "technical_assessment": {
                            "methodology_score": "1-10",
                            "statistical_rigor": "1-10",
                            "validation_quality": "1-10",
                            "technical_depth": "1-10"
                        }
                    }"""
                },
                {"role": "user", "content": content}
            ]
        )
        o1_response_content = o1_response.choices[0].message.content
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user", 
                    "content": f"""
        Given the following data, format it with the given response format: {o1_response_content}
        """
                }
            ],
            response_format={"type":"json_object"},
        )
        summary_result = response.choices[0].message.content
        # Save the summary
        paper_summary = PaperSummary.objects.create(
            paper=paper,
            summary_data=summary_result,
            generated_at=timezone.now()
        )
    
        # Update paper status if needed
        paper.has_summary = True  # Add this field to Paper model if needed
        paper.save()
        
        return summary_result
    except Paper.DoesNotExist:
        logger.error(f"Paper with ID {metadata.get('paper_id')} not found")
        raise Exception("Paper not found")
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        raise Exception(str(e))

@shared_task()
def openai_api_calculate_cost(prompt_tokens, completion_tokens, total_tokens, model="o1-preview"):
    pricing = {
        'gpt-3.5-turbo-1106': {
            'prompt': 0.001,
            'completion': 0.002,
        },
        'gpt-4-1106-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'o1-mini': {
            'prompt': 0.003,
            'completion': 0.012,
        },
        'o1-preview': {
            'prompt': 0.015,
            'completion': 0.06,
        }

    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = prompt_tokens * model_pricing['prompt'] / 1000
    completion_cost = completion_tokens * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    # round to 6 decimals
    total_cost = round(total_cost, 6)

    print(f"\nTokens used:  {prompt_tokens:,} prompt + {completion_tokens:,} completion = {total_tokens:,} tokens")
    print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

@shared_task()
def generate_summary_prompt(content:str):
    return """As a scientific paper analysis expert, extract and analyze the following with high precision:

                    1. METADATA EXTRACTION:
                    - Author names with affiliations (comma-separated)
                    - Paper DOI/URL (verify format)
                    - Author/Institution links (including ORCID if available)
                    - Full paper title
                    - Publication date and journal/conference
                    - Keywords and research areas

                    2. HIERARCHICAL SUMMARY ANALYSIS:
                    Create detailed but concise summaries (maximum 250 words each) that capture all key aspects:
                    
                    a) Basic Level (General Audience):
                        Explain the research, its findings, and real-world impact using everyday language 
                        and clear examples that anyone can understand. Focus on practical implications 
                        and benefits to society.
                    
                    b) Intermediate Level (Advanced Undergrad/Graduate):
                        Present the research objectives, methodology, and results using appropriate 
                        technical terminology while maintaining clarity. Include key statistical 
                        findings and theoretical framework references.
                    
                    c) Expert Level (Specialist/PhD):
                        Provide a technical analysis covering methodology, implementation details, 
                        statistical significance, theoretical implications, and research gaps. 
                        Include critical evaluation and future research directions.

                    3. TECHNICAL METRICS:
                    - Sample size and methodology rigor
                    - Statistical methods used
                    - Validation approaches
                    - Technical limitations

                    Format response as JSON:
                    {
                        "metadata": {
                            "authors": "["author1 (affiliation)", "author2 (affiliation)", ...]",
                            "paper_link": "doi/url",
                            "institution_links": ["url1", "url2"],
                            "title": "complete title",
                            "publication_info": {
                                "date": "YYYY-MM-DD",
                                "journal": "name",
                                "keywords": ["keyword1", "keyword2"]
                            }
                        },
                        "summary": {
                            "child": "Clear explanation in everyday language (max 250 words)",
                            "college": "Technical summary with key findings (max 250 words)",
                            "phd": "Detailed technical analysis (max 250 words)"
                        },
                        "technical_assessment": {
                            "methodology_score": "1-10",
                            "statistical_rigor": "1-10",
                            "validation_quality": "1-10",
                            "technical_depth": "1-10"
                        }
                    }

                    Guidelines for summaries:
                    1. Basic: Use analogies and real-world examples, avoid jargon
                    2. Intermediate: Balance technical accuracy with accessibility
                    3. Expert: Maintain technical rigor while being concise
                    4. Each summary must not exceed 250 words
                    5. Include essential information without redundancy
                    6. Use proper scientific writing style appropriate for each level

                    Ensure:
                    1. Summaries are clear and well-structured
                    2. Technical accuracy is maintained
                    3. Word limit is strictly observed
                    4. Key findings are highlighted
                    5. Empty fields use null instead of empty string""" + content

@shared_task()
def generate_analysis_prompt(content:str):
    return """Act as an expert academic research reviewer. Analyze the following paper comprehensively and provide me a structured analysis in JSON format.

                1. First provide me detailed analysis of following error categories:
                    Focus on these key error categories:

                    1). Mathematical and Calculation Analysis:
                    - Mathematical equations and formulas
                    - Statistical analyses and computations
                    - Data processing and numerical results
                    - Unit conversions and measurements

                    2). Methodological Issues:
                    - Research design and methodology
                    - Sampling procedures and sample size
                    - Data collection methods
                    - Control measures
                    - Variable operationalization

                    3). Logical Framework:
                    - Theoretical foundations
                    - Argument structure and flow
                    - Causal relationships
                    - Hypothesis formulation
                    - Conclusions validity

                    4). Data Analysis:
                    - Statistical test appropriateness
                    - Results interpretation
                    - Data visualization
                    - Significance testing
                    - Effect size reporting

                    5). Technical Presentation:
                    - Figure and table accuracy
                    - Formatting consistency
                    - Citation accuracy
                    - Writing clarity
                    - Structural organization

                    6). Research Quality:
                    - Internal/external validity
                    - Reliability measures
                    - Methodological bias
                    - Ethical considerations
                    - Replicability

                    For each error found, provide:
                    - Clear error description
                    - Severity rating (high/medium/low)
                    - Specific text location
                    - Improvement recommendation
                    - Reference to academic standards

                2. Then provide me a comprehensive summary including:
                    - Total error count
                    - Major concerns
                    - Improvement priorities
                    - Overall quality assessment
                    - Quality score (1-10)

                Must provide all 6 error categories.(if the error count is zero) Return BOTH sections in this exact JSON structure:
                {
                    "analysis": [
                        {
                            "type": "category_name",
                            "findings": [
                                {
                                    "error": "specific error title",
                                    "explanation": "detailed description",
                                    "solution": "recommended fix",
                                    "location": "where in the paper",
                                    "severity": "high/medium/low"
                                }
                            ],
                            "counts": "number_of_errors"
                        }
                    ],
                    "summary": {
                        "total_errors": "total number of errors found",
                        "major_concerns": [
                            "list of most critical issues requiring immediate attention"
                        ],
                        "improvement_priority": [
                            "prioritized list of what should be fixed first, second, etc."
                        ],
                        "overall_assessment": "brief evaluation of paper quality",
                        "quality_score": "numerical score 1-10"
                    }
                }

                Analyze this paper: """ + content