# ...existing code...

def analyze_paper_comprehensive(text):
    prompt = """Act as an expert academic research reviewer with extensive experience in academic paper analysis. 
    Conduct a thorough examination of the following academic paper, focusing on these key areas:

    1. Mathematical and Calculation Analysis:
    - Verify all mathematical equations, formulas, and calculations
    - Check statistical analyses and numerical results
    - Identify computational errors or inconsistencies

    2. Methodological Assessment:
    - Evaluate research design appropriateness
    - Assess sampling methods and sample size adequacy
    - Review data collection procedures
    - Examine variable operationalization
    - Check validity and reliability measures

    3. Logical and Theoretical Framework:
    - Evaluate theoretical foundations
    - Assess argument coherence and flow
    - Identify logical fallacies or gaps
    - Check causality claims
    - Review hypothesis formulation

    4. Data Analysis and Results:
    - Verify statistical test appropriateness
    - Check data interpretation accuracy
    - Assess result presentation clarity
    - Evaluate findings' significance
    - Review data visualization accuracy

    5. Technical and Presentation:
    - Check figures, tables, and graphs for accuracy
    - Assess formatting consistency
    - Review citations and references
    - Evaluate writing clarity and structure
    - Check for typographical errors

    6. Research Quality Indicators:
    - Assess research validity (internal and external)
    - Evaluate reliability measures
    - Check for bias in methodology and interpretation
    - Review ethical considerations
    - Assess replicability

    For each area, provide:
    1. A clear description of any errors or issues found
    2. The severity level of each issue (Critical, Major, Minor)
    3. Specific examples from the text
    4. Recommendations for improvement
    5. Citations of relevant academic standards or best practices

    Please organize your response in a structured format with clear headings and bullet points for each major category.

    Here's the paper for analysis:
    {text}
    """
    
    messages = [
        {"role": "system", "content": "You are an expert academic paper reviewer."},
        {"role": "user", "content": prompt.format(text=text)}
    ]

    # ...existing code...
