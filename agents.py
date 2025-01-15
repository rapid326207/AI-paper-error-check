from abc import ABC, abstractmethod
from openai import OpenAI
import json
import re

def latex_to_markdown(text):
    """Convert LaTeX-style formatting to Markdown
    
    Args:
        text (str): Text containing LaTeX formatting
        
    Returns:
        str: Text converted to Markdown format
    
    Example:
        >>> latex_to_markdown("[ 7,000, \\text{ng/kg/day} \\times 60, \\text{kg} = 420,000, \\text{ng/day} ]")
        "**7,000 ng/kg/day × 60 kg = 420,000 ng/day**"
    """
    converted_text = text
    
    # Replace LaTeX math mode brackets with bold markers
    converted_text = converted_text.replace('[', '**')
    converted_text = converted_text.replace(']', '**')
    
    # Replace LaTeX \text{} with plain text
    while '\\text{' in converted_text:
        start = converted_text.find('\\text{')
        end = converted_text.find('}', start)
        if end != -1:
            text_content = converted_text[start+6:end]
            converted_text = converted_text[:start] + text_content + converted_text[end+1:]
    
    # Replace LaTeX symbols with their Unicode equivalents
    latex_symbols = {
        '\\times': '×',
        '\\div': '÷',
        '\\pm': '±',
        '\\leq': '≤',
        '\\geq': '≥',
        '\\neq': '≠',
        '\\approx': '≈',
        '\\alpha': 'α',
        '\\beta': 'β',
        '\\gamma': 'γ',
        '\\delta': 'δ',
        '\\mu': 'μ',
        '\\sigma': 'σ',
        '\\degree': '°'
    }
    
    for latex, symbol in latex_symbols.items():
        converted_text = converted_text.replace(latex, symbol)
    
    return converted_text



class BaseAgent(ABC):
    def __init__(self, client: OpenAI):
        self.client = client

    @abstractmethod
    async def analyze(self, text: str) -> dict:
        pass

    def count_errors(self, text: str) -> int:
        """Count errors in the analysis text based on error/issue markers"""
        # Look for numbered lists, bullet points, or "Error:" markers
        error_patterns = [
            r'^\d+\.',  # Numbered lists
            r'[-•]\s',  # Bullet points
            r'Error:',  # Explicit error markers
            r'Issue:',  # Issue markers
            r'Problem:'  # Problem markers
        ]
        
        lines = text.split('\n')
        error_count = 0
        
        for line in lines:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in error_patterns):
                error_count += 1
                
        return error_count

class MathChecker(BaseAgent):
    async def analyze(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a mathematical review expert. For each error found, provide both the error and its solution in the following format:

                    1. Error: [Description]
                       Solution: [How to fix this error]
                    2. Error: [Description]
                       Solution: [How to fix this error]
                    etc.
                    
                    Analyze the text for:
                    - Equation correctness
                    - Arithmetic operations
                    - Unit conversions
                    - Statistical calculations
                    - Numerical consistency"""
                },
                {"role": "user", "content": text}
            ]
        )
        findings = latex_to_markdown(response.choices[0].message.content)
        error_count = self.count_errors(findings)
        return {
            "type": "mathematical",
            "error_count": error_count,
            "findings": findings
        }

class MethodologyChecker(BaseAgent):
    async def analyze(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a research methodology expert. For each issue found, provide both the error and its solution in the following format:

                    1. Error: [Description]
                       Solution: [How to fix this error]
                    2. Error: [Description]
                       Solution: [How to fix this error]
                    etc.
                    
                    Analyze the text for:
                    - Study design validity
                    - Sample size adequacy
                    - Statistical test appropriateness
                    - Control group usage
                    - Reproducibility concerns"""
                },
                {"role": "user", "content": text}
            ]
        )
        findings = latex_to_markdown(response.choices[0].message.content)
        error_count = self.count_errors(findings)
        return {
            "type": "methodology",
            "error_count": error_count,
            "findings": findings
        }

class LogicChecker(BaseAgent):
    async def analyze(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a logic and reasoning expert. For each error found, provide both the error and its solution in the following format:

                    1. Error: [Description]
                       Solution: [How to fix this error]
                    2. Error: [Description]
                       Solution: [How to fix this error]
                    etc.
                    
                    Analyze the text for:
                    - Argument coherence
                    - Contradictory statements
                    - Unsupported conclusions
                    - Hypothesis alignment
                    - Causal relationship validity"""
                },
                {"role": "user", "content": text}
            ]
        )
        findings = latex_to_markdown(response.choices[0].message.content)
        error_count = self.count_errors(findings)
        return {
            "type": "logic",
            "error_count": error_count,
            "findings": findings
        }

class ReferenceChecker(BaseAgent):
    async def analyze(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a reference and citation expert. For each issue found, provide both the error and its solution in the following format:

                    1. Error: [Description]
                       Solution: [How to fix this error]
                    2. Error: [Description]
                       Solution: [How to fix this error]
                    etc.
                    
                    Analyze the text for:
                    - Citation format correctness
                    - Reference list completeness
                    - Citation-content alignment
                    - Source reliability
                    - Quote accuracy"""
                },
                {"role": "user", "content": text}
            ]
        )
        findings = latex_to_markdown(response.choices[0].message.content)
        error_count = self.count_errors(findings)
        return {
            "type": "references",
            "error_count": error_count,
            "findings": findings
        }

class WritingChecker(BaseAgent):
    async def analyze(self, text: str) -> dict:
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """You are a scientific writing expert. For each issue found, provide both the error and its solution in the following format:

                    1. Error: [Description]
                       Solution: [How to fix this error]
                    2. Error: [Description]
                       Solution: [How to fix this error]
                    etc.
                    
                    Analyze the text for:
                    - Grammar and spelling
                    - Clarity and conciseness
                    - Academic writing style
                    - Structure and organization
                    - Technical terminology usage
                    - Language consistency"""
                },
                {"role": "user", "content": text}
            ]
        )
        findings = latex_to_markdown(response.choices[0].message.content)
        error_count = self.count_errors(findings)
        return {
            "type": "writing",
            "error_count": error_count,
            "findings": findings
        }
