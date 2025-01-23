from agents import MathChecker, MethodologyChecker, LogicChecker, ReferenceChecker, WritingChecker
import asyncio
from typing import List, Dict
import logging
import json

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, client):
        self.client = client
        self.agents = {
            'math': MathChecker(client),
            'methodology': MethodologyChecker(client),
            'logic': LogicChecker(client),
            'references': ReferenceChecker(client),
            'writing': WritingChecker(client)
        }

    async def analyze_paper(self, text: str, metadata: Dict) -> Dict:
        """Coordinate the analysis among different agents"""
        tasks = []
        for agent_name, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.analyze(text)))

        try:
            results = await asyncio.gather(*tasks)
            
            # Calculate total errors
            total_errors = sum(result['error_count'] for result in results)
            
            # Add error statistics to summary
            # summary = await self._generate_summary(results, text)
            # summary['error_statistics'] = {
            #     'total_errors': total_errors,
            #     'error_counts': {
            #         result['type']: result['error_count'] 
            #         for result in results
            #     }
            # }
            
            return {
                "metadata": metadata,
                "analysis": results,
                # "summary": summary
            }
        except Exception as e:
            logger.error(f"Orchestration error: {str(e)}")
            raise

    async def _generate_summary(self, results: List[Dict], text: str) -> Dict:
        """Generate a consolidated summary of all findings"""
        summary = {
            "critical_issues": [],
            "recommendations": [],
            "solutions": [],
            "overall_assessment": "",
            "authors": [],
            "paper_link": "",
            "homepage_link": "",
            "title": ""
        }
        # Extract authors and links using a separate OpenAI call
        link_response = self.client.chat.completions.create(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": """Extract the following from the text:
                    1. Author names (as comma-separated list)
                    2. Paper URL/DOI link
                    3. Author/Institution homepage link
                    4. Title of the paper
                    
                    Format your response as JSON:
                    {
                        "authors": "author1, author2, ...",
                        "paper_link": "url or doi",
                        "homepage_link": "url"
                        "title": "title of the paper"
                    }
                    
                    If any item is not found, use empty string."""
                },
                {"role": "user", "content": text}
            ]
        )
        
        try:
            link_data = json.loads(link_response.choices[0].message.content)
            summary["authors"] = [
                author.strip() 
                for author in link_data.get("authors", "").split(",")
                if author.strip()
            ]
            summary["paper_link"] = link_data.get("paper_link", "")
            summary["homepage_link"] = link_data.get("homepage_link", "")
            summary["title"] = link_data.get("title", "")
        except json.JSONDecodeError:
            logger.error("Failed to parse link extraction response", {"response": link_response.choices[0].message.content})
            summary["authors"] = []
            summary["paper_link"] = ""
            summary["homepage_link"] = ""
            summary["title"] = ""
        
        # Aggregate findings and categorize solutions
        for result in results:
            if isinstance(result.get("findings"), list):
                for finding in result["findings"]:
                    if finding.get("impact", "").lower().startswith("critical"):
                        summary["critical_issues"].append({
                            "type": result["type"],
                            "issue": finding["issue"],
                            "solution": finding["solution"]
                        })
                    
                    # Add all solutions to the solutions list
                    summary["solutions"].append({
                        "type": result["type"],
                        "issue": finding["issue"],
                        "impact": finding["impact"],
                        "solution": finding["solution"]
                    })
                    
                    if "recommend" in finding.get("solution", "").lower():
                        summary["recommendations"].append({
                            "type": result["type"],
                            "recommendation": finding["solution"]
                        })

        # Generate a solution-focused assessment
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user",
                    "content": """Create a solution-focused summary of the analysis results. Include:
                    1. Most critical issues and their solutions
                    2. Key recommendations for improvement
                    3. Prioritized action items"""
                },
                {"role": "user", "content": str(summary["solutions"])}
            ]
        )
        summary["overall_assessment"] = response.choices[0].message.content
        
        return summary
