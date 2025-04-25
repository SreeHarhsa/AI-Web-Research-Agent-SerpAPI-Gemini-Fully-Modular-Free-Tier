import os
import logging
import google.generativeai as genai
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiSummarizer:
    """
    Content summarization tool using Google's Gemini API.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini summarizer.
        
        Args:
            api_key: Gemini API key, defaults to None (will try to get from environment)
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("No Gemini API key provided!")
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        self.model_name = model_name
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
    def summarize(self, content: str, query: str, is_brief: bool = False) -> Tuple[bool, str]:
        """
        Summarize content using Gemini API.
        
        Args:
            content: The content to summarize
            query: The original search query
            is_brief: Whether to generate a brief summary
            
        Returns:
            Tuple of (success, summary or error message)
        """
        try:
            logger.info(f"Summarizing content for query: '{query}' (brief: {is_brief})")
            
            # Create the model
            model = genai.GenerativeModel(self.model_name)
            
            # Determine length instruction based on brief flag
            length_instruction = "brief and concise (100-150 words)" if is_brief else "comprehensive (300-500 words)"
            
            # Create the prompt
            prompt = self._create_summary_prompt(content, query, length_instruction)
            
            # Generate summary
            response = model.generate_content(prompt)
            
            if not hasattr(response, 'text'):
                logger.error("Unexpected response format from Gemini API")
                return False, "Error: Unexpected response format from API"
                
            summary = response.text
            
            if not summary or len(summary) < 50:
                logger.warning("Summary too short or empty")
                return False, "Generated summary was too short or empty"
                
            return True, summary
            
        except Exception as e:
            logger.error(f"Error summarizing with Gemini API: {str(e)}")
            return False, f"Failed to summarize: {str(e)}"
    
    def create_comprehensive_summary(self, summaries: List[Dict], query: str) -> Tuple[bool, str]:
        """
        Create a comprehensive summary from multiple individual summaries.
        
        Args:
            summaries: List of dictionaries containing individual summaries
            query: The original search query
            
        Returns:
            Tuple of (success, comprehensive summary or error message)
        """
        try:
            logger.info(f"Creating comprehensive summary for query: '{query}'")
            
            # Create the model
            model = genai.GenerativeModel(self.model_name)
            
            # Combine content from all summaries
            combined_content = ""
            for i, summary_data in enumerate(summaries, 1):
                combined_content += f"\n\nSOURCE {i}: {summary_data['title']}\n"
                combined_content += f"{summary_data['summary']}\n"
                combined_content += f"URL: {summary_data['link']}\n"
            
            # Create the prompt
            prompt = self._create_comprehensive_prompt(combined_content, query)
            
            # Generate comprehensive summary
            response = model.generate_content(prompt)
            
            if not hasattr(response, 'text'):
                logger.error("Unexpected response format from Gemini API")
                return False, "Error: Unexpected response format from API"
                
            comprehensive_summary = response.text
            
            return True, comprehensive_summary
            
        except Exception as e:
            logger.error(f"Error creating comprehensive summary: {str(e)}")
            return False, f"Failed to create comprehensive summary: {str(e)}"
    
    def _create_summary_prompt(self, content: str, query: str, length_instruction: str) -> str:
        """Create a prompt for individual content summarization."""
        return f"""
        Please provide a {length_instruction} summary of the following content in relation to this search query: "{query}"
        
        Focus on:
        1. Key facts and information directly relevant to the query
        2. Main insights, findings, or advancements mentioned
        3. Any important conclusions or future implications
        
        Content:
        {content}
        
        Format your summary with clear sections and bullet points where appropriate to enhance readability.
        Use markdown formatting to improve structure and highlight important information.
        """
    
    def _create_comprehensive_prompt(self, combined_content: str, query: str) -> str:
        """Create a prompt for comprehensive summary synthesis."""
        return f"""
        Please synthesize a comprehensive research conclusion from the following summaries related to this search query: "{query}"
        
        Your task is to:
        1. Identify the main findings and key points across all sources
        2. Note any consensus or disagreements between sources
        3. Highlight the most important and relevant information
        4. Organize the information logically with clear section headings
        5. Include a brief conclusion with the most important takeaways
        
        Here are the summaries from various sources:
        {combined_content}
        
        Please create a well-structured, comprehensive research conclusion that synthesizes all this information.
        Include a "Key Findings" section at the beginning and a "Conclusion" at the end.
        Use markdown formatting to improve structure and readability.
        """
