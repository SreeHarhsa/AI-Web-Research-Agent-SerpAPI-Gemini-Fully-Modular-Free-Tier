import os
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import our custom modules
from search_tool import SerpApiSearchTool
from scraper import WebScraper
from summarizer import GeminiSummarizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebResearchAgent:
    """
    Web Research Agent that orchestrates the search, scraping, and summarization pipeline.
    """
    def __init__(
        self, 
        gemini_key: Optional[str] = None, 
        serpapi_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize the Web Research Agent with necessary API keys.
        
        Args:
            gemini_key: API key for Gemini, defaults to GEMINI_API_KEY environment variable
            serpapi_key: API key for SerpAPI, defaults to SERPAPI_KEY environment variable
            model_name: Name of the Gemini model to use
        """
        # Use provided keys or fall back to environment variables
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        
        # Validate API keys
        if not self.gemini_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it directly.")
        if not self.serpapi_key:
            raise ValueError("SerpAPI key is required. Set SERPAPI_KEY environment variable or pass it directly.")
        
        # Initialize components
        self.search_tool = SerpApiSearchTool(api_key=self.serpapi_key)
        self.scraper = WebScraper()
        self.summarizer = GeminiSummarizer(api_key=self.gemini_key, model_name=model_name)
        
        logger.info("Web Research Agent initialized successfully")
    
    def research(
        self, 
        query: str, 
        num_results: int = 5, 
        brief_summaries: bool = False
    ) -> Dict:
        """
        Execute the full research pipeline for a given query.
        
        Args:
            query: The search query to research
            num_results: Number of top results to process
            brief_summaries: Whether to generate brief summaries
            
        Returns:
            Dictionary containing research results and comprehensive summary
        """
        logger.info(f"Starting research for query: '{query}'")
        
        # Step 1: Search the web
        search_results = self.search_tool.get_top_results(query, num_results)
        if not search_results:
            return {
                "success": False,
                "error": "No search results found",
                "results": [],
                "comprehensive_summary": None
            }
        
        logger.info(f"Found {len(search_results)} search results")
        
        # Step 2: Process each search result
        successful_summaries = []
        
        for i, result in enumerate(search_results):
            result_data = {
                "title": result['title'],
                "link": result['link'],
                "snippet": result['snippet'],
                "success": False
            }
            
            # Step 2a: Scrape the webpage
            scrape_success, content = self.scraper.scrape(result['link'])
            
            if not scrape_success:
                result_data["error"] = content  # content contains error message
                logger.warning(f"Failed to scrape content from {result['link']}: {content}")
            else:
                # Step 2b: Summarize content
                summary_success, summary = self.summarizer.summarize(content, query, brief_summaries)
                
                if not summary_success:
                    result_data["error"] = summary  # summary contains error message
                    logger.warning(f"Failed to summarize content from {result['link']}: {summary}")
                else:
                    result_data["success"] = True
                    result_data["summary"] = summary
                    result_data["content"] = content[:1000] + "..." if len(content) > 1000 else content
                    
                    # Add to successful summaries for comprehensive summary later
                    successful_summaries.append({
                        'title': result['title'],
                        'link': result['link'],
                        'summary': summary
                    })
            
            search_results[i] = result_data
        
        # Step 3: Create comprehensive summary if we have successful summaries
        comprehensive_summary = None
        comprehensive_success = False
        
        if successful_summaries:
            logger.info(f"Creating comprehensive summary from {len(successful_summaries)} successful summaries")
            comprehensive_success, comprehensive_summary = self.summarizer.create_comprehensive_summary(
                successful_summaries, query
            )
        
        # Step 4: Compile final results
        return {
            "success": len(successful_summaries) > 0,
            "query": query,
            "results": search_results,
            "comprehensive_summary": comprehensive_summary if comprehensive_success else None
        }
