import os
import time
import requests
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SerpApiSearchTool:
    """
    A wrapper for querying SerpAPI and extracting result URLs.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SerpAPI search tool.
        
        Args:
            api_key: SerpAPI key, defaults to None (will try to get from environment)
        """
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            logger.error("No SerpAPI key provided!")
            raise ValueError("SerpAPI key is required. Set SERPAPI_KEY environment variable or pass it directly.")
        
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query: str, num_results: int = 5, max_retries: int = 3) -> List[Dict]:
        """
        Search using SerpAPI and extract organic search results.
        
        Args:
            query: The search query
            num_results: Number of results to return (max 10)
            max_retries: Number of retry attempts on failure
            
        Returns:
            List of dictionaries containing search results
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": min(num_results, 10)  # Cap at 10 to respect SerpAPI free tier
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"SerpAPI search attempt {attempt+1}/{max_retries} for query: '{query}'")
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 401:
                    logger.error("SerpAPI authentication error (401). Check your API key.")
                    return []
                
                response.raise_for_status()
                results = response.json()
                
                # Extract organic results
                if "organic_results" in results:
                    logger.info(f"Found {len(results['organic_results'])} search results")
                    return results["organic_results"]
                else:
                    logger.warning("No organic results found in SerpAPI response.")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error from SerpAPI: {e}")
                    return []
                    
            except Exception as e:
                logger.error(f"Error searching with SerpAPI: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return []
        
        # If we get here, all retries failed
        return []
    
    def get_top_results(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Get top search results with title, snippet, and link.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with processed search results
        """
        raw_results = self.search(query, num_results)
        
        processed_results = []
        for result in raw_results:
            processed_results.append({
                'title': result.get('title', 'No Title'),
                'snippet': result.get('snippet', 'No snippet available'),
                'link': result.get('link', '#')
            })
            
        return processed_results
