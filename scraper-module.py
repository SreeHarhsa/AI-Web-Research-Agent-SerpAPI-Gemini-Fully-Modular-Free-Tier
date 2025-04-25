import time
import requests
import logging
import re
from typing import Tuple, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """
    Web scraping tool that extracts readable content from URLs.
    """
    def __init__(self, timeout: int = 10):
        """
        Initialize the web scraper.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def scrape(self, url: str, max_retries: int = 2) -> Tuple[bool, str]:
        """
        Scrape content from a URL.
        
        Args:
            url: The URL to scrape
            max_retries: Number of retry attempts on failure
            
        Returns:
            Tuple of (success, content or error message)
        """
        if not self._is_valid_url(url):
            return False, "Invalid URL format."
        
        # Skip URLs that are likely to be problematic
        if self._should_skip_url(url):
            return False, f"Skipping URL that may contain non-HTML content: {url}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Scraping attempt {attempt+1}/{max_retries} for URL: {url}")
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                
                # Check content type to make sure it's HTML
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    return False, f"Not an HTML page. Content-Type: {content_type}"
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract the main content
                content = self._extract_main_content(soup)
                
                if not content or len(content) < 100:
                    return False, "Content too short or empty."
                
                # Cap content length to prevent overloading the summary API
                if len(content) > 15000:
                    content = content[:15000] + "..."
                    
                return True, content
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out for URL: {url}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                return False, "Request timed out."
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else 'unknown'
                logger.error(f"HTTP error {status_code} when scraping URL: {url}")
                return False, f"HTTP error: {e}"
                
            except requests.exceptions.TooManyRedirects:
                return False, "Too many redirects."
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception when scraping URL: {url} - {str(e)}")
                return False, f"Request error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Unexpected error when scraping URL: {url} - {str(e)}")
                return False, f"Error: {str(e)}"
        
        return False, "Failed after multiple attempts."
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from a webpage, removing navigation, ads, etc.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Extracted text content
        """
        # Remove common non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.decompose()
            
        # Try to find the main content area (different sites use different structures)
        main_content = None
        for selector in ["main", "article", "#content", ".content", "#main", ".main"]:
            content_area = soup.select_one(selector)
            if content_area:
                main_content = content_area
                break
        
        # If no main content area found, use the whole body
        if not main_content:
            main_content = soup.body if soup.body else soup
            
        # Get text content
        text = main_content.get_text(separator="\n")
        
        # Clean up text
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
        text = re.sub(r'\.{2,}', '...', text)  # Normalize ellipses
        text = text.strip()
        
        return text
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
            
    def _should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped (e.g., PDFs, images)."""
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.zip', '.doc', '.docx', '.ppt', '.pptx']
        
        for ext in skip_extensions:
            if url.lower().endswith(ext):
                return True
                
        return False
