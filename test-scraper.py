import pytest
import responses
from bs4 import BeautifulSoup
from unittest.mock import patch
from scraper import WebScraper

# Sample HTML content for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <script>console.log("This should be removed");</script>
    <style>body { color: red; }</style>
</head>
<body>
    <header>Header content to remove</header>
    <nav>Navigation to remove</nav>
    <main>
        <h1>Main Heading</h1>
        <p>This is the first paragraph with important content.</p>
        <p>This is the second paragraph with more important content.</p>
    </main>
    <footer>Footer content to remove</footer>
</body>
</html>
"""

# Non-HTML content
PDF_CONTENT = "%PDF-1.5\n% Test PDF content"

class TestWebScraper:
    
    def setup_method(self):
        self.scraper = WebScraper(timeout=5)
    
    @responses.activate
    def test_scrape_success(self):
        # Mock a successful response
        responses.add(
            responses.GET,
            "https://example.com",
            body=SAMPLE_HTML,
            status=200,
            headers={'Content-Type': 'text/html'}
        )
        
        # Test the scrape method
        success, content = self.scraper.scrape("https://example.com")
        
        # Verify success and content
        assert success is True
        assert "Main Heading" in content
        assert "first paragraph" in content
        assert "second paragraph" in content
        assert "console.log" not in content
        assert "Header content to remove" not in content
    
    @responses.activate
    def test_scrape_non_html_content(self):
        # Mock a PDF response
        responses.add(
            responses.GET,
            "https://example.com/doc.pdf",
            body=PDF_CONTENT,
            status=200,
            headers={'Content-Type': 'application/pdf'}
        )
        
        # Test scraping non-HTML content
        success, content = self.scraper.scrape("https://example.com/doc.pdf")
        
        # Verify failure for non-HTML content
        assert success is False
        assert "Not an HTML page" in content
    
    @responses.activate
    def test_scrape_timeout(self):
        # Mock a timeout
        responses.add(
            responses.GET,
            "https://example.com/timeout",
            body=responses.CallbackResponse(
                callback=lambda request: (200, {}, "Timeout error"),
                content_type="text/html"
            )
        )
        
        # Patch requests.get to raise a Timeout error
        with patch('requests.get', side_effect=requests.exceptions.Timeout):
            success, content = self.scraper.scrape("https://example.com/timeout", max_retries=1)
            
            # Verify timeout handling
            assert success is False
            assert "timed out" in content.lower()
    
    @responses.activate
    def test_scrape_http_error(self):
        # Mock a 404 error
        responses.add(
            responses.GET,
            "https://example.com/not-found",
            status=404
        )
        
        # Test HTTP error handling
        success, content = self.scraper.scrape("https://example.com/not-found")
        
        # Verify HTTP error handling
        assert success is False
        assert "HTTP error" in content
    
    def test_invalid_url(self):
        # Test with invalid URL
        success, content = self.scraper.scrape("not-a-valid-url")
        
        # Verify invalid URL handling
        assert success is False
        assert "Invalid URL" in content
    
    def test_extract_main_content(self):
        # Create a BeautifulSoup object
        soup = BeautifulSoup(SAMPLE_HTML, 'html.parser')
        
        # Test content extraction
        content = self.scraper._extract_main_content(soup)
        
        # Verify content extraction
        assert "Main Heading" in content
        assert "first paragraph" in content
        assert "second paragraph" in content
        assert "Header content" not in content
        assert "Footer content" not in content
    
    def test_should_skip_url(self):
        # Test URL skipping logic
        assert self.scraper._should_skip_url("https://example.com/doc.pdf") is True
        assert self.scraper._should_skip_url("https://example.com/image.jpg") is True
        assert self.scraper._should_skip_url("https://example.com/page.html") is False
        assert self.scraper._should_skip_url("https://example.com/article") is False
