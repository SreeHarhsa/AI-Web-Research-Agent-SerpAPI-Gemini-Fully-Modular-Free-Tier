import pytest
import responses
import json
import os
from unittest.mock import patch
from search_tool import SerpApiSearchTool

# Mock data
MOCK_SERPAPI_RESPONSE = {
    "organic_results": [
        {
            "position": 1,
            "title": "Test Result 1",
            "link": "https://example.com/1",
            "snippet": "This is the first test result."
        },
        {
            "position": 2,
            "title": "Test Result 2",
            "link": "https://example.com/2",
            "snippet": "This is the second test result."
        }
    ]
}

class TestSerpApiSearchTool:
    
    def setup_method(self):
        # Set up a mock API key for testing
        os.environ["SERPAPI_KEY"] = "mock_api_key"
        self.search_tool = SerpApiSearchTool()
    
    @responses.activate
    def test_search_success(self):
        # Mock the SerpAPI response
        responses.add(
            responses.GET,
            "https://serpapi.com/search",
            json=MOCK_SERPAPI_RESPONSE,
            status=200
        )
        
        # Test the search method
        results = self.search_tool.search("test query")
        
        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[1]["link"] == "https://example.com/2"
    
    @responses.activate
    def test_search_auth_error(self):
        # Mock an authentication error
        responses.add(
            responses.GET,
            "https://serpapi.com/search",
            status=401
        )
        
        # Test the search method with an authentication error
        results = self.search_tool.search("test query")
        
        # Verify empty results on auth error
        assert len(results) == 0
    
    @responses.activate
    def test_search_rate_limit_with_retry(self):
        # Mock a rate limit error followed by a successful response
        responses.add(
            responses.GET,
            "https://serpapi.com/search",
            status=429
        )
        responses.add(
            responses.GET,
            "https://serpapi.com/search",
            json=MOCK_SERPAPI_RESPONSE,
            status=200
        )
        
        # Patch the time.sleep function to avoid waiting during tests
        with patch('time.sleep'):
            results = self.search_tool.search("test query", max_retries=2)
            
            # Verify results after retry
            assert len(results) == 2
    
    @responses.activate
    def test_search_no_results(self):
        # Mock an empty response
        responses.add(
            responses.GET,
            "https://serpapi.com/search",
            json={"organic_results": []},
            status=200
        )
        
        # Test the search with no results
        results = self.search_tool.search("test query")
        
        # Verify empty results
        assert len(results) == 0
    
    def test_get_top_results(self):
        # Mock the search method to return our test data
        with patch.object(SerpApiSearchTool, 'search', return_value=MOCK_SERPAPI_RESPONSE["organic_results"]):
            results = self.search_tool.get_top_results("test query", 2)
            
            # Verify processed results
            assert len(results) == 2
            assert results[0]["title"] == "Test Result 1"
            assert results[1]["snippet"] == "This is the second test result."
    
    def test_api_key_required(self):
        # Test that API key is required
        os.environ.pop("SERPAPI_KEY", None)
        with pytest.raises(ValueError):
            SerpApiSearchTool()
