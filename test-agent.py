import pytest
import os
from unittest.mock import patch, MagicMock
from agent import WebResearchAgent

class TestWebResearchAgent:
    
    def setup_method(self):
        # Set up mock API keys for testing
        os.environ["GEMINI_API_KEY"] = "mock_gemini_key"
        os.environ["SERPAPI_KEY"] = "mock_serpapi_key"
    
    def test_initialization(self):
        # Test successful initialization
        agent = WebResearchAgent()
        assert agent.gemini_key == "mock_gemini_key"
        assert agent.serpapi_key == "mock_serpapi_key"
    
    def test_missing_api_keys(self):
        # Test handling of missing API keys
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("SERPAPI_KEY", None)
        
        # Should raise ValueError for missing Gemini key
        with pytest.raises(ValueError) as excinfo:
            WebResearchAgent()
        assert "Gemini API key is required" in str(excinfo.value)
        
        # Should raise ValueError for missing SerpAPI key but present Gemini key
        os.environ["GEMINI_API_KEY"] = "mock_gemini_key"
        with pytest.raises(ValueError) as excinfo:
            WebResearchAgent()
        assert "SerpAPI key is required" in str(excinfo.value)
    
    @patch('agent.SerpApiSearchTool')
    @patch('agent.WebScraper')
    @patch('agent.GeminiSummarizer')
    def test_research_flow(self, mock_summarizer, mock_scraper, mock_search_tool):
        # Set up mock return values
        
        # Mock search results
        mock_search_instance = MagicMock()
        mock_search_instance.get_top_results.return_value = [
            {
                'title': 'Test Result',
                'link': 'https://example.com',
                'snippet': 'Test snippet'
            }
        ]
        mock_search_tool.return_value = mock_search_instance
        
        # Mock scraper results
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.scrape.return_value = (True, "Test content")
        mock_scraper.return_value = mock_scraper_instance
        
        # Mock summarizer results
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.summarize.return_value = (True, "Test summary")
        mock_summarizer_instance.create_comprehensive_summary.return_value = (True, "Comprehensive summary")
        mock_summarizer.return_value = mock_summarizer_instance
        
        # Create agent and run research
        agent = WebResearchAgent()
        result = agent.research("test query")
        
        # Verify the research flow
        assert result["success"] is True
        assert result["query"] == "test query"
        assert len(result["results"]) == 1
        assert result["results"][0]["success"] is True
        assert result["results"][0]["summary"] == "Test summary"
        assert result["comprehensive_summary"] == "Comprehensive summary"
        
        # Verify method calls
        mock_search_instance.get_top_results.assert_called_once_with("test query", 5)
        mock_scraper_instance.scrape.assert_called_once_with("https://example.com")
        mock_summarizer_instance.summarize.assert_called_once()
        mock_summarizer_instance.create_comprehensive_summary.assert_called_once()
    
    @patch('agent.SerpApiSearchTool')
    def test_research_no_results(self, mock_search_tool):
        # Mock empty search results
        mock_search_instance = MagicMock()
        mock_search_instance.get_top_results.return_value = []
        mock_search_tool.return_value = mock_search_instance
        
        # Create agent and run research
        agent = WebResearchAgent()
        result = agent.research("test query")
        
        # Verify empty results handling
        assert result["success"] is False
        assert result["error"] == "No search results found"
        assert result["results"] == []
        assert result["comprehensive_summary"] is None
    
    @patch('agent.SerpApiSearchTool')
    @patch('agent.WebScraper')
    @patch('agent.GeminiSummarizer')
    def test_research_scraping_failure(self, mock_summarizer, mock_scraper, mock_search_tool):
        # Mock search results
        mock_search_instance = MagicMock()
        mock_search_instance.get_top_results.return_value = [
            {
                'title': 'Test Result',
                'link': 'https://example.com',
                'snippet': 'Test snippet'
            }
        ]
        mock_search_tool.return_value = mock_search_instance
        
        # Mock scraper failure
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.scrape.return_value = (False, "Failed to scrape content")
        mock_scraper.return_value = mock_scraper_instance
        
        # Mock summarizer
        mock_summarizer_instance = MagicMock()
        mock_summarizer.return_value = mock_summarizer_instance
        
        # Create agent and run research
        agent = WebResearchAgent()
        result = agent.research("test query")
        
        # Verify scraping failure handling
        assert result["success"] is False
        assert len(result["results"]) == 1
        assert result["results"][0]["success"] is False
        assert result["results"][0]["error"] == "Failed to scrape content"
        assert result["comprehensive_summary"] is None
        
        # Verify summarize was not called
        mock_summarizer_instance.summarize.assert_not_called()
        mock_summarizer_instance.create_comprehensive_summary.assert_not_called()
