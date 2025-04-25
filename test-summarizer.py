import pytest
import os
from unittest.mock import patch, MagicMock
from summarizer import GeminiSummarizer

# Mock response class to simulate Gemini API responses
class MockResponse:
    def __init__(self, text):
        self.text = text

class TestGeminiSummarizer:
    
    def setup_method(self):
        # Set up a mock API key for testing
        os.environ["GEMINI_API_KEY"] = "mock_api_key"
        self.summarizer = GeminiSummarizer()
    
    def test_api_key_required(self):
        # Test that API key is required
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError):
            GeminiSummarizer()
    
    @patch('google.generativeai.GenerativeModel')
    def test_summarize_success(self, mock_model):
        # Mock the Gemini model response
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MockResponse("This is a test summary.")
        mock_model.return_value = mock_instance
        
        # Test successful summarization
        success, summary = self.summarizer.summarize(
            "This is test content to summarize.",
            "test query",
            is_brief=False
        )
        
        # Verify successful summarization
        assert success is True
        assert summary == "This is a test summary."
        
        # Verify prompt includes the right instructions
        args, _ = mock_instance.generate_content.call_args
        prompt = args[0]
        assert "comprehensive (300-500 words)" in prompt
        assert "test query" in prompt
        assert "This is test content to summarize" in prompt
    
    @patch('google.generativeai.GenerativeModel')
    def test_summarize_brief(self, mock_model):
        # Mock the Gemini model response
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MockResponse("Brief test summary.")
        mock_model.return_value = mock_instance
        
        # Test brief summarization
        success, summary = self.summarizer.summarize(
            "This is test content to summarize.",
            "test query",
            is_brief=True
        )
        
        # Verify brief summarization
        assert success is True
        assert summary == "Brief test summary."
        
        # Verify prompt includes brief instruction
        args, _ = mock_instance.generate_content.call_args
        prompt = args[0]
        assert "brief and concise (100-150 words)" in prompt
    
    @patch('google.generativeai.GenerativeModel')
    def test_summarize_api_error(self, mock_model):
        # Mock an API error
        mock_instance = MagicMock()
        mock_instance.generate_content.side_effect = Exception("API error")
        mock_model.return_value = mock_instance
        
        # Test error handling
        success, error_msg = self.summarizer.summarize(
            "This is test content.",
            "test query"
        )
        
        # Verify error handling
        assert success is False
        assert "Failed to summarize" in error_msg
    
    @patch('google.generativeai.GenerativeModel')
    def test_create_comprehensive_summary(self, mock_model):
        # Mock the Gemini model response
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = MockResponse("Comprehensive research conclusion.")
        mock_model.return_value = mock_instance
        
        # Test data
        summaries = [
            {
                "title": "Source 1",
                "link": "https://example.com/1",
                "summary": "Summary of source 1"
            },
            {
                "title": "Source 2",
                "link": "https://example.com/2",
                "summary": "Summary of source 2"
            }
        ]
        
        # Test comprehensive summary creation
        success, summary = self.summarizer.create_comprehensive_summary(summaries, "test query")
        
        # Verify comprehensive summary
        assert success is True
        assert summary == "Comprehensive research conclusion."
        
        # Verify prompt structure
        args, _ = mock_instance.generate_content.call_args
        prompt = args[0]
        assert "synthesize a comprehensive research conclusion" in prompt.lower()
        assert "Source 1" in prompt
        assert "Source 2" in prompt
        assert "test query" in prompt
    
    @patch('google.generativeai.GenerativeModel')
    def test_unexpected_response_format(self, mock_model):
        # Mock an unexpected response format (no text attribute)
        mock_instance = MagicMock()
        mock_instance.generate_content.return_value = object()  # Object without .text attribute
        mock_model.return_value = mock_instance
        
        # Test handling of unexpected response format
        success, error_msg = self.summarizer.summarize(
            "This is test content.",
            "test query"
        )
        
        # Verify error handling for unexpected format
        assert success is False
        assert "Unexpected response format" in error_msg
