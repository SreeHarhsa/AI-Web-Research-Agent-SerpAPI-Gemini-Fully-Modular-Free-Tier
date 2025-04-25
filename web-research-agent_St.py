import streamlit as st
import requests
import json
import time
import re
import docx
import io
import zipfile
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import google.generativeai as genai
from typing import List, Dict, Optional, Tuple, Union
import traceback
import logging
import os
import concurrent.futures
import hashlib
import pickle
from datetime import datetime, timedelta

# Configure advanced logging with rotation
import logging.handlers
os.makedirs("logs", exist_ok=True)
log_handler = logging.handlers.RotatingFileHandler(
    "logs/web_research_agent.log", 
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3
)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Set the page configuration and title
st.set_page_config(
    page_title="Web Research Agent Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create cache directory
os.makedirs("cache", exist_ok=True)

# Cache implementation for search results and web content
def get_cache_path(query_or_url: str) -> str:
    """Generate a cache file path based on hash of the query or URL"""
    hash_obj = hashlib.md5(query_or_url.encode()).hexdigest()
    return f"cache/{hash_obj}.pkl"

def save_to_cache(query_or_url: str, data: any, expiry_hours: int = 24) -> None:
    """Save data to cache with expiration time"""
    cache_path = get_cache_path(query_or_url)
    cache_data = {
        "data": data,
        "expires": datetime.now() + timedelta(hours=expiry_hours)
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    logger.info(f"Saved to cache: {query_or_url[:50]}...")

def load_from_cache(query_or_url: str) -> Optional[any]:
    """Load data from cache if not expired"""
    cache_path = get_cache_path(query_or_url)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            
            if datetime.now() < cache_data["expires"]:
                logger.info(f"Cache hit: {query_or_url[:50]}...")
                return cache_data["data"]
            else:
                logger.info(f"Cache expired: {query_or_url[:50]}...")
                os.remove(cache_path)  # Clean up expired cache
        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
    return None

# Function to read Gemini API key from various sources
def read_gemini_key(file_path: str) -> str:
    """Read Gemini API key from a document file"""
    try:
        logger.info(f"Reading Gemini API key from: {file_path}")
        if file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            full_text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                full_text = f.read()
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return ""
        
        # Try different patterns for API key
        api_key_patterns = [
            r'AIza[A-Za-z0-9_-]{32}',  # Standard Google API key format
            r'[A-Za-z0-9_-]{39}',       # Gemini API key format
            r'[A-Za-z0-9_-]{30,50}'     # Generic pattern
        ]
        
        for pattern in api_key_patterns:
            api_key_match = re.search(pattern, full_text)
            if api_key_match:
                return api_key_match.group(0)
                
        logger.error("Couldn't find Gemini API key in the document.")
        return ""
    except Exception as e:
        logger.error(f"Error reading Gemini API key: {str(e)}")
        return ""

# Function to read SerpAPI key from various document formats
def read_serpapi_key(file_path: str) -> str:
    """Read SerpAPI key from a document file"""
    try:
        logger.info(f"Reading SerpAPI key from: {file_path}")
        
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                full_text = f.read()
                
        elif file_path.endswith('.odt'):
            with zipfile.ZipFile(file_path, 'r') as z:
                with z.open('content.xml') as f:
                    content = f.read()
                
                # Parse the XML content
                root = ET.fromstring(content)
                
                # Extract all text elements
                namespaces = {
                    'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
                    'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0'
                }
                
                all_text = ""
                
                # Find paragraphs
                for elem in root.findall('.//text:p', namespaces):
                    if elem.text:
                        all_text += elem.text + " "
                        
                    # Get text from spans inside paragraphs
                    for span in elem.findall('.//text:span', namespaces):
                        if span.text:
                            all_text += span.text + " "
                
                # If we didn't find much text, try a more generic approach
                if len(all_text) < 10:
                    logger.warning("Using fallback text extraction method for ODT")
                    xml_str = ET.tostring(root, encoding='unicode')
                    all_text = re.sub(r'<[^>]+>', ' ', xml_str)
                
                full_text = all_text
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return ""
        
        # Look for an API key pattern (typical SerpAPI key format - 32 hex characters)
        api_key_patterns = [
            r'[a-z0-9]{32}',           # Standard SerpAPI key format
            r'[a-z0-9]{20,40}'          # Generic pattern
        ]
        
        for pattern in api_key_patterns:
            api_key_match = re.search(pattern, full_text)
            if api_key_match:
                return api_key_match.group(0)
                
        logger.error("Couldn't find SerpAPI key pattern in the document.")
        return ""
    except Exception as e:
        logger.error(f"Error reading SerpAPI key: {str(e)}")
        return ""

# Function to list available Gemini models
def list_available_models(api_key: str) -> List[str]:
    """List available Gemini models"""
    try:
        genai.configure(api_key=api_key)
        available_models = genai.list_models()
        model_names = [model.name for model in available_models]
        logger.info(f"Available models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        return []

# Test API keys silently
def test_api_keys(gemini_key: str, serpapi_key: str) -> Tuple[bool, bool, str, str]:
    """Test both API keys and return detailed status"""
    gemini_success = False
    serpapi_success = False
    gemini_message = ""
    serpapi_message = ""
    
    # Test Gemini API
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Test")
        if response and hasattr(response, 'text'):
            gemini_success = True
            gemini_message = "Gemini API key is valid"
            logger.info(gemini_message)
        else:
            gemini_message = "Gemini API test failed: Invalid response"
            logger.error(gemini_message)
    except Exception as e:
        gemini_message = f"Gemini API test failed: {str(e)}"
        logger.error(gemini_message)
        
        # Try to list available models for debugging
        list_available_models(gemini_key)
    
    # Test SerpAPI
    try:
        response = requests.get("https://serpapi.com/account", params={"api_key": serpapi_key})
        if response.status_code == 200:
            serpapi_success = True
            serpapi_message = "SerpAPI key is valid"
            logger.info(serpapi_message)
        else:
            serpapi_message = f"SerpAPI key test failed with status code: {response.status_code}"
            logger.error(serpapi_message)
    except Exception as e:
        serpapi_message = f"SerpAPI test failed: {str(e)}"
        logger.error(serpapi_message)
    
    return gemini_success, serpapi_success, gemini_message, serpapi_message

# Search Google using SerpAPI with improved error handling
def search_with_serpapi(
    query: str, 
    api_key: str, 
    num_results: int = 5, 
    search_region: str = "us", 
    max_retries: int = 3
) -> List[Dict]:
    """Search Google using SerpAPI with caching"""
    # Check cache first
    cache_key = f"search:{query}:{num_results}:{search_region}"
    cached_results = load_from_cache(cache_key)
    if cached_results is not None:
        return cached_results
    
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
        "gl": search_region  # Country/region for search results
    }
    
    status_container = st.empty()
    with st.status("Searching Google...", expanded=False) as status:
        for attempt in range(max_retries):
            try:
                logger.info(f"SerpAPI search attempt {attempt+1}/{max_retries}")
                status.update(label=f"Searching Google (attempt {attempt+1}/{max_retries})...")
                
                response = requests.get("https://serpapi.com/search", params=params, timeout=15)
                
                if response.status_code == 401:
                    status.update(label="Search failed: Authentication error with SerpAPI", state="error")
                    time.sleep(1)
                    return []
                    
                response.raise_for_status()
                results = response.json()
                
                # Extract organic results
                if "organic_results" in results:
                    status.update(label="Search complete!", state="complete")
                    
                    # Cache the results
                    organic_results = results["organic_results"]
                    save_to_cache(cache_key, organic_results)
                    
                    return organic_results
                else:
                    status.update(label="No results found in SerpAPI response", state="error")
                    logger.warning("No organic results found in SerpAPI response.")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    status.update(label=f"Rate limited. Retrying in {wait_time}s ({attempt+1}/{max_retries})...", state="running")
                    time.sleep(wait_time)  # Exponential backoff
                else:
                    status.update(label=f"Search error: {e}", state="error")
                    logger.error(f"HTTP error from SerpAPI: {e}")
                    return []
                    
            except Exception as e:
                status.update(label=f"Search error: {str(e)}", state="error")
                logger.error(f"Error searching with SerpAPI: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return []
    
    return []

# Improved web scraping function
def scrape_web_page(url: str, max_retries: int = 2) -> Tuple[bool, str]:
    """Scrape content from a URL with caching"""
    # Check cache first
    cached_content = load_from_cache(url)
    if cached_content is not None:
        return True, cached_content
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Scraping attempt {attempt+1}/{max_retries} for URL: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle PDF and other non-HTML content
            if 'application/pdf' in content_type:
                return False, "PDF content not supported for scraping."
            
            # Proceed with HTML parsing
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and irrelevant elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'svg', 'form']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            
            # Look for common content containers
            content_selectors = [
                'main', 'article', '#content', '.content', '.post', '.article', '.post-content',
                '[role="main"]', '#main', '.main'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Use the largest content element
                    main_content = max(elements, key=lambda x: len(x.get_text()))
                    break
            
            # If no content container found, use the body
            if not main_content:
                main_content = soup.body
            
            # Get text content
            if main_content:
                text = main_content.get_text(separator="\n")
            else:
                text = soup.get_text(separator="\n")
            
            # Clean up text
            text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
            text = re.sub(r'\s+', ' ', text)    # Remove multiple spaces
            text = re.sub(r'\.+', '.', text)    # Remove multiple periods
            text = text.strip()
            
            # Check if meaningful content was extracted
            if not text or len(text) < 150:
                if attempt < max_retries - 1:
                    continue
                return False, "Content too short or empty."
            
            # Truncate if too long (to handle API limits)
            max_text_length = 20000
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # Cache the content
            save_to_cache(url, text)
                
            return True, text
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            return False, "Request timed out."
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else "unknown"
            return False, f"HTTP error {status_code}: {str(e)}"
            
        except Exception as e:
            logger.error(f"Scraping error for {url}: {str(e)}")
            return False, f"Error: {str(e)}"
    
    return False, "Failed after multiple attempts."

# Summarize content using Gemini API with improved prompt
def summarize_with_gemini(
    content: str, 
    query: str, 
    api_key: str,
    summary_style: str = "comprehensive"
) -> Tuple[bool, str]:
    """Generate a summary of content using Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Different prompts based on summary style
        if summary_style == "concise":
            prompt = f"""
            Provide a brief summary (250-350 words) of the following content, focusing specifically on information relevant to: "{query}"
            
            Extract only:
            1. The most critical facts directly related to the query
            2. Key conclusions or findings
            
            Content:
            {content}
            
            Keep your summary focused, factual, and brief.
            """
        else:  # comprehensive
            prompt = f"""
            Please provide a comprehensive summary of the following content in relation to this search query: "{query}"
            
            Extract and organize:
            1. Key facts and primary information directly related to the query
            2. Important supporting details and context
            3. Significant findings, insights or conclusions
            4. Any notable contradictions or limitations mentioned
            
            Content:
            {content}
            
            Create a well-structured summary (400-600 words) that thoroughly captures the essence of this content 
            as it relates to the query. Use bullet points where appropriate for clarity.
            """
        
        response = model.generate_content(prompt)
        summary = response.text
        
        return True, summary
    except Exception as e:
        logger.error(f"Failed to summarize with Gemini API: {str(e)}")
        return False, f"Failed to summarize with Gemini API: {str(e)}"

# Create a comprehensive summary from all individual summaries
def create_comprehensive_summary(
    summaries: List[Dict], 
    query: str, 
    api_key: str,
    research_style: str = "balanced"
) -> Tuple[bool, str]:
    """Create a comprehensive research summary from individual summaries"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create a context from all summaries
        combined_content = ""
        for i, summary_data in enumerate(summaries, 1):
            combined_content += f"\n\nSOURCE {i}: {summary_data['title']}\n"
            combined_content += f"{summary_data['summary']}\n"
            combined_content += f"URL: {summary_data['link']}\n"
        
        # Different prompts based on research style
        if research_style == "academic":
            prompt = f"""
            Please synthesize a formal academic research summary from the following sources related to: "{query}"
            
            Your task is to:
            1. Identify the main findings, key points, and methodologies across all sources
            2. Note areas of scholarly consensus and significant disagreements 
            3. Evaluate the strength of evidence and methodological approaches
            4. Organize information into a structured academic format with clear section headings
            5. Include a literature review section and research implications
            
            Here are the summaries from various sources:
            {combined_content}
            
            Please create a well-structured, academically-oriented research synthesis with the following sections:
            - Abstract (brief overview)
            - Introduction & Research Context
            - Literature Review & Methodology
            - Key Findings (with subsections as needed)
            - Discussion & Analysis
            - Limitations of Current Research
            - Conclusion & Future Directions
            
            Format this as a formal academic summary suitable for an educated audience.
            """
        elif research_style == "business":
            prompt = f"""
            Please synthesize an executive-style business research brief from the following sources related to: "{query}"
            
            Your task is to:
            1. Extract actionable insights and business implications
            2. Identify market trends, opportunities, and challenges
            3. Highlight competitive factors and strategic considerations
            4. Provide clear, decision-relevant information
            5. Organize with executive-friendly formatting and bullet points
            
            Here are the summaries from various sources:
            {combined_content}
            
            Please create a business-focused research brief with the following sections:
            - Executive Summary
            - Key Market Insights
            - Strategic Implications
            - Competitive Analysis
            - Recommendations & Next Steps
            
            Format this as a concise, action-oriented business brief with bullet points for easy scanning. Focus on practical business applications rather than theoretical discussions.
            """
        else:  # balanced
            prompt = f"""
            Please synthesize a comprehensive research conclusion from the following summaries related to this search query: "{query}"
            
            Your task is to:
            1. Identify the main findings and key points across all sources
            2. Note any consensus or disagreements between sources
            3. Highlight the most important and relevant information
            4. Evaluate the reliability and comprehensiveness of the sources
            5. Organize the information logically with clear section headings
            6. Include a brief conclusion with the most important takeaways
            
            Here are the summaries from various sources:
            {combined_content}
            
            Please create a well-structured, comprehensive research conclusion that synthesizes all this information.
            Include a "Key Findings" section at the beginning and a "Conclusion" at the end.
            Use bullet points where appropriate for clarity and readability.
            """
        
        response = model.generate_content(prompt)
        comprehensive_summary = response.text
        
        return True, comprehensive_summary
    except Exception as e:
        logger.error(f"Failed to create comprehensive summary: {str(e)}")
        return False, f"Failed to create comprehensive summary: {str(e)}"

# Generate research questions for further exploration
def generate_followup_questions(
    summary: str,
    query: str,
    api_key: str
) -> Tuple[bool, List[str]]:
    """Generate follow-up research questions based on the summary"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        Based on the following research summary about "{query}", please generate 5 specific follow-up questions
        that would help deepen understanding or explore important aspects not fully covered in the current research.
        
        Research Summary:
        {summary}
        
        For each question:
        1. Focus on gaps, unexplored angles, or areas needing clarification
        2. Make questions specific and answerable through further research
        3. Avoid questions already thoroughly addressed in the summary
        
        Format your response as a numbered list of 5 questions only, without additional explanation.
        """
        
        response = model.generate_content(prompt)
        questions_text = response.text
        
        # Parse the questions into a list
        questions = []
        for line in questions_text.split('\n'):
            # Clean up and extract questions
            line = line.strip()
            if re.match(r'^\d+\.', line):  # Matches lines starting with number and period
                # Remove leading number and whitespace
                question = re.sub(r'^\d+\.\s*', '', line)
                if question:
                    questions.append(question)
        
        # Ensure we have questions
        if not questions:
            # Try an alternative parsing approach
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        
        # Limit to 5 questions
        questions = questions[:5]
        
        return True, questions
    except Exception as e:
        logger.error(f"Failed to generate follow-up questions: {str(e)}")
        return False, []

# Function to handle parallel web scraping
def scrape_urls_parallel(urls: List[str], max_workers: int = 3) -> Dict[str, Tuple[bool, str]]:
    """Scrape multiple URLs in parallel"""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_url = {executor.submit(scrape_web_page, url): url for url in urls}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                success, content = future.result()
                results[url] = (success, content)
            except Exception as e:
                logger.error(f"Exception while scraping {url}: {str(e)}")
                results[url] = (False, f"Error: {str(e)}")
    
    return results

# Main app function
def main():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: #0D47A1;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E0E0E0;
        margin-bottom: 1rem;
        background-color: #F9F9F9;
    }
    .source-link {
        color: #1976D2;
        text-decoration: none;
        font-weight: 500;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .snippet-text {
        font-style: italic;
        color: #616161;
        margin-bottom: 1rem;
    }
    .summary-section {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .error-text {
        color: #D32F2F;
        font-weight: 500;
    }
    .success-text {
        color: #388E3C;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔍 Web Research Agent Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p>Advanced web research assistant powered by Google Search, AI summarization with Google Gemini 1.5 Pro, 
    and intelligent content analysis. Get comprehensive research results on any topic.</p>
    """, unsafe_allow_html=True)
    
    # Initialize session state for API keys and settings
    if 'api_keys_loaded' not in st.session_state:
        st.session_state.api_keys_loaded = False
        st.session_state.gemini_api_key = ""
        st.session_state.serpapi_key = ""
        st.session_state.keys_tested = False
        st.session_state.show_key_input = False
        st.session_state.search_history = []
    
    # Default API key file paths
    default_gemini_key_path = "/home/harsha/Downloads/Gemini-API.docx"
    default_serpapi_key_path = "/home/harsha/Downloads/Serp_API.odt"
    
    # Sidebar for API key configuration and advanced settings
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # Try auto-loading keys first (only once)
        if not st.session_state.api_keys_loaded and not st.session_state.show_key_input:
            with st.spinner("Loading API keys..."):
                # Check if default paths exist and try to read keys
                gemini_key = ""
                serpapi_key = ""
                
                if os.path.exists(default_gemini_key_path):
                    gemini_key = read_gemini_key(default_gemini_key_path)
                
                if os.path.exists(default_serpapi_key_path):    
                    serpapi_key = read_serpapi_key(default_serpapi_key_path)
                
                if gemini_key and serpapi_key:
                    st.session_state.gemini_api_key = gemini_key
                    st.session_state.serpapi_key = serpapi_key
                    st.session_state.api_keys_loaded = True
                    st.success("API keys loaded successfully!")
                else:
                    st.session_state.show_key_input = True
                    st.info("Please enter your API keys manually.")
        
        # Manual key entry
        if st.session_state.show_key_input or st.checkbox("Edit API Keys", value=False):
            # Gemini API Key
            gemini_key_input = st.text_input(
                "Gemini API Key", 
                type="password", 
                key="gemini_key_input", 
                value=st.session_state.gemini_api_key if st.session_state.api_keys_loaded else "",
                help="Enter your Google Gemini API key"
            )
            
            # File upload option for Gemini API key
            gemini_key_file = st.file_uploader(
                "Or upload a file containing Gemini API key", 
                type=["docx", "txt"],
                help="Upload a document containing your Gemini API key"
            )
            
            # SerpAPI Key
            serpapi_key_input = st.text_input(
                "SerpAPI Key", 
                type="password", 
                key="serpapi_key_input",
                value=st.session_state.serpapi_key if st.session_state.api_keys_loaded else "",
                help="Enter your SerpAPI key"
            )
            
            # File upload option for SerpAPI key
            serpapi_key_file = st.file_uploader(
                "Or upload a file containing SerpAPI key", 
                type=["odt", "txt"],
                help="Upload a document containing your SerpAPI key"
            )
            
            if st.button("Save Keys"):
                # Process uploaded files if any
                # Process uploaded files if any
                if gemini_key_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"temp_gemini_key.{gemini_key_file.name.split('.')[-1]}"
                    with open(temp_path, "wb") as f:
                        f.write(gemini_key_file.getbuffer())
                    gemini_key_from_file = read_gemini_key(temp_path)
                    if gemini_key_from_file:
                        gemini_key_input = gemini_key_from_file
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                if serpapi_key_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"temp_serpapi_key.{serpapi_key_file.name.split('.')[-1]}"
                    with open(temp_path, "wb") as f:
                        f.write(serpapi_key_file.getbuffer())
                    serpapi_key_from_file = read_serpapi_key(temp_path)
                    if serpapi_key_from_file:
                        serpapi_key_input = serpapi_key_from_file
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                if gemini_key_input and serpapi_key_input:
                    st.session_state.gemini_api_key = gemini_key_input
                    st.session_state.serpapi_key = serpapi_key_input
                    st.session_state.api_keys_loaded = True
                    st.session_state.keys_tested = False  # Reset so keys will be tested
                    
                    # Test the keys silently
                    with st.spinner("Testing API keys..."):
                        gemini_success, serpapi_success, gemini_msg, serpapi_msg = test_api_keys(
                            st.session_state.gemini_api_key,
                            st.session_state.serpapi_key
                        )
                        
                        if gemini_success and serpapi_success:
                            st.success("Both API keys are valid! Ready to research.")
                        else:
                            if not gemini_success:
                                st.error(f"Gemini API key issue: {gemini_msg}")
                            if not serpapi_success:
                                st.error(f"SerpAPI key issue: {serpapi_msg}")
                else:
                    st.error("Both API keys are required.")
        
        # Add advanced settings section if API keys are loaded
        if st.session_state.api_keys_loaded:
            st.header("⚙️ Advanced Settings")
            
            # Search Settings
            st.subheader("Search Settings")
            st.session_state.num_results = st.slider(
                "Number of results to fetch", 
                min_value=3, 
                max_value=10, 
                value=5,
                help="More results provide broader coverage but take longer to process"
            )
            
            st.session_state.search_region = st.selectbox(
                "Search region",
                options=["us", "uk", "ca", "au", "in", "de", "fr", "es", "it", "jp"],
                index=0,
                help="Country/region for search results"
            )
            
            # Summary Settings
            st.subheader("Summary Settings")
            st.session_state.summary_style = st.radio(
                "Individual summary style",
                options=["comprehensive", "concise"],
                index=0,
                help="Comprehensive provides more details, concise is shorter"
            )
            
            st.session_state.research_style = st.radio(
                "Final research style",
                options=["balanced", "academic", "business"],
                index=0,
                help="Choose the style of the final research summary"
            )
            
            # Cache Settings
            st.subheader("Cache Settings")
            if st.button("Clear Cache"):
                try:
                    for file in os.listdir("cache"):
                        if file.endswith(".pkl"):
                            os.remove(os.path.join("cache", file))
                    st.success("Cache cleared successfully!")
                except Exception as e:
                    st.error(f"Failed to clear cache: {str(e)}")
            
            # Add debug options
            st.subheader("Debug Options")
            if st.button("Check Available Gemini Models"):
                with st.spinner("Fetching available models..."):
                    models = list_available_models(st.session_state.gemini_api_key)
                    if models:
                        st.success(f"Available models: {', '.join(models)}")
                    else:
                        st.error("Failed to fetch available models.")
            
            if st.button("Test API Keys"):
                with st.spinner("Testing API keys..."):
                    gemini_success, serpapi_success, gemini_msg, serpapi_msg = test_api_keys(
                        st.session_state.gemini_api_key,
                        st.session_state.serpapi_key
                    )
                    
                    if gemini_success:
                        st.success(f"Gemini API: ✅ {gemini_msg}")
                    else:
                        st.error(f"Gemini API: ❌ {gemini_msg}")
                        
                    if serpapi_success:
                        st.success(f"SerpAPI: ✅ {serpapi_msg}")
                    else:
                        st.error(f"SerpAPI: ❌ {serpapi_msg}")

    # Main content area - Search interface
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., Latest advancements in quantum computing",
        help="Be specific for better results"
    )
    
    # Show recent searches if available
    if 'search_history' in st.session_state and st.session_state.search_history:
        with st.expander("Recent searches", expanded=False):
            for i, past_query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"🔍 {past_query}", key=f"history_{i}"):
                    query = past_query
    
    # Search control row
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button(
            "🔍 Research",
            type="primary",
            disabled=not st.session_state.api_keys_loaded or not query,
            help="Start the research process"
        )
    
    with col2:
        export_format = st.selectbox(
            "Export as",
            options=["Markdown", "Text", "HTML"],
            index=0,
            disabled=not hasattr(st.session_state, 'comprehensive_summary'),
            help="Choose export format for research results"
        )
    
    with col3:
        if hasattr(st.session_state, 'comprehensive_summary'):
            if export_format == "Markdown":
                export_content = st.session_state.comprehensive_summary
            elif export_format == "Text":
                # Convert markdown to plain text (simple conversion)
                export_content = re.sub(r'#+ ', '', st.session_state.comprehensive_summary)
                export_content = re.sub(r'\*\*(.*?)\*\*', r'\1', export_content)
                export_content = re.sub(r'\*(.*?)\*', r'\1', export_content)
            else:  # HTML
                # Convert markdown to HTML using a simple approach
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Research: {query}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                        h1, h2, h3 {{ color: #2C3E50; }}
                        h1 {{ border-bottom: 2px solid #3498DB; padding-bottom: 10px; }}
                        h2 {{ border-bottom: 1px solid #BDC3C7; padding-bottom: 5px; margin-top: 30px; }}
                        p {{ margin-bottom: 16px; }}
                        ul, ol {{ padding-left: 25px; }}
                        blockquote {{ background-color: #F8F9F9; padding: 15px; border-left: 5px solid #3498DB; margin: 15px 0; }}
                        .footer {{ margin-top: 40px; font-size: 0.8em; color: #7F8C8D; border-top: 1px solid #EAECEE; padding-top: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>Research: {query}</h1>
                    <div class="content">
                        {st.session_state.comprehensive_summary.replace("\n", "<br>").replace("# ", "<h1>").replace("## ", "<h2>").replace("### ", "<h3>")}
                    </div>
                    <div class="footer">
                        <p>Generated by Web Research Agent Pro on {datetime.now().strftime('%Y-%m-%d')}</p>
                    </div>
                </body>
                </html>
                """
                export_content = html_content
            
            st.download_button(
                label="⬇️ Download Results",
                data=export_content,
                file_name=f"research_{re.sub(r'[^\w]', '_', query)[:30]}_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                mime="text/plain"
            )
    
    if search_button and query:
        if not st.session_state.api_keys_loaded:
            st.error("Please enter your API keys in the sidebar first.")
            st.session_state.show_key_input = True
            return
        
        # Add to search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
            # Keep only the last 10 searches
            if len(st.session_state.search_history) > 10:
                st.session_state.search_history.pop(0)
        
        # Get search settings from session state
        num_results = st.session_state.get('num_results', 5)
        search_region = st.session_state.get('search_region', 'us')
        summary_style = st.session_state.get('summary_style', 'comprehensive')
        research_style = st.session_state.get('research_style', 'balanced')
        
        # Search for results
        search_results = search_with_serpapi(
            query, 
            st.session_state.serpapi_key,
            num_results=num_results,
            search_region=search_region
        )
        
        if not search_results:
            st.error("No search results found. Please try a different query or check your SerpAPI key.")
            return
        
        st.success(f"Found {len(search_results)} search results")
        
        # Container for results
        st.markdown('<h2 class="subheader">Research Sources</h2>', unsafe_allow_html=True)
        
        # Extract URLs for parallel scraping
        urls_to_scrape = [result.get('link', '') for result in search_results if result.get('link')]
        
        # Scrape all web pages in parallel
        with st.spinner(f"Scraping web content from {len(urls_to_scrape)} sources..."):
            scraped_results = scrape_urls_parallel(urls_to_scrape)
        
        # Initialize list to store successful summaries
        successful_summaries = []
        
        # Progress bar for summarization
        summarization_progress = st.progress(0)
        
        # Process each search result
        for i, result in enumerate(search_results):
            # Update progress
            progress_value = i / len(search_results)
            summarization_progress.progress(progress_value, text=f"Summarizing source {i+1}/{len(search_results)}...")
            
            with st.expander(f"Source {i+1}: {result.get('title', 'No Title')}", expanded=False):
                st.markdown(f"<strong>Source:</strong> <a href='{result.get('link', '#')}' class='source-link' target='_blank'>{result.get('title', 'No Title')}</a>", unsafe_allow_html=True)
                st.markdown(f"<div class='snippet-text'>{result.get('snippet', 'No snippet available')}</div>", unsafe_allow_html=True)
                
                # Get scraped content
                url = result.get('link', '')
                if url in scraped_results:
                    success, content = scraped_results[url]
                    
                    if not success:
                        st.error(f"Failed to scrape content: {content}")
                        continue
                    
                    # Summarize content
                    with st.spinner("Generating summary with Gemini 1.5 Pro..."):
                        summary_success, summary = summarize_with_gemini(
                            content, 
                            query, 
                            st.session_state.gemini_api_key,
                            summary_style=summary_style
                        )
                        
                        if not summary_success:
                            st.error(f"Failed to generate summary: {summary}")
                            # Show a snippet of the raw content as fallback
                            st.markdown("### Raw Content Preview (Fallback)")
                            st.markdown(content[:300] + "...")
                        else:
                            st.markdown("<div class='summary-section'>", unsafe_allow_html=True)
                            st.markdown("### Summary")
                            st.markdown(summary)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Store successful summary
                            successful_summaries.append({
                                'title': result.get('title', 'No Title'),
                                'link': result.get('link', '#'),
                                'summary': summary
                            })
        
        # Complete the progress bar
        summarization_progress.progress(1.0, text="Summarization complete!")
        time.sleep(0.5)
        summarization_progress.empty()
        
        # Final comprehensive summary of findings
        st.markdown('<h2 class="subheader">Research Synthesis</h2>', unsafe_allow_html=True)
        
        if successful_summaries:
            with st.spinner(f"Generating comprehensive research synthesis ({research_style} style)..."):
                comprehensive_success, comprehensive_summary = create_comprehensive_summary(
                    successful_summaries,
                    query,
                    st.session_state.gemini_api_key,
                    research_style=research_style
                )
                
                if comprehensive_success:
                    # Store the summary in session state for export functionality
                    st.session_state.comprehensive_summary = comprehensive_summary
                    
                    # Display the summary
                    st.markdown(comprehensive_summary)
                    
                    # Generate follow-up questions
                    success, followup_questions = generate_followup_questions(
                        comprehensive_summary,
                        query,
                        st.session_state.gemini_api_key
                    )
                    
                    if success and followup_questions:
                        st.markdown("### Suggested Further Research Questions")
                        for q in followup_questions:
                            col1, col2 = st.columns([10, 1])
                            with col1:
                                st.markdown(f"- {q}")
                            with col2:
                                # Add button to start new search with this question
                                if st.button("🔍", key=f"search_{q[:20]}", help=f"Research this question"):
                                    query = q
                                    # Trigger a rerun with the new query
                                    st.experimental_rerun()
                else:
                    st.error("Failed to generate comprehensive summary.")
                    st.markdown("""
                    Review the individual summaries above to get insights on your query.
                    """)
        else:
            st.info("""
            No successful summaries were generated. Please try a different query or check if content 
            was properly scraped from the search results.
            """)

# Initialize the app
if __name__ == "__main__":
    main()
