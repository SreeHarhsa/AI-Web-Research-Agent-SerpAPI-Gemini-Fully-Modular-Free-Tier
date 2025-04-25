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
from typing import List, Dict, Optional, Tuple
import traceback
import logging

# Configure logging to file instead of UI
logging.basicConfig(level=logging.INFO, filename="web_research_agent.log")
logger = logging.getLogger(__name__)

# Set the page configuration and title
st.set_page_config(
    page_title="Web Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Function to read Gemini API key from a .docx file
def read_gemini_key(docx_path: str) -> str:
    try:
        logger.info(f"Reading Gemini API key from: {docx_path}")
        doc = docx.Document(docx_path)
        full_text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        api_key_match = re.search(r'[A-Za-z0-9_-]{39}', full_text)
        if api_key_match:
            return api_key_match.group(0)
        
        # Try alternative pattern
        api_key_match = re.search(r'AIza[A-Za-z0-9_-]{32}', full_text)
        if api_key_match:
            return api_key_match.group(0)
            
        # Try a more generic pattern
        api_key_match = re.search(r'[A-Za-z0-9_-]{30,50}', full_text)
        if api_key_match:
            return api_key_match.group(0)
            
        logger.error("Couldn't find Gemini API key in the document.")
        return ""
    except Exception as e:
        logger.error(f"Error reading Gemini API key: {str(e)}")
        return ""

# Function to read SerpAPI key from an .odt file
def read_serpapi_key(odt_path: str) -> str:
    try:
        logger.info(f"Reading SerpAPI key from: {odt_path}")
        with zipfile.ZipFile(odt_path, 'r') as z:
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
            
            # Look for an API key pattern (typical SerpAPI key format - 32 hex characters)
            api_key_match = re.search(r'[a-z0-9]{32}', all_text)
            if api_key_match:
                return api_key_match.group(0)
            
            # Try a more generic pattern if specific pattern fails
            api_key_match = re.search(r'[a-z0-9]{20,40}', all_text)
            if api_key_match:
                return api_key_match.group(0)
                
            logger.error("Couldn't find SerpAPI key pattern in the document.")
            return ""
    except Exception as e:
        logger.error(f"Error reading SerpAPI key: {str(e)}")
        return ""

# Function to list available Gemini models - for debugging purposes
def list_available_models(api_key: str) -> List[str]:
    try:
        genai.configure(api_key=api_key)
        available_models = genai.list_models()
        model_names = [model.name for model in available_models]
        logger.info(f"Available models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        return []

# Test API keys silently - no UI output
def test_api_keys(gemini_key: str, serpapi_key: str) -> Tuple[bool, bool]:
    gemini_success = False
    serpapi_success = False
    
    # Test Gemini API
    try:
        genai.configure(api_key=gemini_key)
        # Use gemini-1.5-pro instead of gemini-1.0-pro
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Test")
        if response and hasattr(response, 'text'):
            gemini_success = True
            logger.info("Gemini API key is valid")
        else:
            logger.error("Gemini API key appears invalid")
    except Exception as e:
        logger.error(f"Gemini API test failed: {str(e)}")
        
        # Try to list available models to see what's accessible
        list_available_models(gemini_key)
    
    # Test SerpAPI
    try:
        response = requests.get("https://serpapi.com/account", params={"api_key": serpapi_key})
        if response.status_code == 200:
            serpapi_success = True
            logger.info("SerpAPI key is valid")
        else:
            logger.error(f"SerpAPI key test failed with status code: {response.status_code}")
    except Exception as e:
        logger.error(f"SerpAPI test failed: {str(e)}")
    
    return gemini_success, serpapi_success

# Search Google using SerpAPI
def search_with_serpapi(query: str, api_key: str, max_retries: int = 3) -> List[Dict]:
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": 5  # Limit to top 5 results
    }
    
    with st.status("Searching Google...") as status:
        for attempt in range(max_retries):
            try:
                logger.info(f"SerpAPI search attempt {attempt+1}/{max_retries}")
                response = requests.get("https://serpapi.com/search", params=params)
                
                if response.status_code == 401:
                    status.update(label="Search failed: Authentication error", state="error")
                    time.sleep(1)  # Give user time to see the error
                    return []
                    
                response.raise_for_status()
                results = response.json()
                
                # Extract organic results
                if "organic_results" in results:
                    status.update(label="Search complete!", state="complete")
                    return results["organic_results"]
                else:
                    status.update(label="No results found", state="error")
                    logger.warning("No organic results found in SerpAPI response.")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    status.update(label=f"Rate limited. Retrying ({attempt+1}/{max_retries})...", state="running")
                    time.sleep(wait_time)  # Exponential backoff
                else:
                    status.update(label=f"Search error: {e}", state="error")
                    logger.error(f"HTTP error from SerpAPI: {e}")
                    return []
                    
            except Exception as e:
                status.update(label=f"Search error: {str(e)}", state="error")
                logger.error(f"Error searching with SerpAPI: {str(e)}")
                return []
    
    return []

# Scrape content from a URL
def scrape_web_page(url: str, max_retries: int = 2) -> Tuple[bool, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Scraping attempt {attempt+1}/{max_retries} for URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()
            
            # Get text content
            text = soup.get_text(separator="\n")
            
            # Clean up text: remove multiple newlines and spaces
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            if not text or len(text) < 100:
                return False, "Content too short or empty."
            
            # Truncate if too long (to handle API limits)
            if len(text) > 15000:
                text = text[:15000] + "..."
                
            return True, text
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
            return False, "Request timed out."
            
        except requests.exceptions.HTTPError as e:
            return False, f"HTTP error: {e}"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    return False, "Failed after multiple attempts."

# Summarize content using Gemini API
def summarize_with_gemini(content: str, query: str, api_key: str) -> Tuple[bool, str]:
    try:
        genai.configure(api_key=api_key)
        # Use gemini-1.5-pro instead of gemini-1.0-pro
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        Please provide a comprehensive summary of the following content in relation to this search query: "{query}"
        
        Extract and highlight:
        1. Key facts and information
        2. Main points relevant to the query
        3. Important insights or conclusions
        
        Content:
        {content}
        
        Please provide a concise but thorough summary that captures the essence of this content as it relates to the query.
        """
        
        response = model.generate_content(prompt)
        summary = response.text
        
        return True, summary
    except Exception as e:
        logger.error(f"Failed to summarize with Gemini API: {str(e)}")
        return False, f"Failed to summarize with Gemini API: {str(e)}"

# New function for individual article summarization
def create_individual_summary(content: str, query: str, api_key: str, is_brief: bool = False) -> Tuple[bool, str]:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        length_instruction = "brief and concise (100-150 words)" if is_brief else "comprehensive (300-500 words)"
        
        prompt = f"""
        Please provide a {length_instruction} summary of the following content in relation to this search query: "{query}"
        
        Focus on:
        1. Key facts and information directly relevant to the query
        2. Main insights, findings, or advancements mentioned
        3. Any important conclusions or future implications
        
        Content:
        {content}
        
        Format your summary with clear sections and bullet points where appropriate to enhance readability.
        """
        
        response = model.generate_content(prompt)
        summary = response.text
        
        return True, summary
    except Exception as e:
        logger.error(f"Failed to create individual summary: {str(e)}")
        return False, f"Failed to create individual summary: {str(e)}"

# Function to create comprehensive summary from all individual summaries
def create_comprehensive_summary(summaries: List[Dict], query: str, api_key: str) -> Tuple[bool, str]:
    try:
        genai.configure(api_key=api_key)
        # Use gemini-1.5-pro instead of gemini-1.0-pro
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create a context from all summaries
        combined_content = ""
        for i, summary_data in enumerate(summaries, 1):
            combined_content += f"\n\nSOURCE {i}: {summary_data['title']}\n"
            combined_content += f"{summary_data['summary']}\n"
            combined_content += f"URL: {summary_data['link']}\n"
        
        prompt = f"""
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
        """
        
        response = model.generate_content(prompt)
        comprehensive_summary = response.text
        
        return True, comprehensive_summary
    except Exception as e:
        logger.error(f"Failed to create comprehensive summary: {str(e)}")
        return False, f"Failed to create comprehensive summary: {str(e)}"

# Main app function
def main():
    st.title("🔍 Web Research Agent")
    st.markdown("""
    Enter a search query to get summaries from the top web results, powered by Google Search, web scraping, 
    and AI summarization with Google Gemini 1.5 Pro.
    """)
    
    # Initialize session state for API keys and other state
    if 'api_keys_loaded' not in st.session_state:
        st.session_state.api_keys_loaded = False
        st.session_state.gemini_api_key = ""
        st.session_state.serpapi_key = ""
        st.session_state.keys_tested = False
        st.session_state.show_key_input = False
        st.session_state.summary_length = "comprehensive"  # New state for summary length
    
    # API key file paths
    gemini_key_path = "/home/harsha/Downloads/Gemini-API.docx"
    serpapi_key_path = "/home/harsha/Downloads/Serp_API.odt"
    
    # Sidebar for API key configuration
    with st.sidebar:
        st.header("API Configuration")
        
        # Try auto-loading keys first (only once)
        if not st.session_state.api_keys_loaded and not st.session_state.show_key_input:
            with st.spinner("Loading API keys..."):
                # Silently try to read keys from files
                gemini_key = read_gemini_key(gemini_key_path)
                serpapi_key = read_serpapi_key(serpapi_key_path)
                
                if gemini_key and serpapi_key:
                    st.session_state.gemini_api_key = gemini_key
                    st.session_state.serpapi_key = serpapi_key
                    st.session_state.api_keys_loaded = True
                    st.success("API keys loaded successfully!")
                else:
                    st.session_state.show_key_input = True
        
        # Manual key entry (show if auto-loading failed or user wants to change keys)
        if st.session_state.show_key_input or st.checkbox("Edit API Keys", value=False):
            st.text_input("Gemini API Key", type="password", key="gemini_key_input", 
                        value=st.session_state.gemini_api_key if st.session_state.api_keys_loaded else "",
                        help="Enter your Google Gemini API key")
            
            st.text_input("SerpAPI Key", type="password", key="serpapi_key_input",
                        value=st.session_state.serpapi_key if st.session_state.api_keys_loaded else "",
                        help="Enter your SerpAPI key")
            
            if st.button("Save Keys"):
                if st.session_state.gemini_key_input and st.session_state.serpapi_key_input:
                    st.session_state.gemini_api_key = st.session_state.gemini_key_input
                    st.session_state.serpapi_key = st.session_state.serpapi_key_input
                    st.session_state.api_keys_loaded = True
                    st.session_state.keys_tested = False  # Reset so keys will be tested
                    
                    # List available models if API key is provided
                    with st.spinner("Checking available Gemini models..."):
                        list_available_models(st.session_state.gemini_api_key)
                        
                    st.success("API keys saved!")
                else:
                    st.error("Both API keys are required.")
        
        # Add a button to check available models
        if st.session_state.api_keys_loaded:
            if st.button("Check Available Gemini Models"):
                with st.spinner("Fetching available models..."):
                    models = list_available_models(st.session_state.gemini_api_key)
                    if models:
                        st.success(f"Available models: {', '.join(models)}")
                    else:
                        st.error("Failed to fetch available models.")
        
        # Settings section in sidebar
        st.header("Settings")
        
        # Summary length option
        summary_length = st.radio(
            "Individual Summary Length:",
            options=["brief", "comprehensive"],
            index=1 if st.session_state.summary_length == "comprehensive" else 0,
            help="Brief: 100-150 words | Comprehensive: 300-500 words"
        )
        st.session_state.summary_length = summary_length
    
    # Main content area
    # Input for search query
    query = st.text_input("Enter your search query:", placeholder="e.g., Latest advancements in quantum computing")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        # Research button
        search_button = st.button("Research", type="primary", disabled=not st.session_state.api_keys_loaded)
    
    if search_button and query:
        if not st.session_state.api_keys_loaded:
            st.error("Please enter your API keys in the sidebar first.")
            st.session_state.show_key_input = True
            return
            
        # Search for results
        search_results = search_with_serpapi(query, st.session_state.serpapi_key)
        
        if not search_results:
            st.error("No search results found. Please try a different query or check your SerpAPI key.")
            return
            
        st.success(f"Found {len(search_results)} search results")
        
        # Container for results
        st.subheader("Research Results")
        
        # Initialize list to store successful summaries
        successful_summaries = []
        
        # Process each search result
        for i, result in enumerate(search_results):
            with st.expander(f"Result {i+1}: {result.get('title', 'No Title')}"):
                st.markdown(f"**Source**: [{result.get('title', 'No Title')}]({result.get('link', '#')})")
                st.markdown(f"**Snippet**: {result.get('snippet', 'No snippet available')}")
                
                # Scrape the webpage
                with st.spinner(f"Scraping content from result {i+1}..."):
                    success, content = scrape_web_page(result.get('link', ''))
                    
                    if not success:
                        st.error(f"Failed to scrape content: {content}")
                        continue
                    
                    # Create tabs for different views
                    sum_tab, raw_tab = st.tabs(["Summary", "Raw Content"])
                    
                    with sum_tab:
                        # Summarize content
                        is_brief = st.session_state.summary_length == "brief"
                        with st.spinner(f"Generating {st.session_state.summary_length} summary with Gemini 1.5 Pro..."):
                            summary_success, summary = create_individual_summary(
                                content, 
                                query, 
                                st.session_state.gemini_api_key,
                                is_brief
                            )
                            
                            if not summary_success:
                                st.error(f"Failed to generate summary: {summary}")
                            else:
                                st.markdown(summary)
                                
                                # Store successful summary
                                successful_summaries.append({
                                    'title': result.get('title', 'No Title'),
                                    'link': result.get('link', '#'),
                                    'summary': summary
                                })
                    
                    with raw_tab:
                        st.markdown("### Raw Content")
                        # Show the first 1000 characters of raw content
                        st.markdown(content[:1000] + "..." if len(content) > 1000 else content)
                        
                        # Download button for full raw content
                        st.download_button(
                            label="Download Full Content",
                            data=content,
                            file_name=f"raw_content_{i+1}.txt",
                            mime="text/plain"
                        )
        
        # Final comprehensive summary of findings
        if successful_summaries:
            st.subheader("Research Conclusion")
            
            with st.spinner("Generating comprehensive research conclusion..."):
                comprehensive_success, comprehensive_summary = create_comprehensive_summary(
                    successful_summaries,
                    query,
                    st.session_state.gemini_api_key
                )
                
                if comprehensive_success:
                    st.markdown(comprehensive_summary)
                    
                    # Add download button for the research conclusion
                    st.download_button(
                        label="Download Research Conclusion",
                        data=comprehensive_summary,
                        file_name=f"research_conclusion_{query[:30]}.md",
                        mime="text/markdown"
                    )
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
