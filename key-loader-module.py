import os
import re
import logging
import docx
import zipfile
import xml.etree.ElementTree as ET
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_gemini_key(docx_path: str) -> Optional[str]:
    """
    Read Gemini API key from a .docx file.
    
    Args:
        docx_path: Path to the .docx file containing the Gemini API key
        
    Returns:
        API key if found, None otherwise
    """
    try:
        logger.info(f"Reading Gemini API key from: {docx_path}")
        doc = docx.Document(docx_path)
        full_text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        
        # Try different patterns for API key
        patterns = [
            r'[A-Za-z0-9_-]{39}',  # Standard Gemini API key pattern
            r'AIza[A-Za-z0-9_-]{32}',  # Alternative pattern
            r'[A-Za-z0-9_-]{30,50}'  # Generic pattern
        ]
        
        for pattern in patterns:
            api_key_match = re.search(pattern, full_text)
            if api_key_match:
                key = api_key_match.group(0)
                logger.info("Successfully extracted Gemini API key")
                return key
                
        logger.error("Couldn't find Gemini API key in the document.")
        return None
    except Exception as e:
        logger.error(f"Error reading Gemini API key: {str(e)}")
        return None

def read_serpapi_key(odt_path: str) -> Optional[str]:
    """
    Read SerpAPI key from an .odt file.
    
    Args:
        odt_path: Path to the .odt file containing the SerpAPI key
        
    Returns:
        API key if found, None otherwise
    """
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
            
            # Look for SerpAPI key pattern (typical SerpAPI key format - 32 hex characters)
            api_key_match = re.search(r'[a-z0-9]{32}', all_text)
            if api_key_match:
                key = api_key_match.group(0)
                logger.info("Successfully extracted SerpAPI key")
                return key
            
            # Try a more generic pattern if specific pattern fails
            api_key_match = re.search(r'[a-z0-9]{20,40}', all_text)
            if api_key_match:
                key = api_key_match.group(0)
                logger.info("Successfully extracted SerpAPI key using generic pattern")
                return key
                
            logger.error("Couldn't find SerpAPI key pattern in the document.")
            return None
    except Exception as e:
        logger.error(f"Error reading SerpAPI key: {str(e)}")
        return None

def load_api_keys(gemini_docx_path: str, serpapi_odt_path: str):
    """
    Load both API keys from respective files and set environment variables.
    
    Args:
        gemini_docx_path: Path to the .docx file containing Gemini API key
        serpapi_odt_path: Path to the .odt file containing SerpAPI key
        
    Returns:
        Tuple of (gemini_key, serpapi_key)
    """
    # Read keys from files
    gemini_key = read_gemini_key(gemini_docx_path)
    serpapi_key = read_serpapi_key(serpapi_odt_path)
    
    # Set environment variables if keys were found
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if serpapi_key:
        os.environ["SERPAPI_KEY"] = serpapi_key
        
    return gemini_key, serpapi_key
