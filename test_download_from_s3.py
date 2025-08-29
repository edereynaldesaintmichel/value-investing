import boto3
import gzip
from io import BytesIO
import warcio
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from bs4 import BeautifulSoup, Comment, NavigableString
import pdfplumber
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import logging
from datetime import datetime
import html2text
import re

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s'
)

class HTMLSanitizer:
    """Python translation of the custom JS HTML sanitization logic"""
    
    TAGS_TO_REMOVE = ['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'svg', 'code', 'noscript', 'i', 'img']
    QUERY_SELECTORS_TO_REMOVE = ["#ultimate_extension_div", '#toggle_ultimext']
    TO_REMOVE_IF_EMPTY = ['div', 'span', 'li', 'p', 'td', 'th', 'tr', 'table', 'a', 'button', 'input']
    TAGS_TO_KEEP = {'table', 'tr', 'th', 'td', 'thead', 'li', 'p'}
    
    @staticmethod
    def clean_attributes(element):
        """Remove all attributes from an element"""
        if hasattr(element, 'attrs'):
            element.attrs = {}
    
    @staticmethod
    def normalize_whitespace(text):
        """Normalize whitespace in text"""
        if not text:
            return text
            
        old_text = None
        while old_text != text:
            old_text = text
            # Replace 2+ line breaks with single line break
            text = re.sub(r'\n\s*\n', '\n', text)
            # Replace 2+ spaces with single space
            text = re.sub(r'[ \t]+', ' ', text)
        
        return text
    
    @classmethod
    def only_child_policy(cls, element):
        """Apply the only child policy from the JS version"""
        if not element or isinstance(element, NavigableString):
            return element
            
        # Clean attributes
        cls.clean_attributes(element)
        
        # Remove empty text nodes
        for child in list(element.children):
            if isinstance(child, NavigableString) and not child.strip():
                child.extract()
        
        children = list(element.children)
        
        # If element has only one child
        if len(children) == 1:
            child = children[0]
            
            # If we should keep this tag, process child and return element
            if element.name and element.name.lower() in cls.TAGS_TO_KEEP:
                if not isinstance(child, NavigableString):
                    new_child = cls.only_child_policy(child)
                    if new_child != child and new_child:
                        # Replace the child with the processed version
                        child.replace_with(new_child)
                return element
            
            # If child is text, add space and return a new NavigableString
            if isinstance(child, NavigableString):
                # Create a new NavigableString with space prepended
                new_string = NavigableString(" " + str(child))
                return new_string
            
            # Otherwise, return the processed child
            return cls.only_child_policy(child)
        
        # Process all children
        for child in list(element.children):
            if not isinstance(child, NavigableString):
                new_child = cls.only_child_policy(child)
                if new_child != child and new_child:
                    try:
                        child.replace_with(new_child)
                    except:
                        # If replacement fails, just continue
                        pass
        
        return element
    
    @classmethod
    def clean_html(cls, html_string):
        """Main cleaning function"""
        if not html_string:
            return ""
            
        try:
            soup = BeautifulSoup(html_string, 'html.parser')
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove specific tags
            for tag in cls.TAGS_TO_REMOVE:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove elements by query selector
            for selector in cls.QUERY_SELECTORS_TO_REMOVE:
                for element in soup.select(selector):
                    element.decompose()
            
            # Apply only child policy to body
            body = soup.body if soup.body else soup
            
            # Process the body
            if body and body.name:
                new_body = cls.only_child_policy(body)
                if new_body != body and isinstance(new_body, NavigableString):
                    # If body was replaced with text, wrap it in a new body tag
                    new_body_tag = soup.new_tag('body')
                    new_body_tag.append(new_body)
                    body = new_body_tag
                elif new_body != body:
                    body = new_body
            
            # Remove empty elements
            if body:
                for tag_name in cls.TO_REMOVE_IF_EMPTY:
                    elements = body.find_all(tag_name)
                    # Sort by content length
                    elements.sort(key=lambda x: len(x.get_text(strip=True)))
                    for element in elements:
                        if not element.get_text(strip=True):
                            element.decompose()
                
                # Add context_element class to first child
                if body.contents:
                    first_child = None
                    for child in body.contents:
                        if not isinstance(child, NavigableString):
                            first_child = child
                            break
                    
                    if first_child and hasattr(first_child, 'attrs'):
                        if 'class' not in first_child.attrs:
                            first_child.attrs['class'] = []
                        if 'context_element' not in first_child.attrs['class']:
                            first_child.attrs['class'].append('context_element')
            
            result = str(body) if body else ""
            return cls.normalize_whitespace(result)
            
        except Exception as e:
            logging.error(f"Error in clean_html: {str(e)}")
            # Return original HTML if cleaning fails
            return html_string
    
    @classmethod
    def clean_html_iterative(cls, html_string):
        """Apply cleaning iteratively until no more changes"""
        if not html_string:
            return ""
            
        last_length = float('inf')
        html = html_string
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while len(html) < last_length and iterations < max_iterations:
            last_length = len(html)
            html = cls.clean_html(html)
            iterations += 1
        
        return html


def html_to_markdown(html_string):
    """Convert HTML to markdown using html2text"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # Don't wrap lines
    h.unicode_snob = True
    h.skip_internal_links = False
    
    try:
        markdown = h.handle(html_string)
        # Clean up excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        return markdown.strip()
    except Exception as e:
        logging.error(f"Error converting HTML to markdown: {str(e)}")
        return None


def create_s3_client():
    """Create S3 client for each process"""
    return boto3.client('s3')


def fetch_page_content(row, s3_client):
    """
    Fetch the actual page content from Common Crawl S3 bucket
    """
    bucket = 'commoncrawl'
    key = row['warc_filename']
    
    record_length = row['warc_record_length']
    buffer_size = 4096 
    
    # Calculate the byte range to fetch
    offset = row['warc_record_offset']
    end_offset = offset + record_length + buffer_size - 1
    byte_range = f"bytes={offset}-{end_offset}"
    
    try:
        # Fetch only the specific record using byte range
        response = s3_client.get_object(
            Bucket=bucket, 
            Key=key, 
            Range=byte_range,
            RequestPayer='requester'
        )
        
        # The data is gzipped
        raw_data = response['Body'].read()
        
        # Parse the WARC record
        stream = BytesIO(raw_data)
        for record in warcio.ArchiveIterator(stream):
            if record.rec_type == 'response':
                # Get the HTTP response
                content = record.content_stream().read()
                
                # Extract content type from headers
                headers = record.http_headers
                content_type = headers.get_header('Content-Type', '')
                
                return {
                    'content': content,
                    'content_type': content_type,
                    'url': row['url']
                }
    
    except Exception as e:
        logging.error(f"Error fetching {row['url']}: {str(e)}")
        return None


def extract_text_from_content(content_data):
    """Extract text from content data"""
    if not content_data or not content_data.get('content'):
        return None

    content = content_data['content']
    content_type = content_data.get('content_type', '').lower()
    url = content_data.get('url', '')

    # Handle PDF content
    if 'application/pdf' in content_type or url.endswith('.pdf'):
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = "".join(page.extract_text() or "" for page in pdf.pages)
            return full_text.strip() if full_text.strip() else None
        except Exception as e:
            logging.error(f"Error extracting PDF from {url}: {str(e)}")
            return None

    # Handle HTML content
    is_html_like = ('text/html' in content_type or
                    url.endswith(('.html', '.htm', '.asp', '.aspx')) or
                    'text/html' in content_type.lower())

    if is_html_like:
        try:
            # Decode HTML
            try:
                html_text = content.decode('utf-8')
            except UnicodeDecodeError:
                html_text = content.decode('latin-1', errors='ignore')
            
            # Sanitize HTML using custom sanitizer
            sanitized_html = HTMLSanitizer.clean_html_iterative(html_text)
            
            # Convert to markdown
            markdown_text = html_to_markdown(sanitized_html)
            
            return markdown_text

        except Exception as e:
            logging.error(f"Error processing HTML from {url}: {str(e)}")
            return None

    return None


def save_host_content(host_content, host_name, base_path='/Users/eloireynal/Documents/My projects/crawl_data/txt'):
    """Save concatenated content for each host as markdown"""
    os.makedirs(base_path, exist_ok=True)
    safe_host = host_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    filepath = os.path.join(base_path, f"{safe_host}.md")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for url, text in host_content.items():
                f.write(f"# {url}\n\n")
                f.write(text)
                f.write("\n\n---\n\n")  # Markdown separator between pages
        
        logging.info(f"Saved content for {host_name} to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving content for {host_name}: {str(e)}")
        return False


def process_single_url(row, s3_client):
    """Process a single URL and return the result"""
    try:
        content_data = fetch_page_content(row, s3_client)
        text = extract_text_from_content(content_data)
        
        return row['url'], text, len(text) if text else 0
    except Exception as e:
        logging.error(f"Error processing {row['url']}: {str(e)}")
        return row['url'], None, 0


def process_host_group(host_data):
    """Process all URLs for a single host"""
    host_name, group_df = host_data
    
    logging.info(f"Starting processing for host: {host_name} ({len(group_df)} URLs)")
    
    # Create S3 client for this process
    s3_client = create_s3_client()
    
    host_content = {}
    successful_extractions = 0
    
    # Process URLs for this host using thread pool for I/O operations
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_url, row, s3_client): row 
            for _, row in group_df.iterrows()
        }
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            url, text, text_length = future.result()
            
            if text:
                host_content[url] = text
                successful_extractions += 1
                logging.debug(f"Successfully extracted {text_length} characters from {url}")
    
    # Save the content for this host
    if host_content:
        save_host_content(host_content, host_name)
    
    logging.info(f"Completed host: {host_name} - {successful_extractions}/{len(group_df)} URLs extracted")
    
    return host_name, successful_extractions, len(group_df)


def process_all_records_parallel(start_index=0, max_workers=None):
    """Process all records in parallel, grouped by host"""
    
    # Load the dataframe
    logging.info("Loading dataframe...")
    df_filtered = pd.read_parquet("commoncrawl_sanitized.parquet")
    
    if start_index > 0:
        df_filtered = df_filtered.iloc[start_index:]
    
    # Group by host
    logging.info("Grouping by host...")
    grouped = df_filtered.groupby('url_host_name')
    host_groups = list(grouped)
    
    logging.info(f"Found {len(host_groups)} unique hosts to process")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 16)
    
    logging.info(f"Using {max_workers} worker processes")
    
    # Process hosts in parallel
    total_hosts = len(host_groups)
    completed_hosts = 0
    total_urls_processed = 0
    total_urls_extracted = 0
    
    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_host = {
            executor.submit(process_host_group, host_data): host_data[0] 
            for host_data in host_groups
        }
        
        # Process completed tasks
        for future in as_completed(future_to_host):
            try:
                host_name, extracted, total = future.result()
                completed_hosts += 1
                total_urls_processed += total
                total_urls_extracted += extracted
                
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed_hosts / elapsed if elapsed > 0 else 0
                eta = (total_hosts - completed_hosts) / rate if rate > 0 else 0
                
                logging.info(
                    f"Progress: {completed_hosts}/{total_hosts} hosts "
                    f"({total_urls_extracted}/{total_urls_processed} URLs extracted) "
                    f"- Rate: {rate:.2f} hosts/sec - ETA: {eta/60:.1f} minutes"
                )
                
            except Exception as e:
                host_name = future_to_host[future]
                logging.error(f"Failed to process host {host_name}: {str(e)}")
    
    elapsed_total = (datetime.now() - start_time).total_seconds()
    logging.info(
        f"\nProcessing complete!\n"
        f"Total time: {elapsed_total/60:.1f} minutes\n"
        f"Hosts processed: {completed_hosts}/{total_hosts}\n"
        f"URLs extracted: {total_urls_extracted}/{total_urls_processed}\n"
        f"Average rate: {completed_hosts/elapsed_total:.2f} hosts/sec"
    )


if __name__ == "__main__":
    # You can adjust max_workers based on your system capabilities
    # More workers = faster processing but more memory/CPU usage
    process_all_records_parallel(start_index=0, max_workers=8)