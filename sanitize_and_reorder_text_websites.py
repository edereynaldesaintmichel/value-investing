import os
import re
from pathlib import Path
import subprocess
import tempfile
from typing import Tuple

def count_slashes(url):
    """Count slashes in URL, excluding trailing slash"""
    url = url.rstrip('/')
    # Remove protocol if present
    url = re.sub(r'^https?://', '', url)
    return url.count('/')

def is_bullshit_page(url):
    """Check if URL contains patterns indicating non-content pages"""
    bullshit_patterns = [
        'privacy', 'terms', 'conditions', 'certificate', 'bylaws', 'legal', 'cookie',
        'disclaimer', 'copyright',
        'sitemap', 'accessibility', 'gdpr', 'compliance',
        'unsubscribe', 'preferences', 'settings', 'login', 'signup',
        'register', 'forgot-password', 'reset-password', '404',
        'error', 'search', 'tag/', 'wp-admin', 'wp-content'
    ]
    
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in bullshit_patterns)


def process_document(input_path, output_path):
    """Process a single document: filter and reorder pages"""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by page separator
    pages = content.split('# http')
    
    # Parse pages into structured format
    parsed_pages = []
    for page in pages:
        if not page.strip():
            continue
        
        lines = page.strip().split('\n')
        if not lines:
            continue
            
        url = 'http' + lines[0].strip()
        path = url.split('//')[1].strip('/')
        if 'cibt' in url:
            debug = True
        # Skip bullshit pages
        if is_bullshit_page(url):
            continue
            
        page_content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        page_content = page_content
        
        parsed_pages.append({
            'path': path,
            'url': url,
            'content': page_content,
            'slash_count': count_slashes(url),
            'url_length': len(url.rstrip('/'))
        })
    
    # Sort pages by slash count, then by URL length
    sorted_pages = sorted(parsed_pages, key=lambda x: (x['slash_count'], x['path']))
    pages_dict = {page['path']: page['content'] for page in sorted_pages}
    
    # Reconstruct document
    output_lines = []
    for index, page in enumerate(sorted_pages):
        output_lines.append(f"########## {page['url']}")
        if page['content']:
            reference_to_git_diff_against = None
            path = page['path']
            content = page['content']
            for _ in range(1):
                path = path[:path.rfind('/')]
                reference_to_git_diff_against = pages_dict.get(path)
                if '/' not in path or reference_to_git_diff_against is not None:
                    break
                
            if index > 0 and reference_to_git_diff_against is None: # Fallback for these shitty websites whom all pages have an .html or .aspx extension and thus have no real path tree.
                reference_to_git_diff_against = sorted_pages[index-1]['content']
            if reference_to_git_diff_against is not None:
                content = strip_header_footer(reference_to_git_diff_against, content)
            output_lines.append(content)
    

    compression_ratio = len('\n'.join([page['content'] for page in sorted_pages])) / len('\n'.join(output_lines))
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    return len(sorted_pages)


def strip_header_footer(reference_text: str, target_text: str) -> str:
    """
    Use git diff to find and return only the unique content in target_text.
    
    Args:
        reference_text: The reference document text
        target_text: The document to be cleaned
    
    Returns:
        The unique content from target_text (headers/footers removed)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_file = os.path.join(tmpdir, 'ref.txt')
        target_file = os.path.join(tmpdir, 'target.txt')
        
        # Write texts to temporary files
        with open(ref_file, 'w') as f:
            f.write(reference_text)
        with open(target_file, 'w') as f:
            f.write(target_text)
        
        # Run git diff
        result = subprocess.run(
            ['git', 'diff', '--no-index', '--no-prefix', ref_file, target_file],
            capture_output=True,
            text=True
        )
        
        # Extract lines that are unique to target (lines starting with '+')
        unique_lines = []
        for line in result.stdout.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                unique_lines.append(line[1:])  # Remove the '+' prefix
        
        return '\n'.join(unique_lines)

def process_all_documents(input_dir, output_dir):
    """Process all .txt files in the input directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for txt_file in input_path.glob('*.md'):
        input_file = str(txt_file)
        output_file = str(output_path / txt_file.name)
        
        try:
            num_pages = process_document(input_file, output_file)
            print(f"Processed {txt_file.name}: {num_pages} pages kept")
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")

# Run the script
if __name__ == "__main__":
    input_directory = "/Users/eloireynal/Documents/My projects/crawl_data/txt/"
    output_directory = "/Users/eloireynal/Documents/My projects/crawl_data/sanitized_txt/"
    
    process_all_documents(input_directory, output_directory)