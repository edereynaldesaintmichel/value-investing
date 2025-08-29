import subprocess
import tempfile
import os
from typing import Tuple

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


# Example usage
if __name__ == "__main__":
    with open("nutrixeal_1.txt", 'r+') as file:
        reference = file.read()

    with open("nutrixeal_2.txt", 'r+') as file:
        target = file.read()
    
    cleaned = strip_header_footer(reference, target)
    print(cleaned)