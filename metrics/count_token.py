import torch
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

def calculate_average_svg_token_length(folder_path, tokenizer_path="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Calculate average token length for all SVG files in a folder"""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # List all SVG files in the folder
    folder = Path(folder_path)
    svg_files = list(folder.glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {folder_path}")
        return 0
    
    # Calculate token length for each SVG file
    total_tokens = 0
    valid_files = 0
    
    for svg_file in tqdm(svg_files, desc="Processing SVG files"):
        try:
            svg_content = svg_file.read_text(encoding='utf-8')
            tokens = tokenizer.encode(svg_content)
            total_tokens += len(tokens)
            valid_files += 1
        except Exception as e:
            print(f"Error processing file {svg_file}: {e}")
    
    # Calculate average token length
    if valid_files > 0:
        average_tokens = total_tokens / valid_files
    else:
        average_tokens = 0
    
    return average_tokens

if __name__ == "__main__":
    # Specify the path to your folder containing SVG files
    folder_path = "/path/to/generated_svg"
    
    average_token_length = calculate_average_svg_token_length(folder_path)
    print(f"Average token length of SVG files in {folder_path}: {average_token_length:.2f}")
    

  