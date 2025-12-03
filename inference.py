import torch
import os
from PIL import Image
import cairosvg
import io
import tempfile
import argparse
import gc
import yaml
import glob
import numpy as np
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from decoder import SketchDecoder
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tokenizer import SVGTokenizer

# Load config
CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Use a default device, but we'll get the actual device from the model later
default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Global Models
tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None
current_model_size = None

# Constants from config
SYSTEM_PROMPT = """You are an expert SVG code generator. 
Generate precise, valid SVG path commands that accurately represent the described scene or object.
Focus on capturing key shapes, spatial relationships, and visual composition."""

SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
AVAILABLE_MODEL_SIZES = list(config.get('models', {}).keys())
DEFAULT_MODEL_SIZE = config.get('default_model_size', '8B')


def get_config_value(model_size, *keys):
    """Get config value with model-specific override support."""
    model_cfg = config.get('models', {}).get(model_size, {})
    value = model_cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    if value is None:
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    
    return value


# Image processing settings from config
image_config = config.get('image', {})
TARGET_IMAGE_SIZE = image_config.get('target_size', 448)
RENDER_SIZE = image_config.get('render_size', 512)
BACKGROUND_THRESHOLD = image_config.get('background_threshold', 240)
EMPTY_THRESHOLD_ILLUSTRATION = image_config.get('empty_threshold_illustration', 250)
EMPTY_THRESHOLD_ICON = image_config.get('empty_threshold_icon', 252)
EDGE_SAMPLE_RATIO = image_config.get('edge_sample_ratio', 0.1)
COLOR_SIMILARITY_THRESHOLD = image_config.get('color_similarity_threshold', 30)
MIN_EDGE_SAMPLES = image_config.get('min_edge_samples', 10)

# Color settings from config
colors_config = config.get('colors', {})
BLACK_COLOR_TOKEN = colors_config.get('black_color_token', 
                                       colors_config.get('color_token_start', 40010) + 2)

# Model settings from config
model_config = config.get('model', {})
BOS_TOKEN_ID = model_config.get('bos_token_id', 196998)
EOS_TOKEN_ID = model_config.get('eos_token_id', 196999)
PAD_TOKEN_ID = model_config.get('pad_token_id', 151643)
MAX_LENGTH = model_config.get('max_length', 1024)
MIN_MAX_LENGTH = 256
MAX_MAX_LENGTH = 2048

# Task configurations with defaults from config
task_config = config.get('task_configs', {})

TASK_CONFIGS = {
    "text-to-svg-icon": task_config.get('text_to_svg_icon', {
        "default_temperature": 0.5,
        "default_top_p": 0.88,
        "default_top_k": 50,
        "default_repetition_penalty": 1.05,
    }),
    "text-to-svg-illustration": task_config.get('text_to_svg_illustration', {
        "default_temperature": 0.6,
        "default_top_p": 0.90,
        "default_top_k": 60,
        "default_repetition_penalty": 1.03,
    }),
    "image-to-svg": task_config.get('image_to_svg', {
        "default_temperature": 0.3,
        "default_top_p": 0.90,
        "default_top_k": 50,
        "default_repetition_penalty": 1.05,
    })
}

# Generation parameters from config
gen_config = config.get('generation', {})
DEFAULT_NUM_CANDIDATES = gen_config.get('default_num_candidates', 4)
MAX_NUM_CANDIDATES = gen_config.get('max_num_candidates', 8)
EXTRA_CANDIDATES_BUFFER = gen_config.get('extra_candidates_buffer', 4)

# Validation settings from config
validation_config = config.get('validation', {})
MIN_SVG_LENGTH = validation_config.get('min_svg_length', 20)


def get_model_input_device():
    """
    Get the device where model inputs should be placed.
    This handles multi-GPU scenarios where the model is distributed across devices.
    """
    global sketch_decoder
    
    if sketch_decoder is None:
        return default_device
    
    try:
        # Get the transformer model
        model = sketch_decoder.transformer
        
        # Try to get device from the embedding layer (this is where input_ids will be processed)
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_device = next(model.model.embed_tokens.parameters()).device
            return embed_device
        elif hasattr(model, 'embed_tokens'):
            embed_device = next(model.embed_tokens.parameters()).device
            return embed_device
        
        # Alternative: try to get from the first parameter
        first_param = next(model.parameters())
        return first_param.device
        
    except (StopIteration, AttributeError) as e:
        print(f"Warning: Could not determine model device, using default: {default_device}")
        return default_device


def get_model_devices_info():
    """Get information about which devices the model is using (for debugging)."""
    global sketch_decoder
    
    if sketch_decoder is None:
        return "Model not loaded"
    
    devices = set()
    try:
        model = sketch_decoder.transformer
        for name, param in model.named_parameters():
            devices.add(str(param.device))
    except Exception as e:
        return f"Error getting device info: {e}"
    
    return f"Model distributed across: {sorted(devices)}"


def parse_args():
    parser = argparse.ArgumentParser(description='OmniSVG Inference Script')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True, choices=['text-to-svg', 'image-to-svg'],
                        help='Task type: text-to-svg or image-to-svg')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (.txt for text-to-svg) or directory (for image-to-svg)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for generated SVGs')
    
    # Model settings
    parser.add_argument('--model-size', type=str, default=DEFAULT_MODEL_SIZE,
                        choices=AVAILABLE_MODEL_SIZES,
                        help=f'Model size to use (default: {DEFAULT_MODEL_SIZE})')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Local path or HuggingFace repo ID for Qwen model (overrides config)')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='Local path or HuggingFace repo ID for OmniSVG weights (overrides config)')
    
    # Generation parameters
    parser.add_argument('--num-candidates', type=int, default=DEFAULT_NUM_CANDIDATES,
                        help=f'Number of candidates to generate (default: {DEFAULT_NUM_CANDIDATES})')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: task-specific)')
    parser.add_argument('--top-p', type=float, default=None,
                        help='Top-p sampling (default: task-specific)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-k sampling (default: task-specific)')
    parser.add_argument('--repetition-penalty', type=float, default=None,
                        help='Repetition penalty (default: task-specific)')
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH,
                        help=f'Max token length (default: {MAX_LENGTH})')
    
    # Image-specific options
    parser.add_argument('--replace-background', action='store_true', default=True,
                        help='Replace non-white background in images (default: True)')
    parser.add_argument('--no-replace-background', action='store_false', dest='replace_background',
                        help='Do not replace background')
    
    # Output options
    parser.add_argument('--save-png', action='store_true', default=False,
                        help='Also save rendered PNG images')
    parser.add_argument('--save-all-candidates', action='store_true', default=False,
                        help='Save all candidates (default: save only the best one)')
    
    # Debug
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose output')
    
    return parser.parse_args()


def download_model_weights(repo_id: str, filename: str = "pytorch_model.bin") -> str:
    """Download model weights from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
        )
        print(f"Successfully downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading from {repo_id}: {e}")
        raise


def is_local_path(path: str) -> bool:
    """Check if a path is a local filesystem path or a HuggingFace repo ID."""
    if os.path.exists(path):
        return True
    if path.startswith('/') or path.startswith('./') or path.startswith('../'):
        return True
    if os.path.sep in path and os.path.exists(os.path.dirname(path)):
        return True
    if len(path) > 1 and path[1] == ':':
        return True
    return False


def load_models(model_size: str, weight_path: str = None, model_path: str = None):
    """Load all models for a specific model size."""
    global tokenizer, processor, sketch_decoder, svg_tokenizer, current_model_size
    
    if weight_path is None:
        weight_path = get_config_value(model_size, 'huggingface', 'omnisvg_model')
    if model_path is None:
        model_path = get_config_value(model_size, 'huggingface', 'qwen_model')
    
    print(f"\n{'='*60}")
    print(f"Loading {model_size} Model")
    print(f"{'='*60}")
    print(f"Qwen model: {model_path}")
    print(f"OmniSVG weights: {weight_path}")
    print(f"Precision: {DTYPE}")
    
    # Load Qwen tokenizer and processor
    print("\n[1/3] Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="left",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        padding_side="left",
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    print("Tokenizer and processor loaded successfully!")

    # Initialize sketch decoder with model_size
    print("\n[2/3] Initializing SketchDecoder...")
    sketch_decoder = SketchDecoder(
        config_path=CONFIG_PATH,
        model_path=model_path,
        model_size=model_size,
        pix_len=MAX_MAX_LENGTH,
        text_len=config.get('text', {}).get('max_length', 200),
        torch_dtype=DTYPE
    )
    
    # Load OmniSVG weights
    print("\n[3/3] Loading OmniSVG weights...")
    
    if is_local_path(weight_path):
        bin_path = os.path.join(weight_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            if os.path.exists(weight_path) and weight_path.endswith('.bin'):
                bin_path = weight_path
            else:
                raise FileNotFoundError(
                    f"Could not find pytorch_model.bin at {weight_path}. "
                    f"Please provide a valid local path or HuggingFace repo ID."
                )
        print(f"Loading weights from local path: {bin_path}")
    else:
        print(f"Downloading weights from HuggingFace: {weight_path}")
        bin_path = download_model_weights(weight_path, "pytorch_model.bin")
    
    state_dict = torch.load(bin_path, map_location='cpu')
    sketch_decoder.load_state_dict(state_dict)
    print("OmniSVG weights loaded successfully!")
    
    # Note: We don't move to a specific device here if using device_map="auto"
    # The model is already distributed by accelerate
    sketch_decoder = sketch_decoder.eval()
    
    # Initialize SVG tokenizer with model_size
    svg_tokenizer = SVGTokenizer(CONFIG_PATH, model_size=model_size)
    
    current_model_size = model_size
    
    # Print device distribution info
    print(f"\n{get_model_devices_info()}")
    print(f"Input device will be: {get_model_input_device()}")
    
    print("\n" + "="*60)
    print(f"All {model_size} models loaded successfully!")
    print("="*60 + "\n")


def detect_text_subtype(text_prompt):
    """Auto-detect text prompt subtype"""
    text_lower = text_prompt.lower()
    
    icon_keywords = ['icon', 'logo', 'symbol', 'badge', 'button', 'emoji', 'glyph', 'simple', 
                     'arrow', 'triangle', 'circle', 'square', 'heart', 'star', 'checkmark']
    if any(kw in text_lower for kw in icon_keywords):
        return "icon"
    
    illustration_keywords = [
        'illustration', 'scene', 'person', 'people', 'character', 'man', 'woman', 'boy', 'girl',
        'avatar', 'portrait', 'face', 'head', 'body',
        'cat', 'dog', 'bird', 'animal', 'pet', 'fox', 'rabbit',
        'sitting', 'standing', 'walking', 'running', 'sleeping', 'holding', 'playing',
        'house', 'building', 'tree', 'garden', 'landscape', 'mountain', 'forest', 'city',
        'ocean', 'beach', 'sunset', 'sunrise', 'sky'
    ]
    
    match_count = sum(1 for kw in illustration_keywords if kw in text_lower)
    if match_count >= 1 or len(text_prompt) > 50:
        return "illustration"
    
    return "icon"


def detect_and_replace_background(image, threshold=None, edge_sample_ratio=None):
    """Detect if image has non-white background and optionally replace it."""
    if threshold is None:
        threshold = BACKGROUND_THRESHOLD
    if edge_sample_ratio is None:
        edge_sample_ratio = EDGE_SAMPLE_RATIO
    
    img_array = np.array(image)
    
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, image)
        return composite.convert('RGB'), True
    
    h, w = img_array.shape[:2]
    edge_pixels = []
    
    sample_count = max(MIN_EDGE_SAMPLES, int(min(h, w) * edge_sample_ratio))
    
    for i in range(0, w, max(1, w // sample_count)):
        edge_pixels.append(img_array[0, i])
        edge_pixels.append(img_array[h-1, i])
    
    for i in range(0, h, max(1, h // sample_count)):
        edge_pixels.append(img_array[i, 0])
        edge_pixels.append(img_array[i, w-1])
    
    edge_pixels = np.array(edge_pixels)
    
    if len(edge_pixels) > 0:
        mean_edge = edge_pixels.mean(axis=0)
        if np.all(mean_edge > threshold):
            return image, False
    
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        if img_array.shape[2] == 4:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = np.mean(img_array, axis=2)
        
        edge_colors = []
        for i in range(w):
            edge_colors.append(tuple(img_array[0, i, :3]))
            edge_colors.append(tuple(img_array[h-1, i, :3]))
        for i in range(h):
            edge_colors.append(tuple(img_array[i, 0, :3]))
            edge_colors.append(tuple(img_array[i, w-1, :3]))
        
        from collections import Counter
        color_counts = Counter(edge_colors)
        bg_color = color_counts.most_common(1)[0][0]
        
        color_diff = np.sqrt(np.sum((img_array[:, :, :3].astype(float) - np.array(bg_color)) ** 2, axis=2))
        bg_mask = color_diff < COLOR_SIMILARITY_THRESHOLD
        
        result = img_array.copy()
        if result.shape[2] == 4:
            result[bg_mask] = [255, 255, 255, 255]
        else:
            result[bg_mask] = [255, 255, 255]
        
        return Image.fromarray(result).convert('RGB'), True
    
    return image, False


def preprocess_image_for_svg(image, replace_background=True, target_size=None):
    """Preprocess image for SVG generation."""
    if target_size is None:
        target_size = TARGET_IMAGE_SIZE
    
    if isinstance(image, str):
        raw_img = Image.open(image)
    else:
        raw_img = image
    
    was_modified = False
    
    if raw_img.mode == 'RGBA':
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode == 'LA' or raw_img.mode == 'PA':
        raw_img = raw_img.convert('RGBA')
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode != 'RGB':
        img_with_bg = raw_img.convert('RGB')
    else:
        img_with_bg = raw_img
    
    if replace_background:
        img_with_bg, bg_replaced = detect_and_replace_background(img_with_bg)
        was_modified = was_modified or bg_replaced
    
    img_resized = img_with_bg.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img_resized, was_modified


def prepare_inputs(task_type, content):
    """Prepare model inputs"""
    if task_type == "text-to-svg":
        prompt_text = str(content).strip()
        
        instruction = f"""Generate an SVG illustration for: {prompt_text}
Requirements:
- Create complete SVG path commands
- Include proper coordinates and colors
- Maintain visual clarity and composition"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_input], padding=True, truncation=True, return_tensors="pt")
        
    else:  # image-to-svg
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                {"type": "image", "image": content},
            ]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, padding=True, truncation=True, return_tensors="pt")

    return inputs


def render_svg_to_image(svg_str, size=None):
    """Render SVG to high-quality PIL Image"""
    if size is None:
        size = RENDER_SIZE
    
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])
        return bg
    except Exception as e:
        print(f"Render error: {e}")
        return None


def is_valid_candidate(svg_str, img, subtype="illustration"):
    """Check candidate validity"""
    if not svg_str or len(svg_str) < MIN_SVG_LENGTH:
        return False, "too_short"
    
    if '<svg' not in svg_str:
        return False, "no_svg_tag"
    
    if img is None:
        return False, "render_failed"
    
    img_array = np.array(img)
    mean_val = img_array.mean()
    
    threshold = EMPTY_THRESHOLD_ILLUSTRATION if subtype == "illustration" else EMPTY_THRESHOLD_ICON
    
    if mean_val > threshold:
        return False, "empty_image"
    
    return True, "ok"


def generate_candidates(inputs, task_type, subtype, temperature, top_p, top_k, repetition_penalty, 
                       max_length, num_samples, verbose=False):
    """Generate candidate SVGs with full parameter control"""
    
    # Get the correct device from the model's embedding layer
    input_device = get_model_input_device()
    
    if verbose:
        print(f"  Using input device: {input_device}")
    
    input_ids = inputs['input_ids'].to(input_device)
    attention_mask = inputs['attention_mask'].to(input_device)
    
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if 'pixel_values' in inputs:
        model_inputs["pixel_values"] = inputs['pixel_values'].to(input_device, dtype=DTYPE)
    
    if 'image_grid_thw' in inputs:
        model_inputs["image_grid_thw"] = inputs['image_grid_thw'].to(input_device)
    
    all_candidates = []
    
    gen_cfg = {
        'do_sample': True,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': int(top_k),
        'repetition_penalty': repetition_penalty,
        'early_stopping': True,
        'no_repeat_ngram_size': 0,
        'eos_token_id': EOS_TOKEN_ID,
        'pad_token_id': PAD_TOKEN_ID,
        'bos_token_id': BOS_TOKEN_ID,
    }
    
    actual_samples = num_samples + EXTRA_CANDIDATES_BUFFER
    
    try:
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                **model_inputs,
                max_new_tokens=max_length,
                num_return_sequences=actual_samples,
                use_cache=True,
                **gen_cfg
            )
            
            input_len = input_ids.shape[1]
            generated_ids_batch = results[:, input_len:]
        
        for i in range(min(actual_samples, generated_ids_batch.shape[0])):
            try:
                current_ids = generated_ids_batch[i:i+1]
                
                # Move to CPU for post-processing to avoid device issues
                current_ids_cpu = current_ids.cpu()
                
                fake_wrapper = torch.cat([
                    torch.full((1, 1), BOS_TOKEN_ID, device='cpu'),
                    current_ids_cpu,
                    torch.full((1, 1), EOS_TOKEN_ID, device='cpu')
                ], dim=1)

                generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
                if len(generated_xy) == 0:
                    continue

                svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
                if not svg_tensors or not svg_tensors[0]:
                    continue

                num_paths = len(svg_tensors[0])
                while len(color_tensors) < num_paths:
                    color_tensors.append(BLACK_COLOR_TOKEN)
                
                svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
                svg_str = svg.to_str()
                
                if 'width=' not in svg_str:
                    svg_str = svg_str.replace('<svg', f'<svg width="{TARGET_IMAGE_SIZE}" height="{TARGET_IMAGE_SIZE}"', 1)
                
                png_image = render_svg_to_image(svg_str, size=RENDER_SIZE)
                
                is_valid, reason = is_valid_candidate(svg_str, png_image, subtype)
                if is_valid:
                    all_candidates.append({
                        'svg': svg_str,
                        'img': png_image,
                        'path_count': num_paths,
                        'index': len(all_candidates) + 1
                    })
                    
                    if verbose:
                        print(f"  Found valid candidate {len(all_candidates)} with {num_paths} paths")
                    
                    if len(all_candidates) >= num_samples:
                        break
                elif verbose:
                    print(f"  Candidate {i} invalid: {reason}")
                        
            except Exception as e:
                if verbose:
                    print(f"  Candidate {i} error: {e}")
                continue

    except Exception as e:
        print(f"Generation Error: {e}")
        import traceback
        traceback.print_exc()
    
    return all_candidates


def save_results(candidates, output_dir, base_name, save_png=False, save_all=False):
    """Save generated SVG(s) and optionally PNG(s)"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    if not candidates:
        return saved_files
    
    if save_all:
        for i, cand in enumerate(candidates):
            svg_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(cand['svg'])
            saved_files.append(svg_path)
            
            if save_png and cand['img'] is not None:
                png_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.png")
                cand['img'].save(png_path)
                saved_files.append(png_path)
    else:
        # Save only the best (first valid) candidate
        best = candidates[0]
        svg_path = os.path.join(output_dir, f"{base_name}.svg")
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(best['svg'])
        saved_files.append(svg_path)
        
        if save_png and best['img'] is not None:
            png_path = os.path.join(output_dir, f"{base_name}.png")
            best['img'].save(png_path)
            saved_files.append(png_path)
    
    return saved_files


def process_text_to_svg(args):
    """Process text-to-svg task"""
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Read prompts from text file
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    prompts = [line.strip() for line in lines if line.strip()]
    
    if not prompts:
        print("Error: No prompts found in input file")
        return
    
    print(f"\nFound {len(prompts)} prompts to process")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each prompt
    total_success = 0
    total_failed = 0
    
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Processing: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        start_time = time.time()
        
        # Detect subtype
        subtype = detect_text_subtype(prompt)
        task_key = f"text-to-svg-{subtype}"
        
        # Get default parameters based on task
        temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.5)
        top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
        top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
        rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
        
        if args.verbose:
            print(f"  Subtype: {subtype}")
            print(f"  Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
        
        # Prepare inputs
        inputs = prepare_inputs("text-to-svg", prompt)
        
        # Generate candidates
        candidates = generate_candidates(
            inputs, "text-to-svg", subtype,
            temperature, top_p, top_k, rep_penalty,
            args.max_length, args.num_candidates,
            verbose=args.verbose
        )
        
        elapsed = time.time() - start_time
        
        if candidates:
            # Create safe filename from prompt
            safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt[:50]).strip()
            safe_name = f"{idx+1:04d}_{safe_name}"
            
            saved = save_results(candidates, args.output, safe_name, 
                               save_png=args.save_png, save_all=args.save_all_candidates)
            
            print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
            print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
            total_success += 1
        else:
            print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
            total_failed += 1
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Text-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(prompts)}")
    print(f"  Failed: {total_failed}/{len(prompts)}")
    print(f"  Output: {args.output}")
    print("="*60)


def process_image_to_svg(args):
    """Process image-to-svg task"""
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        return
    
    # Find all image files
    if os.path.isfile(input_path):
        image_files = [input_path]
    else:
        image_files = []
        for ext in SUPPORTED_FORMATS:
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"Error: No image files found in {input_path}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get default parameters
    task_key = "image-to-svg"
    temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.3)
    top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
    top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
    
    if args.verbose:
        print(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
    
    # Process each image
    total_success = 0
    total_failed = 0
    
    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_name}")
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(img_path)
            img_processed, was_modified = preprocess_image_for_svg(
                image, 
                replace_background=args.replace_background,
                target_size=TARGET_IMAGE_SIZE
            )
            
            if args.verbose and was_modified:
                print("  Background processed/replaced")
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img_processed.save(tmp_file.name, format='PNG', quality=100)
                tmp_path = tmp_file.name
            
            try:
                # Prepare inputs
                inputs = prepare_inputs("image-to-svg", tmp_path)
                
                # Generate candidates
                candidates = generate_candidates(
                    inputs, "image-to-svg", "image",
                    temperature, top_p, top_k, rep_penalty,
                    args.max_length, args.num_candidates,
                    verbose=args.verbose
                )
                
                elapsed = time.time() - start_time
                
                if candidates:
                    # Use original filename (without extension) as base name
                    base_name = os.path.splitext(img_name)[0]
                    
                    saved = save_results(candidates, args.output, base_name, 
                                       save_png=args.save_png, save_all=args.save_all_candidates)
                    
                    print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
                    print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
                    total_success += 1
                else:
                    print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
                    total_failed += 1
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            total_failed += 1
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Image-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(image_files)}")
    print(f"  Failed: {total_failed}/{len(image_files)}")
    print(f"  Output: {args.output}")
    print("="*60)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    print("="*60)
    print("OmniSVG Inference Script")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Model Size: {args.model_size}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Default Device: {default_device}")
    print(f"Precision: {DTYPE}")
    print(f"Num Candidates: {args.num_candidates}")
    print(f"Max Length: {args.max_length}")
    print("="*60)
    
    # Load models
    load_models(args.model_size, args.weight_path, args.model_path)
    
    # Process based on task type
    if args.task == "text-to-svg":
        process_text_to_svg(args)
    else:  # image-to-svg
        process_image_to_svg(args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()