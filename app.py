import gradio as gr
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
import threading

from huggingface_hub import hf_hub_download, snapshot_download

from decoder import SketchDecoder
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tokenizer import SVGTokenizer

# Load config
CONFIG_PATH = './config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Global Models (will be loaded based on selected model size)
tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None
current_model_size = None  # Track which model is currently loaded

# Thread lock for model inference
generation_lock = threading.Lock()
model_loading_lock = threading.Lock()

# Constants from config
SYSTEM_PROMPT = """You are an expert SVG code generator. 
Generate precise, valid SVG path commands that accurately represent the described scene or object.
Focus on capturing key shapes, spatial relationships, and visual composition."""

SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
AVAILABLE_MODEL_SIZES = list(config.get('models', {}).keys())
DEFAULT_MODEL_SIZE = config.get('default_model_size', '8B')

# ============================================================
# Helper function to get config value (model-specific or shared)
# ============================================================
def get_config_value(model_size, *keys):
    """Get config value with model-specific override support."""
    # Try model-specific config first
    model_cfg = config.get('models', {}).get(model_size, {})
    value = model_cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    # Fallback to shared config if not found
    if value is None:
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    
    return value


# ============================================================
# Image processing settings from config (shared)
# ============================================================
image_config = config.get('image', {})
TARGET_IMAGE_SIZE = image_config.get('target_size', 448)
RENDER_SIZE = image_config.get('render_size', 512)
BACKGROUND_THRESHOLD = image_config.get('background_threshold', 240)
EMPTY_THRESHOLD_ILLUSTRATION = image_config.get('empty_threshold_illustration', 250)
EMPTY_THRESHOLD_ICON = image_config.get('empty_threshold_icon', 252)
EDGE_SAMPLE_RATIO = image_config.get('edge_sample_ratio', 0.1)
COLOR_SIMILARITY_THRESHOLD = image_config.get('color_similarity_threshold', 30)
MIN_EDGE_SAMPLES = image_config.get('min_edge_samples', 10)

# ============================================================
# Color settings from config (shared)
# ============================================================
colors_config = config.get('colors', {})
BLACK_COLOR_TOKEN = colors_config.get('black_color_token', 
                                       colors_config.get('color_token_start', 40010) + 2)

# ============================================================
# Model settings from config (shared)
# ============================================================
model_config = config.get('model', {})
BOS_TOKEN_ID = model_config.get('bos_token_id', 196998)
EOS_TOKEN_ID = model_config.get('eos_token_id', 196999)
PAD_TOKEN_ID = model_config.get('pad_token_id', 151643)
MAX_LENGTH = model_config.get('max_length', 1024)
MIN_MAX_LENGTH = 256
MAX_MAX_LENGTH = 2048

# ============================================================
# Task configurations with defaults from config (shared)
# ============================================================
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

# ============================================================
# Generation parameters from config (shared)
# ============================================================
gen_config = config.get('generation', {})
DEFAULT_NUM_CANDIDATES = gen_config.get('default_num_candidates', 4)
MAX_NUM_CANDIDATES = gen_config.get('max_num_candidates', 8)
EXTRA_CANDIDATES_BUFFER = gen_config.get('extra_candidates_buffer', 4)

# ============================================================
# Validation settings from config (shared)
# ============================================================
validation_config = config.get('validation', {})
MIN_SVG_LENGTH = validation_config.get('min_svg_length', 20)

# Custom CSS
CUSTOM_CSS = """
/* Main container centering */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}
/* Header styling */
.header-container {
    text-align: center;
    margin-bottom: 20px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    color: white;
}
.header-container h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 700;
}
.header-container p {
    margin: 10px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
}
/* Model selector styling */
.model-selector {
    background: #f0f4f8;
    border: 2px solid #667eea;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 20px;
}
.model-selector-title {
    font-weight: 700;
    color: #667eea;
    margin-bottom: 10px;
}
/* Tips section */
.tips-box {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #e0e0e0;
}
.tips-box h3 {
    margin-top: 0;
    color: #333;
    border-bottom: 2px solid #667eea;
    padding-bottom: 10px;
}
.tip-category {
    background: white;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.tip-category h4 {
    margin: 0 0 10px 0;
    color: #667eea;
}
.tip-category code {
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}
.example-prompt {
    background: #e8f4fd;
    padding: 10px;
    border-radius: 6px;
    margin: 8px 0;
    font-style: italic;
    font-size: 0.95em;
    color: #333;
}
.red-tip {
    color: #dc3545;
    font-weight: 600;
}
.red-box {
    background: #fff5f5;
    border: 1px solid #ffcccc;
    border-left: 4px solid #dc3545;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
}
.red-box strong {
    color: #dc3545;
}
.orange-box {
    background: #fff8e6;
    border: 1px solid #ffc107;
    border-left: 4px solid #ff9800;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
}
.orange-box strong {
    color: #ff9800;
}
.green-box {
    background: #e8f5e9;
    border: 1px solid #81c784;
    border-left: 4px solid #4caf50;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
}
.green-box strong {
    color: #4caf50;
}
.blue-box {
    background: #e3f2fd;
    border: 1px solid #90caf9;
    border-left: 4px solid #2196f3;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
}
.blue-box strong {
    color: #2196f3;
}
/* Tab styling */
.tabs {
    border-radius: 12px !important;
    overflow: hidden;
}
.tabitem {
    padding: 20px !important;
}
/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    font-size: 1.1em !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}
/* Settings group */
.settings-group {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.advanced-settings {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
}
/* Code output */
.code-output textarea {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 12px !important;
    background: #1e1e1e !important;
    color: #d4d4d4 !important;
    border-radius: 8px !important;
}
/* Input image area */
.input-image {
    border: 2px dashed #ccc;
    border-radius: 12px;
    transition: border-color 0.3s;
}
.input-image:hover {
    border-color: #667eea;
}
/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 0.9em;
}
/* Responsive adjustments */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }
    .header-container h1 {
        font-size: 1.8em;
    }
}
"""

# Enhanced Tips HTML
TIPS_HTML = """
<div class="tips-box">
    <h3>Prompting Guide & Best Practices</h3>
    
    <!-- Critical Red Tips Section -->
    <div class="red-box">
        <strong>CRITICAL: Tips That WILL Improve Your Results</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li style="color: #dc3545; font-weight: 600;">
                <strong>Generate 4-8 candidates and pick the best one!</strong> Results vary significantly between generations - this is NORMAL!
            </li>
            <li style="color: #dc3545; font-weight: 600;">
                <strong>Use GEOMETRIC descriptions:</strong> "triangular roof", "circular head", "rectangular body", "curved tail"
            </li>
            <li style="color: #dc3545; font-weight: 600;">
                <strong>ALWAYS specify colors for EACH element:</strong> "black outline", "red roof", "blue shirt", "green grass"
            </li>
            <li style="color: #dc3545; font-weight: 600;">
                <strong>Describe position & orientation:</strong> "centrally positioned", "pointing upward", "facing right", "at the bottom"
            </li>
            <li style="color: #dc3545; font-weight: 600;">
                <strong>Keep it SIMPLE:</strong> Avoid complex sentences. Use short, clear phrases connected by commas.
            </li>
        </ul>
    </div>
    
    <!-- Model Selection Tips -->
    <div class="blue-box">
        <strong>Model Selection Guide</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>8B Model:</strong> Higher quality, more details, better for complex illustrations. Requires more VRAM (~16GB+).</li>
            <li><strong>4B Model:</strong> Faster, less VRAM required (~8GB+). Good for simple icons and basic shapes.</li>
            <li><strong>Note:</strong> First generation with a new model size may take longer to load.</li>
        </ul>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px;">
        
        <div class="tip-category">
            <h4>Icons & Simple Shapes</h4>
            <p>Use clear geometric descriptions with explicit colors.</p>
            <div class="example-prompt">
                "A black triangle pointing downward, centrally positioned."
            </div>
            <div class="example-prompt">
                "A red heart shape with smooth curved edges, centered."
            </div>
            <p><strong>Keywords:</strong> <code>triangle</code> <code>circle</code> <code>arrow</code> <code>heart</code> <code>star</code> <code>centered</code></p>
        </div>
        
        <div class="tip-category">
            <h4>Animals</h4>
            <p>Describe as geometric shapes: oval body, round head, triangular ears, curved tail.</p>
            <div class="example-prompt">
                "Cute cat: orange round head with two triangular ears, oval orange body, curved tail. Simple cartoon style with black outlines, sitting pose."
            </div>
            <div class="example-prompt">
                "Simple black bird: oval body, small round head, pointed triangular beak facing right, triangular tail, two stick legs. Silhouette style."
            </div>
        </div>
        
        <div class="tip-category">
            <h4>Buildings & Objects</h4>
            <p>Use basic shapes: rectangles for walls, triangles for roofs, squares for windows.</p>
            <div class="example-prompt">
                "Simple house: red triangular roof on top, beige rectangular wall, brown rectangular door in center, two small blue square windows. Green ground at bottom."
            </div>
            <div class="example-prompt">
                "Coffee mug: brown cylindrical cup shape with curved handle on right side, three wavy steam lines rising from top. Simple flat style."
            </div>
        </div>
        
    </div>
    
    <!-- Quick Troubleshooting -->
    <div class="green-box" style="margin-top: 15px;">
        <strong>Quick Troubleshooting</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
            <li><strong>Messy/chaotic?</strong> Lower temperature to 0.3-0.4, simplify description, reduce top_k</li>
            <li><strong>Too simple/empty?</strong> Raise temperature to 0.5-0.6, add more shape details</li>
            <li><strong>Wrong colors?</strong> Explicitly name EVERY color: "red roof", "blue shirt", "black outline"</li>
            <li><strong>Missing elements?</strong> Add position words: "at top", "in center", "at bottom left"</li>
            <li><strong>Repetitive patterns?</strong> Increase repetition_penalty to 1.08-1.15</li>
            <li><strong>Inconsistent?</strong> <span class="red-tip">Generate MORE candidates (6-8) and pick the best!</span></li>
        </ul>
    </div>
    
    <!-- Prompt Template -->
    <div style="margin-top: 15px; padding: 12px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #4caf50;">
        <strong>Recommended Prompt Structure</strong>
        <div style="background: white; padding: 10px; border-radius: 6px; margin-top: 8px; font-family: monospace; font-size: 0.9em;">
            [Subject] + [Shape descriptions with colors] + [Position/orientation] + [Style]
        </div>
        <p style="margin: 10px 0 0 0; color: #2e7d32; font-size: 0.95em;">
            Example: "A fox logo: triangular orange head, pointed ears, white chest marking, facing right. Minimalist flat style, centered."
        </p>
    </div>
</div>
"""

# Image-to-SVG specific tips
IMAGE_TIPS_HTML = """
<div class="red-box">
    <strong>Image-to-SVG Tips</strong>
    <ul style="margin: 8px 0 0 0; padding-left: 20px;">
        <li><strong>Best input: Simple images with clean background</strong></li>
        <li><strong>PNG with transparency (RGBA) works best!</strong> We auto-convert to white background.</li>
        <li><strong>For complex backgrounds:</strong> Enable "Replace Background" option below.</li>
        <li><strong>Lower temperature (0.2-0.4)</strong> for more accurate reproduction.</li>
        <li style="color: #dc3545; font-weight: 600;"><strong>Generate 4-8 candidates!</strong> Pick the one that best matches your input.</li>
    </ul>
</div>
"""


def parse_args():
    parser = argparse.ArgumentParser(description='SVG Generator Service')
    parser.add_argument('--listen', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--preload-model', type=str, default=None, 
                        choices=AVAILABLE_MODEL_SIZES,
                        help='Preload a specific model size at startup')
    return parser.parse_args()


def download_model_weights(repo_id: str, filename: str = "pytorch_model.bin") -> str:
    """
    Download model weights from Hugging Face Hub.
    """
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
    """
    Load all models for a specific model size.
    """
    global tokenizer, processor, sketch_decoder, svg_tokenizer, current_model_size
    
    # Use config values if not provided
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
        pix_len=MAX_MAX_LENGTH,  # Use max possible length
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
    
    sketch_decoder = sketch_decoder.to(device).eval()
    
    # Initialize SVG tokenizer with model_size
    svg_tokenizer = SVGTokenizer(CONFIG_PATH, model_size=model_size)
    
    current_model_size = model_size
    
    print("\n" + "="*60)
    print(f"All {model_size} models loaded successfully!")
    print("="*60 + "\n")


def ensure_model_loaded(model_size: str):
    """
    Ensure the specified model is loaded. Load or switch if necessary.
    """
    global current_model_size, sketch_decoder, tokenizer, processor, svg_tokenizer
    
    if current_model_size == model_size and sketch_decoder is not None:
        return  # Already loaded
    
    with model_loading_lock:
        # Double-check after acquiring lock
        if current_model_size == model_size and sketch_decoder is not None:
            return
        
        # Clear old models if switching
        if current_model_size is not None:
            print(f"Switching from {current_model_size} to {model_size}...")
            del sketch_decoder
            del tokenizer
            del processor
            del svg_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load new model
        load_models(model_size)


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
    """
    Detect if image has non-white background and optionally replace it.
    """
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
    """
    Preprocess image for SVG generation.
    """
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


def create_gallery_html(candidates, cols=4):
    """Create HTML gallery for multiple SVG candidates"""
    if not candidates:
        return '<div style="text-align:center;color:#999;padding:50px;">No candidates generated</div>'
    
    items_html = []
    for i, cand in enumerate(candidates):
        svg_str = cand['svg']
        if 'viewBox' not in svg_str:
            svg_str = svg_str.replace('<svg', f'<svg viewBox="0 0 {TARGET_IMAGE_SIZE} {TARGET_IMAGE_SIZE}"', 1)
        
        item_html = f'''
        <div style="
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        " onmouseover="this.style.transform='scale(1.02)';this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)';"
           onmouseout="this.style.transform='scale(1)';this.style.boxShadow='none';">
            <div style="width: 180px; height: 180px; margin: 0 auto; display: flex; justify-content: center; align-items: center; overflow: hidden;">
                {svg_str}
            </div>
            <div style="margin-top: 8px; font-size: 12px; color: #666;">
                #{i+1} | {cand['path_count']} paths
            </div>
        </div>
        '''
        items_html.append(item_html)
    
    grid_html = f'''
    <div style="
        display: grid;
        grid-template-columns: repeat({cols}, 1fr);
        gap: 15px;
        padding: 15px;
        background: #fafafa;
        border-radius: 12px;
    ">
        {''.join(items_html)}
    </div>
    '''
    return grid_html


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
                       max_length, num_samples, progress_callback=None):
    """Generate candidate SVGs with full parameter control"""
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if 'pixel_values' in inputs:
        model_inputs["pixel_values"] = inputs['pixel_values'].to(device, dtype=DTYPE)
    
    if 'image_grid_thw' in inputs:
        model_inputs["image_grid_thw"] = inputs['image_grid_thw'].to(device)
    
    all_candidates = []
    
    # Generation config with user parameters
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
        if progress_callback:
            progress_callback(0.1, "Waiting for model access...")
        
        with generation_lock:
            if progress_callback:
                progress_callback(0.15, "Generating SVG tokens...")
            
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
        
        if progress_callback:
            progress_callback(0.5, "Processing generated tokens...")
        
        for i in range(min(actual_samples, generated_ids_batch.shape[0])):
            try:
                current_ids = generated_ids_batch[i:i+1]
                
                fake_wrapper = torch.cat([
                    torch.full((1, 1), BOS_TOKEN_ID, device=device),
                    current_ids,
                    torch.full((1, 1), EOS_TOKEN_ID, device=device)
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
                    
                    if progress_callback:
                        progress_callback(0.5 + 0.4 * (i / actual_samples), 
                                        f"Found {len(all_candidates)} valid candidates...")
                    
                    if len(all_candidates) >= num_samples:
                        break
                        
            except Exception as e:
                print(f"  Candidate {i} error: {e}")
                continue

    except Exception as e:
        print(f"Generation Error: {e}")
        import traceback
        traceback.print_exc()
    
    if progress_callback:
        progress_callback(0.95, f"Generated {len(all_candidates)} valid candidates")
    
    return all_candidates


def gradio_text_to_svg(text_description, model_size, num_candidates, temperature, top_p, top_k, 
                       repetition_penalty, max_length, progress=gr.Progress()):
    """Gradio interface - text-to-svg with model selection"""
    if not text_description or text_description.strip() == "":
        return '<div style="text-align:center;color:#999;padding:50px;">Please enter a description</div>', "", f"Ready (no model loaded)"
    
    print("\n" + "="*60)
    print(f"[TASK] text-to-svg")
    print(f"[MODEL] Requested: {model_size}")
    print(f"[INPUT] {text_description[:100]}{'...' if len(text_description) > 100 else ''}")
    print(f"[PARAMS] candidates={num_candidates}, temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}, max_length={max_length}")
    print("="*60)
    
    progress(0, "Initializing...")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # Ensure model is loaded
    progress(0.02, f"Loading {model_size} model (first time may take a while)...")
    ensure_model_loaded(model_size)
    model_status = f"✅ Using {model_size} model"
    
    progress(0.05, "Model ready, starting generation...")
    
    subtype = detect_text_subtype(text_description)
    print(f"[SUBTYPE] Detected: {subtype}")
    progress(0.08, f"Detected: {subtype}")
    
    inputs = prepare_inputs("text-to-svg", text_description.strip())
    
    def update_progress(val, msg):
        progress(val, msg)
    
    all_candidates = generate_candidates(
        inputs, "text-to-svg", subtype,
        temperature, top_p, int(top_k), repetition_penalty,
        int(max_length), int(num_candidates),
        progress_callback=update_progress
    )
    
    elapsed = time.time() - start_time
    
    print(f"[RESULT] Generated {len(all_candidates)} valid candidates in {elapsed:.2f}s")
    
    if not all_candidates:
        print("[WARNING] No valid SVG generated")
        return (
            '<div style="text-align:center;color:#999;padding:50px;">No valid SVG generated. Try different parameters or rephrase your prompt.</div>',
            f"<!-- No valid SVG (took {elapsed:.1f}s) -->",
            model_status
        )
    
    svg_codes = []
    for i, cand in enumerate(all_candidates):
        svg_codes.append(f"<!-- ====== Candidate {i+1} | Paths: {cand['path_count']} ====== -->\n{cand['svg']}")
    
    combined_svg = "\n\n".join(svg_codes)
    gallery_html = create_gallery_html(all_candidates)
    
    progress(1.0, f"Done! {len(all_candidates)} candidates in {elapsed:.1f}s")
    print(f"[COMPLETE] text-to-svg finished\n")
    
    return gallery_html, combined_svg, model_status


def gradio_image_to_svg(image, model_size, num_candidates, temperature, top_p, top_k, repetition_penalty,
                        max_length, replace_background, progress=gr.Progress()):
    """Gradio interface - image-to-svg with model selection"""
    
    if image is None:
        return (
            '<div style="text-align:center;color:#999;padding:50px;">Please upload an image</div>',
            "",
            None,
            f"Ready (no model loaded)"
        )
    
    print("\n" + "="*60)
    print(f"[TASK] image-to-svg")
    print(f"[MODEL] Requested: {model_size}")
    print(f"[INPUT] Image size: {image.size if hasattr(image, 'size') else 'unknown'}, mode: {image.mode if hasattr(image, 'mode') else 'unknown'}")
    print(f"[PARAMS] candidates={num_candidates}, temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}, max_length={max_length}, replace_bg={replace_background}")
    print("="*60)
    
    progress(0, "Initializing...")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # Ensure model is loaded
    progress(0.02, f"Loading {model_size} model (first time may take a while)...")
    ensure_model_loaded(model_size)
    model_status = f"✅ Using {model_size} model"
    
    progress(0.05, "Processing input image...")
    
    img_processed, was_modified = preprocess_image_for_svg(
        image, 
        replace_background=replace_background,
        target_size=TARGET_IMAGE_SIZE
    )
    
    if was_modified:
        print("[PREPROCESS] Background processed/replaced")
        progress(0.08, "Background processed")
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img_processed.save(tmp_file.name, format='PNG', quality=100)
        tmp_path = tmp_file.name
    
    try:
        progress(0.1, "Preparing model inputs...")
        inputs = prepare_inputs("image-to-svg", tmp_path)
        
        def update_progress(val, msg):
            progress(val, msg)
        
        all_candidates = generate_candidates(
            inputs, "image-to-svg", "image",
            temperature, top_p, int(top_k), repetition_penalty,
            int(max_length), int(num_candidates),
            progress_callback=update_progress
        )
        
        elapsed = time.time() - start_time
        
        print(f"[RESULT] Generated {len(all_candidates)} valid candidates in {elapsed:.2f}s")
        
        if not all_candidates:
            print("[WARNING] No valid SVG generated")
            return (
                '<div style="text-align:center;color:#999;padding:50px;">No valid SVG generated. Try adjusting parameters.</div>',
                f"<!-- No valid SVG (took {elapsed:.1f}s) -->",
                img_processed,
                model_status
            )
        
        svg_codes = []
        for i, cand in enumerate(all_candidates):
            svg_codes.append(f"<!-- ====== Candidate {i+1} | Paths: {cand['path_count']} ====== -->\n{cand['svg']}")
        
        combined_svg = "\n\n".join(svg_codes)
        gallery_html = create_gallery_html(all_candidates)
        
        progress(1.0, f"Done! {len(all_candidates)} candidates in {elapsed:.1f}s")
        print(f"[COMPLETE] image-to-svg finished\n")
        
        return gallery_html, combined_svg, img_processed, model_status
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_example_images():
    """Get example images from the examples directory"""
    example_dir = "./examples"
    example_images = []
    
    if os.path.exists(example_dir):
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(example_dir, f"*{ext}")
            example_images.extend(glob.glob(pattern))
        example_images.sort()
    
    return example_images


def create_interface():
    """Create Gradio interface"""
    
    # 30 Example prompts covering various categories
    example_texts = [
        # === Simple Icons (1-6) ===
        "A black triangle pointing downward, centrally positioned.",
        "A red heart shape with smooth curved edges, centered.",
        "A yellow star with five sharp points, simple geometric design, flat color.",
        "A blue arrow pointing to the right, thick solid shape, centered.",
        "A green circle with a white checkmark inside, centered.",
        "A black plus sign with equal length arms, thick lines, centered.",
        
        "A simple person standing: round beige head, rectangular blue shirt body, two dark gray rectangular legs, arms at sides. Flat colors.",
        "A girl with long black hair, wearing pink dress with triangular skirt, small circular face with dot eyes and curved smile. Simple cartoon style.",
        "A child waving: large round head with brown messy hair, big circular eyes, small body in red t-shirt and blue shorts, one arm raised. Cheerful cartoon style.",
        "A person sitting on chair: side view, round head, rectangular torso in green sweater, bent legs on simple chair shape. Relaxed pose.",
        "A running person: side view silhouette in black, dynamic pose with one leg forward, arms pumping. Motion style.",
        
        # === Avatars & Portraits (13-17) ===
        "Circular avatar: person with short black hair, round face with two dot eyes and small curved smile, wearing blue collar shirt. Minimal style, centered in circle.",
        "Female avatar: oval face with long wavy brown hair, simple eyes, pink lips, wearing v-neck purple top. Soft cartoon style in circular frame.",
        "Profile silhouette avatar: black side view of head with short hair and glasses outline, facing right. Simple solid shape.",
        "Cute cartoon avatar: round face with big sparkly eyes, rosy cheeks, short bob haircut in orange. Kawaii style, circular frame.",
        "Professional headshot avatar: person with neat hair, neutral expression, wearing suit collar. Corporate minimal style, circular frame.",
        
        # === Landscapes & Scenes (18-23) ===
        "Layered mountain landscape: light blue sky at top, gray triangular snow-capped mountains in middle, dark green triangular pine trees at bottom. Flat colors.",
        "Sunset beach scene: orange gradient sky at top, yellow semicircle sun on horizon, dark blue wavy ocean, tan beach strip at bottom. Simple shapes.",
        "Forest scene: light blue sky, row of 5 dark green triangular pine trees of varying heights on brown trunks, light green grass at bottom.",
        "City skyline at dusk: purple-orange gradient sky, row of black rectangular building silhouettes of different heights, some with yellow window squares.",
        "Desert landscape: light orange sky with white circle sun, tan sand dunes as curved shapes, one green cactus with arms on the right side.",
        "Countryside scene: blue sky with white fluffy clouds, green rolling hills, small red barn with white door in the center, yellow hay bales.",
        
        # === Animals (24-27) ===
        "Cute orange cat sitting: round head with two triangular ears, oval body, curved tail. Black outline cartoon style, facing forward.",
        "Simple black bird: oval body, round head, pointed triangular beak facing right, triangular tail, two stick legs. Silhouette style.",
        "Friendly cartoon dog: brown oval body, round head with floppy ears, black dot nose, wagging curved tail, four short legs. Sitting pose.",
        "Red fox logo: triangular orange face with pointed ears, white chest marking, bushy tail. Minimalist style, facing right, centered.",
        
        # === Objects & Misc (28-30) ===
        "Simple house icon: red triangular roof, beige rectangular walls, brown door in center, two blue square windows, green ground at bottom.",
        "Coffee mug: brown cylindrical cup with curved handle on right, three wavy steam lines rising from top. Flat style.",
        "Open book: two rectangular white pages spread open, black text lines on each page, brown spine in center. Simple top-down view."
    ]
    
    example_images = get_example_images()
    
    with gr.Blocks(title="OmniSVG Generator", css=CUSTOM_CSS) as demo:
        # Header
        gr.HTML("""
        <div class="header-container">
            <h1>OmniSVG Generator</h1>
            <p>Transform images and text descriptions into scalable vector graphics</p>
        </div>
        """)
        
        # Queue status
        gr.HTML("""
        <div style="background: #e7f3ff; border: 1px solid #b3d7ff; border-radius: 8px; padding: 12px 15px; margin: 15px 0;">
            <span style="font-size: 1.5em;">ℹ️</span>
            <strong>Queue System Active</strong> - Requests processed one at a time. First generation with a new model may take longer to load.
        </div>
        """)
                
        with gr.Tabs():
            # ==================== Image-to-SVG Tab ====================
            with gr.TabItem("Image-to-SVG", id="image-tab"):
                gr.HTML(IMAGE_TIPS_HTML)
                
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Upload Image")
                        image_input = gr.Image(
                            label="Drag, upload, or Ctrl+V to paste", 
                            type="pil", 
                            image_mode="RGBA",
                            height=250,
                            sources=["upload", "clipboard"],
                            elem_classes=["input-image"]
                        )
                        
                        with gr.Group(elem_classes=["settings-group"]):
                            gr.Markdown("### Settings")
                            
                            # Model Selection
                            img_model_size = gr.Dropdown(
                                choices=AVAILABLE_MODEL_SIZES,
                                value=DEFAULT_MODEL_SIZE,
                                label="Model Size",
                                info="8B: Higher quality (~16GB VRAM) | 4B: Faster (~8GB VRAM)"
                            )
                            
                            img_num_candidates = gr.Slider(
                                minimum=1, maximum=MAX_NUM_CANDIDATES, value=DEFAULT_NUM_CANDIDATES, step=1,
                                label="Number of Candidates"
                            )
                            img_replace_bg = gr.Checkbox(
                                label="Replace non-white background",
                                value=True,
                                info="Enable for images with colored backgrounds"
                            )
                            
                            # Max Length slider (new)
                            img_max_length = gr.Slider(
                                minimum=MIN_MAX_LENGTH, 
                                maximum=MAX_MAX_LENGTH, 
                                value=MAX_LENGTH, 
                                step=64,
                                label="Max Token Length",
                                info="Lower = faster + simpler SVG | Higher = slower + more complex SVG"
                            )
                            
                            with gr.Accordion("Advanced Parameters", open=False):
                                img_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, 
                                    value=TASK_CONFIGS["image-to-svg"].get("default_temperature", 0.3), 
                                    step=0.05,
                                    label="Temperature (Lower=accurate)",
                                    info="0.2-0.4 recommended"
                                )
                                img_top_p = gr.Slider(
                                    minimum=0.5, maximum=1.0, 
                                    value=TASK_CONFIGS["image-to-svg"].get("default_top_p", 0.90), 
                                    step=0.02,
                                    label="Top-P"
                                )
                                img_top_k = gr.Slider(
                                    minimum=10, maximum=100, 
                                    value=TASK_CONFIGS["image-to-svg"].get("default_top_k", 50), 
                                    step=5,
                                    label="Top-K"
                                )
                                img_rep_penalty = gr.Slider(
                                    minimum=1.0, maximum=1.3, 
                                    value=TASK_CONFIGS["image-to-svg"].get("default_repetition_penalty", 1.05), 
                                    step=0.01,
                                    label="Repetition Penalty"
                                )
                        
                        image_generate_btn = gr.Button(
                            "Generate SVG", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                        
                        # Model status display
                        img_model_status = gr.Textbox(
                            label="Model Status",
                            value="Ready (model loads on first generation)",
                            interactive=False
                        )
                        
                        if example_images:
                            gr.Markdown("### Examples")
                            gr.Examples(examples=example_images, inputs=[image_input], label="")
                    
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown("### Processed Input")
                        image_processed = gr.Image(label="", type="pil", height=120)
                        
                        gr.Markdown("### Generated SVG Candidates")
                        image_gallery = gr.HTML(
                            value='<div style="text-align:center;color:#999;padding:50px;background:#fafafa;border-radius:12px;">Generated SVGs will appear here</div>'
                        )
                        
                        gr.Markdown("### SVG Code")
                        image_svg_output = gr.Code(label="", language="html", lines=10, elem_classes=["code-output"])
                
                image_generate_btn.click(
                    fn=gradio_image_to_svg,
                    inputs=[image_input, img_model_size, img_num_candidates, img_temperature, img_top_p, 
                           img_top_k, img_rep_penalty, img_max_length, img_replace_bg],
                    outputs=[image_gallery, image_svg_output, image_processed, img_model_status],
                    queue=True
                )
            
            # ==================== Text-to-SVG Tab ====================
            with gr.TabItem("Text-to-SVG", id="text-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Description")
                        gr.HTML("""
                        <div style="background: #fff5f5; padding: 10px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 10px;">
                            <strong style="color: #dc3545;">Generate 4-8 candidates and pick the best!</strong>
                        </div>
                        """)
                        text_input = gr.Textbox(
                            label="",
                            placeholder="Describe your SVG with geometric shapes and colors...\n\nExample: A black triangle pointing downward, centrally positioned.",
                            lines=5
                        )
                        
                        with gr.Group(elem_classes=["settings-group"]):
                            gr.Markdown("### Settings")
                            
                            # Model Selection
                            text_model_size = gr.Dropdown(
                                choices=AVAILABLE_MODEL_SIZES,
                                value=DEFAULT_MODEL_SIZE,
                                label="Model Size",
                                info="8B: Higher quality (~16GB VRAM) | 4B: Faster (~8GB VRAM)"
                            )
                            
                            text_num_candidates = gr.Slider(
                                minimum=1, maximum=MAX_NUM_CANDIDATES, value=6, step=1,
                                label="Number of Candidates",
                                info="More = better chances!"
                            )
                            
                            # Max Length slider (new)
                            text_max_length = gr.Slider(
                                minimum=MIN_MAX_LENGTH, 
                                maximum=MAX_MAX_LENGTH, 
                                value=MAX_LENGTH, 
                                step=64,
                                label="Max Token Length",
                                info="Lower = faster + simpler SVG | Higher = slower + more complex SVG"
                            )
                            
                            with gr.Accordion("Advanced Parameters", open=False):
                                text_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, 
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_temperature", 0.5), 
                                    step=0.05,
                                    label="Temperature",
                                    info="Icons: 0.3-0.5 | Complex: 0.5-0.7"
                                )
                                text_top_p = gr.Slider(
                                    minimum=0.5, maximum=1.0, 
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_p", 0.90), 
                                    step=0.02,
                                    label="Top-P"
                                )
                                text_top_k = gr.Slider(
                                    minimum=10, maximum=100, 
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_top_k", 60), 
                                    step=5,
                                    label="Top-K"
                                )
                                text_rep_penalty = gr.Slider(
                                    minimum=1.0, maximum=1.3, 
                                    value=TASK_CONFIGS["text-to-svg-icon"].get("default_repetition_penalty", 1.03), 
                                    step=0.01,
                                    label="Repetition Penalty",
                                    info="Increase if you see repetitive patterns"
                                )
                        
                        text_generate_btn = gr.Button(
                            "Generate SVG", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                        
                        # Model status display
                        text_model_status = gr.Textbox(
                            label="Model Status",
                            value="Ready (model loads on first generation)",
                            interactive=False
                        )
                        
                        gr.Markdown("### Example Prompts (30)")
                        gr.Examples(
                            examples=[[text] for text in example_texts],
                            inputs=[text_input],
                            label=""
                        )
                    
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown("### Generated SVG Candidates")
                        gr.HTML("""
                        <div style="background: #d4edda; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                            <strong>Pick the best from multiple candidates!</strong>
                        </div>
                        """)
                        text_gallery = gr.HTML(
                            value='<div style="text-align:center;color:#999;padding:50px;background:#fafafa;border-radius:12px;">Generated SVGs will appear here</div>'
                        )
                        
                        gr.Markdown("### SVG Code")
                        text_svg_output = gr.Code(label="", language="html", lines=12, elem_classes=["code-output"])
                
                text_generate_btn.click(
                    fn=gradio_text_to_svg,
                    inputs=[text_input, text_model_size, text_num_candidates, text_temperature, text_top_p, 
                           text_top_k, text_rep_penalty, text_max_length],
                    outputs=[text_gallery, text_svg_output, text_model_status],
                    queue=True
                )
        
        gr.HTML(TIPS_HTML)

        # Footer
        gr.HTML(f"""
        <div class="footer">
            <p>Built with OmniSVG | Available models: {', '.join(AVAILABLE_MODEL_SIZES)}</p>
            <p style="color: #dc3545; font-weight: 600;">Remember: Generate 4-8 candidates and pick the best!</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    print("="*60)
    print("OmniSVG Demo Page - Gradio App (Local Version)")
    print("="*60)
    print(f"Available model sizes: {AVAILABLE_MODEL_SIZES}")
    print(f"Default model size: {DEFAULT_MODEL_SIZE}")
    print(f"Device: {device}")
    print(f"Precision: {DTYPE}")
    print(f"Max Length Range: {MIN_MAX_LENGTH} - {MAX_MAX_LENGTH} (default: {MAX_LENGTH})")
    print("="*60)
    
    # Print loaded config values
    print("\n[CONFIG] Shared settings:")
    print(f"  - TARGET_IMAGE_SIZE: {TARGET_IMAGE_SIZE}")
    print(f"  - RENDER_SIZE: {RENDER_SIZE}")
    print(f"  - BLACK_COLOR_TOKEN: {BLACK_COLOR_TOKEN}")
    print(f"  - MAX_LENGTH: {MAX_LENGTH}")
    print(f"  - BOS_TOKEN_ID: {BOS_TOKEN_ID}")
    print(f"  - EOS_TOKEN_ID: {EOS_TOKEN_ID}")
    print(f"  - PAD_TOKEN_ID: {PAD_TOKEN_ID}")
    print("="*60)
    
    # Preload model if specified
    if args.preload_model:
        print(f"\nPreloading {args.preload_model} model...")
        load_models(args.preload_model)
    else:
        print("\nModels will be loaded on-demand when first used.")
    
    print("="*60)
    
    demo = create_interface()
    
    demo.queue(default_concurrency_limit=1, max_size=20)
    
    demo.launch(
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )
