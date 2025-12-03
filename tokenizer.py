import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Optional, Union
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox


class SVGTokenizer:
    """SVG tokenizer - supports both 8B and 4B models via config.yaml"""
    
    def __init__(self, config_path: str = "./config.yaml", model_size: str = None):
        """
        Initialize SVGTokenizer.
        
        Args:
            config_path: Path to config.yaml
            model_size: Model size ("8B" or "4B"). If None, uses default from config.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determine model size
        self.model_size = model_size or self.config.get('default_model_size', '8B')
        if self.model_size not in self.config.get('models', {}):
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be one of: {list(self.config.get('models', {}).keys())}")
        
        self._load_config()
        self.pixel2xy = self._create_pixel2xy_mapping()
    
    def _get_model_specific_config(self, *keys):
        """Get model-specific config value, with fallback to shared config."""
        model_cfg = self.config.get('models', {}).get(self.model_size, {})
        
        # Navigate through nested keys in model-specific config
        value = model_cfg
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        
        # If not found in model-specific, try shared config
        if value is None:
            value = self.config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
        
        return value
    
    def _load_config(self):
        """Load all constants from configuration file with model-specific overrides."""
        # ========== Token-related configs ==========
        # Model-specific tokens
        self.NUM_MASK_AND_EOM = self._get_model_specific_config('tokens', 'num_mask_and_eom')
        self.BASE_OFFSET = self._get_model_specific_config('tokens', 'base_offset')
        
        # Shared tokens
        tokens_cfg = self.config['tokens']
        self.NUM_SVG_END = tokens_cfg['svg_end']
        self.NUM_END_TOKEN = tokens_cfg['num_end_token']
        
        # ========== Coordinate-related configs ==========
        # Model-specific coordinates
        self.PIX_PAD = self._get_model_specific_config('coordinates', 'pix_pad_offset')
        self.COORD_PAD = self._get_model_specific_config('coordinates', 'coord_pad_offset')
        
        # Shared coordinates
        coords_cfg = self.config['coordinates']
        self.BBOX = coords_cfg['bbox']
        
        # ========== Color-related configs ==========
        colors_cfg = self.config['colors']
        self.COLOR_TOKEN_START_RAW = colors_cfg['color_token_start']
        self.MAX_COLOR_TOKENS = colors_cfg['max_color_tokens']
        
        # Model-specific colors
        self.COLOR_START_OFFSET = self._get_model_specific_config('colors', 'color_start_offset')
        self.COLOR_END_OFFSET = self._get_model_specific_config('colors', 'color_end_offset')
        
        # ========== SVG command values ==========
        commands_cfg = self.config['svg_commands']
        self.CMD_MOVE = commands_cfg['move']
        self.CMD_LINE = commands_cfg['line']
        self.CMD_CURVE = commands_cfg['curve']
        self.CMD_ARC = commands_cfg['arc']
        self.CMD_CLOSE = commands_cfg['close']
        
        # ========== Model-related configs ==========
        model_cfg = self.config['model']
        self.BOS_TOKEN_ID = model_cfg['bos_token_id']
        self.EOS_TOKEN_ID = model_cfg['eos_token_id']
        self.PAD_TOKEN_ID = model_cfg['pad_token_id']
        
        # ========== Arc parameter configs ==========
        arc_cfg = self.config.get('arc', {})
        self.ARC_PARAM_OFFSET = arc_cfg.get('param_offset', 44500)
        self.ARC_PARAM_RANGE = arc_cfg.get('param_range', 100)
        self.ARC_PARAM_START = self.ARC_PARAM_OFFSET + self.BASE_OFFSET
        
        # ========== Derived constants ==========
        self.PIXEL_OFFSET = (self.NUM_MASK_AND_EOM - self.BASE_OFFSET + 
                             self.NUM_SVG_END - self.CMD_MOVE)
        
        # Command token range
        self.CMD_TOKEN_START = self.NUM_MASK_AND_EOM + self.NUM_SVG_END
        self.CMD_TOKEN_END = self.PIX_PAD + self.NUM_SVG_END
        
        # Coordinate token start
        self.COORD_TOKEN_START = self.PIX_PAD + self.NUM_SVG_END
        
        # Color-coordinate boundary
        self.COLOR_COORD_BOUNDARY = self.COLOR_TOKEN_START_RAW + 1 + self.BASE_OFFSET
        
        # Color threshold for raster_svg
        self.COLOR_THRESHOLD = self.COLOR_TOKEN_START_RAW - self.PIXEL_OFFSET + 1
        
    def _create_pixel2xy_mapping(self) -> Dict[int, np.ndarray]:
        """Create pixel to xy mapping following dataset.py logic."""
        pixel2xy = {}
        x = np.linspace(0, self.BBOX - 1, self.BBOX)
        y = np.linspace(0, self.BBOX - 1, self.BBOX)
        xx, yy = np.meshgrid(x, y)
        xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        
        for pixel, xy in enumerate(xy_grid):
            pixel2xy[pixel] = xy + self.COORD_PAD + self.NUM_SVG_END
            
        return pixel2xy
    
    def token_to_color(self, color_token: int) -> str:
        """Convert token to color following dataset.py logic."""
        try:
            if color_token == self.COLOR_TOKEN_START_RAW:
                return "none"
            elif color_token == self.COLOR_TOKEN_START_RAW + 1:
                return "currentColor"
            
            color_index = color_token - (self.COLOR_TOKEN_START_RAW + 2)
            
            if color_index < 0 or color_index >= self.MAX_COLOR_TOKENS:
                print(f"Warning: Color token {color_token} out of range")
                return "#808080"
            
            r = (color_index >> 8) & 0xF
            g = (color_index >> 4) & 0xF
            b = color_index & 0xF
            
            r = (r << 4) | r
            g = (g << 4) | g
            b = (b << 4) | b
            
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception as e:
            print(f"Error in token_to_color: {e}")
            return "#808080"

    def process_generated_tokens(self, output_ids: torch.Tensor) -> np.ndarray:
        """Process generated tokens following dataset.py logic."""
        # Remove bos/eos
        generated_pixels = output_ids[:, 1:-1].cpu().numpy().flatten()
        
        sample_xys = []
        
        for pixel in generated_pixels:
            try:
                # 1. Command tokens: CMD_TOKEN_START <= pixel < CMD_TOKEN_END
                if self.CMD_TOKEN_START <= pixel < self.CMD_TOKEN_END:
                    xy = np.array([pixel - self.BASE_OFFSET, 
                                   pixel - self.BASE_OFFSET]).astype(int)
                    sample_xys.append(xy)
                    
                # 2. Coordinate tokens: COORD_TOKEN_START <= pixel < COLOR_COORD_BOUNDARY
                elif self.COORD_TOKEN_START <= pixel < self.COLOR_COORD_BOUNDARY:
                    pixel_index = pixel - self.COORD_TOKEN_START
                    if pixel_index in self.pixel2xy:
                        xy = self.pixel2xy[pixel_index] - self.BASE_OFFSET
                        sample_xys.append(xy)
                    
                # 3. Arc parameters: ARC_PARAM_START + 1 <= pixel < ARC_PARAM_START + 1 + ARC_PARAM_RANGE
                elif (self.ARC_PARAM_START + 1 <= pixel < 
                      self.ARC_PARAM_START + 1 + self.ARC_PARAM_RANGE):
                    value = pixel - self.ARC_PARAM_START - 1
                    xy = np.array([value, value]).astype(int)
                    sample_xys.append(xy)
                    
                # 4. Color tokens: COLOR_COORD_BOUNDARY <= pixel < ARC_PARAM_START
                elif self.COLOR_COORD_BOUNDARY <= pixel < self.ARC_PARAM_START:
                    xy = np.array([pixel - self.BASE_OFFSET, 
                                   pixel - self.BASE_OFFSET]).astype(int)
                    sample_xys.append(xy)
                    
            except Exception as e:
                print(f"Error processing pixel {pixel}: {e}")
                continue
        
        if sample_xys:
            return np.vstack(sample_xys)
        else:
            return np.array([]).reshape(0, 2)

    def raster_svg(self, pixels: np.ndarray) -> Tuple[List[List[torch.Tensor]], List[int]]:
        """Convert pixels to SVG tensors following dataset.py logic."""
        try:
            if len(pixels) == 0:
                return [[]], []
            
            # Key step: subtract PIXEL_OFFSET
            pixels = pixels - self.PIXEL_OFFSET
            
            svg_tensors = []
            color_tensors = []
            path_tensor = []
            
            i = 0
            while i < len(pixels):
                try:
                    pix = pixels[i]
                    
                    # Move command
                    if pix[0] == self.CMD_MOVE:
                        if i + 2 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 0  # Move command index
                        cmd_tensor[12:14] = pixels[i+2]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 3
                        
                    # Line command
                    elif pix[0] == self.CMD_LINE:
                        if i + 1 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 1  # Line command index
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                        
                    # Curve command
                    elif pix[0] == self.CMD_CURVE:
                        if i + 3 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 2  # Curve command index
                        cmd_tensor[8:10] = pixels[i+1]
                        cmd_tensor[10:12] = pixels[i+2]
                        cmd_tensor[12:14] = pixels[i+3]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 4
                        
                    # Arc command
                    elif pix[0] == self.CMD_ARC:
                        if i + 5 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 3  # Arc command index
                        radius = pixels[i+1]
                        x_axis_rot = pixels[i+2][0] + self.PIXEL_OFFSET
                        large_arc_flg = pixels[i+3][0] + self.PIXEL_OFFSET
                        sweep_flg = pixels[i+4][0] + self.PIXEL_OFFSET
                        end_pos = pixels[i+5]
                        cmd_tensor[1:3] = radius
                        cmd_tensor[3] = x_axis_rot
                        cmd_tensor[4] = large_arc_flg
                        cmd_tensor[5] = sweep_flg
                        cmd_tensor[12:14] = end_pos
                        path_tensor.append(cmd_tensor.tolist())
                        i += 6
                        
                    # Close command
                    elif pix[0] == self.CMD_CLOSE:
                        if i + 1 >= len(pixels):
                            break
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 6  # Close command index
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                        
                    # Color token: pix[0] >= COLOR_THRESHOLD
                    elif pix[0] >= self.COLOR_THRESHOLD:
                        if path_tensor:
                            svg_tensors.append(torch.tensor(path_tensor))
                            # Reverse transform: restore original color token
                            color_token = int(pix[0] + self.PIXEL_OFFSET - 1)
                            color_tensors.append(color_token)
                            path_tensor = []
                        i += 1
                    else:
                        i += 1
                        
                except (IndexError, TypeError) as e:
                    print(f"Error at position {i}: {e}")
                    break
            
            # Handle remaining path (without color)
            if path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))
                
            return [svg_tensors], color_tensors
            
        except Exception as e:
            print(f"Error in raster_svg: {e}")
            import traceback
            traceback.print_exc()
            return [[]], []

    def apply_colors_to_svg(self, svg_tensors: List[torch.Tensor], 
                           colors: Optional[List[int]]) -> SVG:
        """Apply colors and create final SVG."""
        paths = []
        
        if not svg_tensors:
            raise ValueError("No valid SVG tensors")
        
        colors = colors or []
        
        for i, path_tensor in enumerate(svg_tensors):
            try:
                path = SVGTensor.from_data(path_tensor)
                path = SVG.from_tensor(path.data, viewbox=Bbox(self.BBOX))
                
                actual_color = self.token_to_color(colors[i]) if i < len(colors) else "none"
                
                for path_group in path:
                    path_group.color = actual_color
                    path_group.stroke_color = "none"
                    
                path.fill_(True)
                paths.append(path)
                
            except Exception as e:
                print(f"Error processing path {i}: {e}")
                continue
        
        if not paths:
            raise ValueError("No valid paths generated")
        
        path_groups = paths[0].svg_path_groups
        for i in range(1, len(paths)):
            path_groups.extend(paths[i].svg_path_groups)
        
        return SVG(path_groups, viewbox=Bbox(self.BBOX))