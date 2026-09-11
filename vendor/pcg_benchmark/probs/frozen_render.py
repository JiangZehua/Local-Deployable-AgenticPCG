"""
Rendering utilities for frozen tiles visualization.
Provides functions to overlay frozen tiles with transparent blue color.
"""

import numpy as np
from PIL import Image, ImageDraw
from pcg_benchmark.spaces import FrozenArraySpace

def create_frozen_overlay(image, frozen_mask, scale=16, opacity=0.3):
    """
    Create a transparent blue overlay for frozen tiles.
    
    Parameters:
        image: PIL Image to overlay on
        frozen_mask: Boolean mask indicating frozen tiles
        scale: Scale factor for each tile (default 16)
        opacity: Opacity of the overlay (0.0 to 1.0, default 0.3)
        
    Returns:
        PIL Image with frozen tile overlay
    """
    # Create a blue overlay image
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Blue color with transparency
    blue_color = (0, 100, 255, int(255 * opacity))
    
    # Draw rectangles for frozen tiles
    for y in range(frozen_mask.shape[0]):
        for x in range(frozen_mask.shape[1]):
            if frozen_mask[y, x]:
                # Calculate pixel coordinates
                left = x * scale
                top = y * scale
                right = left + scale
                bottom = top + scale
                
                # Draw the blue rectangle
                draw.rectangle([left, top, right, bottom], fill=blue_color)
    
    # Composite the overlay onto the original image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    result = Image.alpha_composite(image, overlay)
    return result

def render_with_frozen_tiles(problem, content, info=None, scale=16, frozen_opacity=0.3, frozen_mask=None):
    """
    Render content with frozen tiles highlighted in blue.

    Parameters:
        problem: Problem instance
        content: Content to render
        info: Optional info dictionary (for optimization)
        scale: Scale factor for rendering
        frozen_opacity: Opacity of frozen tile overlay (0.0 to 1.0)
        frozen_mask: Optional boolean mask indicating frozen tiles. If provided,
                     this takes precedence over problem's _content_space.

    Returns:
        PIL Image with frozen tiles overlaid
    """
    # First, render the content normally
    if hasattr(problem, 'render'):
        if info is not None:
            base_image = problem.render(content, info)
        else:
            base_image = problem.render(content)
    else:
        raise ValueError("Problem does not have a render method")

    # Get frozen mask - prefer explicit parameter, then check problem's content space
    mask_to_use = None
    if frozen_mask is not None:
        mask_to_use = np.array(frozen_mask) if not isinstance(frozen_mask, np.ndarray) else frozen_mask
    elif hasattr(problem, '_content_space') and isinstance(problem._content_space, FrozenArraySpace):
        mask_to_use = problem._content_space.get_frozen_mask()

    # If there are frozen tiles, add the overlay
    if mask_to_use is not None and mask_to_use.any():
        # Account for padding in the rendered image
        # Many problems add 1-pixel padding around the content
        padded_mask = np.pad(mask_to_use, 1, constant_values=False)
        base_image = create_frozen_overlay(base_image, padded_mask, scale, frozen_opacity)

    return base_image

def create_frozen_visualization_legend(scale=16):
    """
    Create a legend image showing what frozen tiles look like.
    
    Parameters:
        scale: Scale factor for the legend tiles
        
    Returns:
        PIL Image showing the legend
    """
    # Create a small legend image
    legend_width = scale * 6
    legend_height = scale * 2
    legend = Image.new('RGBA', (legend_width, legend_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    # Draw a normal tile
    draw.rectangle([0, 0, scale, scale], fill=(200, 200, 200, 255), outline=(0, 0, 0, 255))
    draw.text((scale//4, scale//4), "Normal", fill=(0, 0, 0, 255))
    
    # Draw a frozen tile
    draw.rectangle([scale * 2, 0, scale * 3, scale], fill=(200, 200, 200, 255), outline=(0, 0, 0, 255))
    # Add blue overlay
    draw.rectangle([scale * 2, 0, scale * 3, scale], fill=(0, 100, 255, 76))  # 30% opacity
    draw.text((scale * 2 + scale//4, scale//4), "Frozen", fill=(0, 0, 0, 255))
    
    return legend

def compare_frozen_vs_normal(problem, content, info=None, scale=16):
    """
    Create a side-by-side comparison of normal vs frozen tile rendering.
    
    Parameters:
        problem: Problem instance
        content: Content to render
        info: Optional info dictionary
        scale: Scale factor for rendering
        
    Returns:
        PIL Image showing normal and frozen versions side by side
    """
    # Render normal version
    if hasattr(problem, 'render'):
        if info is not None:
            normal_image = problem.render(content, info)
        else:
            normal_image = problem.render(content)
    else:
        raise ValueError("Problem does not have a render method")
    
    # Render frozen version
    frozen_image = render_with_frozen_tiles(problem, content, info, scale)
    
    # Create side-by-side comparison
    total_width = normal_image.width + frozen_image.width + 20  # 20px spacing
    total_height = max(normal_image.height, frozen_image.height) + 40  # 40px for labels
    
    comparison = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 255))
    
    # Paste images
    comparison.paste(normal_image, (10, 30))
    comparison.paste(frozen_image, (normal_image.width + 20, 30))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Normal", fill=(0, 0, 0, 255))
    draw.text((normal_image.width + 20, 10), "With Frozen Tiles", fill=(0, 0, 0, 255))
    
    return comparison

def render_frozen_mask_only(frozen_mask, scale=16):
    """
    Render only the frozen mask as a visualization.
    
    Parameters:
        frozen_mask: Boolean mask indicating frozen tiles
        scale: Scale factor for rendering
        
    Returns:
        PIL Image showing only the frozen tiles
    """
    width = frozen_mask.shape[1] * scale
    height = frozen_mask.shape[0] * scale
    
    image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    for y in range(frozen_mask.shape[0]):
        for x in range(frozen_mask.shape[1]):
            left = x * scale
            top = y * scale
            right = left + scale
            bottom = top + scale
            
            if frozen_mask[y, x]:
                # Frozen tile - blue
                draw.rectangle([left, top, right, bottom], fill=(0, 100, 255, 255), outline=(0, 0, 0, 255))
            else:
                # Normal tile - light gray
                draw.rectangle([left, top, right, bottom], fill=(240, 240, 240, 255), outline=(200, 200, 200, 255))
    
    return image