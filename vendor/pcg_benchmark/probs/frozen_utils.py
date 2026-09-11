import numpy as np
from pcg_benchmark.spaces import FrozenArraySpace, ArraySpace

"""
Utility functions for working with frozen tiles across different problems.
These functions provide common patterns for setting up frozen tile constraints.
All functions are backward compatible - if no frozen constraints are specified,
the problem behaves exactly as before.
"""

def create_frozen_random_problem(problem_class, probability=0.1, seed=None, **kwargs):
    """
    Create a problem instance with randomly frozen tiles.
    
    Parameters:
        problem_class: The problem class to instantiate
        probability: Probability of freezing each tile (default 0.1)
        seed: Random seed for reproducible results
        **kwargs: Additional arguments for the problem constructor
        
    Returns:
        Problem instance with randomly frozen tiles
    """
    problem = problem_class(**kwargs)
    
    # Replace the content space with a frozen version
    original_space = problem._content_space
    if hasattr(original_space, '_dimensions'):
        dimensions = original_space._dimensions
        value_space = original_space._value
    else:
        # Try to infer from the problem's width/height
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = original_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
    
    if dimensions and value_space:
        frozen_space = FrozenArraySpace(dimensions, value_space)
        frozen_space.freeze_random_tiles(probability, seed)
        problem._content_space = frozen_space
    
    return problem

def create_frozen_template_problem(problem_class, template_mask, template_values, **kwargs):
    """
    Create a problem instance with frozen tiles based on a template.
    
    Parameters:
        problem_class: The problem class to instantiate
        template_mask: Boolean mask indicating which tiles to freeze
        template_values: Values for the frozen tiles
        **kwargs: Additional arguments for the problem constructor
        
    Returns:
        Problem instance with template-based frozen tiles
    """
    problem = problem_class(**kwargs)
    
    # Replace the content space with a frozen version
    original_space = problem._content_space
    if hasattr(original_space, '_dimensions'):
        dimensions = original_space._dimensions
        value_space = original_space._value
    else:
        # Try to infer from the problem's width/height
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = original_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
    
    if dimensions and value_space:
        frozen_space = FrozenArraySpace(dimensions, value_space, template_mask)
        
        # Set frozen values based on template
        frozen_coords = np.where(template_mask)
        for i in range(len(frozen_coords[0])):
            coord_tuple = tuple(frozen_coords[j][i] for j in range(len(frozen_coords)))
            frozen_space.set_frozen(coord_tuple, template_values[coord_tuple])
        
        problem._content_space = frozen_space
    
    return problem

def add_frozen_constraints_to_problem(problem, frozen_coordinates, frozen_values):
    """
    Add frozen tile constraints to an existing problem.
    
    Parameters:
        problem: The problem instance to modify
        frozen_coordinates: List of coordinate tuples to freeze
        frozen_values: List of values corresponding to each coordinate
    """
    if not isinstance(problem._content_space, FrozenArraySpace):
        # Convert existing space to FrozenArraySpace
        original_space = problem._content_space
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = original_space._value[0]
            
            frozen_space = FrozenArraySpace(dimensions, value_space)
            problem._content_space = frozen_space
    
    # Add frozen constraints
    if hasattr(problem._content_space, 'set_frozen_region'):
        problem._content_space.set_frozen_region(frozen_coordinates, frozen_values)

def create_frozen_positions_problem(problem_class, tile_type, positions, **kwargs):
    """
    Create a problem instance with specific tile types frozen at given positions.
    
    Parameters:
        problem_class: The problem class to instantiate
        tile_type: The tile type/enum value to freeze at positions
        positions: List of coordinate tuples where to place the frozen tiles
        **kwargs: Additional arguments for the problem constructor
        
    Returns:
        Problem instance with tiles frozen at specified positions
    """
    problem = problem_class(**kwargs)
    
    # Replace the content space with a frozen version
    original_space = problem._content_space
    if hasattr(original_space, '_dimensions'):
        dimensions = original_space._dimensions
        value_space = original_space._value
    else:
        # Try to infer from the problem's width/height
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = original_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
    
    if dimensions and value_space:
        frozen_space = FrozenArraySpace(dimensions, value_space)
        frozen_space.freeze_tiles_at_positions(tile_type, positions)
        problem._content_space = frozen_space
    
    return problem

def create_frozen_tile_types_problem(problem_class, content, tile_types, **kwargs):
    """
    Create a problem instance with all tiles of specified types frozen.
    
    Parameters:
        problem_class: The problem class to instantiate
        content: Reference content to analyze for tile positions
        tile_types: List of tile type/enum values to freeze
        **kwargs: Additional arguments for the problem constructor
        
    Returns:
        Problem instance with all specified tile types frozen
    """
    problem = problem_class(**kwargs)
    
    # Replace the content space with a frozen version
    original_space = problem._content_space
    if hasattr(original_space, '_dimensions'):
        dimensions = original_space._dimensions
        value_space = original_space._value
    else:
        # Try to infer from the problem's width/height
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = original_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
    
    if dimensions and value_space:
        frozen_space = FrozenArraySpace(dimensions, value_space)
        frozen_space.freeze_all_tiles_of_types(content, tile_types)
        problem._content_space = frozen_space
    
    return problem

def create_problem_with_frozen_tiles(problem_class, freeze_tiles=None, **kwargs):
    """
    Create a problem instance with optional frozen tile constraints.
    This function provides full backward compatibility - if no frozen constraints
    are specified, it behaves exactly like calling the problem constructor directly.
    
    Parameters:
        problem_class: The problem class to instantiate
        freeze_tiles: Optional dictionary with freezing options:
            - 'random_probability': Probability for random freezing (default 0.1)
            - 'random_seed': Seed for random freezing (optional)
            - 'tile_type': Tile type to freeze at specific positions
            - 'positions': List of positions for tile_type freezing
            - 'reference_content': Content to analyze for tile type freezing
            - 'tile_types': List of tile types to freeze from reference content
        **kwargs: Additional arguments for the problem constructor
        
    Returns:
        Problem instance with optional frozen constraints
    """
    # Create the problem normally first
    problem = problem_class(**kwargs)
    
    # If no freezing options specified, return the problem as-is (backward compatible)
    if freeze_tiles is None:
        return problem
    
    # Extract dimensions for creating FrozenArraySpace
    dimensions = None
    value_space = None
    
    if hasattr(problem, '_width') and hasattr(problem, '_height'):
        dimensions = (problem._height, problem._width)
        # Extract the value space from the original ArraySpace
        if hasattr(problem._content_space, '_value') and hasattr(problem._content_space._value, '__getitem__'):
            # Navigate through the nested structure to get the actual Space object
            value_space = problem._content_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
    
    # Only replace with FrozenArraySpace if we have valid dimensions and any freezing options
    if dimensions and value_space and any(key in freeze_tiles for key in 
                                          ['random_probability', 'tile_type', 'tile_types']):
        
        # Create FrozenArraySpace
        frozen_space = FrozenArraySpace(dimensions, value_space)
        
        # Apply different freezing strategies
        if 'random_probability' in freeze_tiles:
            probability = freeze_tiles['random_probability']
            seed = freeze_tiles.get('random_seed', None)
            frozen_space.freeze_random_tiles(probability, seed)
        
        if 'tile_type' in freeze_tiles and 'positions' in freeze_tiles:
            frozen_space.freeze_tiles_at_positions(freeze_tiles['tile_type'], freeze_tiles['positions'])
        
        if 'tile_types' in freeze_tiles and 'reference_content' in freeze_tiles:
            frozen_space.freeze_all_tiles_of_types(freeze_tiles['reference_content'], freeze_tiles['tile_types'])
        
        problem._content_space = frozen_space
    
    return problem

def freeze_existing_tiles_by_type(problem, reference_content, tile_types):
    """
    Freeze tiles of specific types in an existing problem based on reference content.
    
    Parameters:
        problem: The problem instance to modify
        reference_content: Content to analyze for tile positions
        tile_types: List of tile type values to freeze
    """
    # Ensure we have a FrozenArraySpace
    if not isinstance(problem._content_space, FrozenArraySpace):
        # Convert to FrozenArraySpace
        if hasattr(problem, '_width') and hasattr(problem, '_height'):
            dimensions = (problem._height, problem._width)
            # Extract the value space from the original ArraySpace
            value_space = problem._content_space._value[0]
            if hasattr(value_space, '__getitem__'):
                value_space = value_space[0]
            problem._content_space = FrozenArraySpace(dimensions, value_space)
    
    # Freeze tiles by type using the new method
    problem._content_space.freeze_all_tiles_of_types(reference_content, tile_types)

def get_backward_compatible_space(original_space, freeze_options=None):
    """
    Get a space that is backward compatible - either the original space or a FrozenArraySpace.
    
    Parameters:
        original_space: The original space
        freeze_options: Optional freezing options
        
    Returns:
        Space: Either the original space or a FrozenArraySpace
    """
    if freeze_options is None:
        return original_space
    
    # If it's already a FrozenArraySpace, return it
    if isinstance(original_space, FrozenArraySpace):
        return original_space
    
    # Convert ArraySpace to FrozenArraySpace if needed
    if isinstance(original_space, ArraySpace):
        # Extract dimensions and value space
        # This is a simplified extraction - might need adjustment based on actual ArraySpace structure
        if hasattr(original_space, '_value') and hasattr(original_space._value, '__getitem__'):
            dimensions = len(original_space._value)
            value_space = original_space._value[0]
            return FrozenArraySpace((dimensions,), value_space)
    
    return original_space