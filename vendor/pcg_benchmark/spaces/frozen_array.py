from pcg_benchmark.spaces.array import ArraySpace
from pcg_benchmark.spaces.space import Space
import numpy as np

"""
An Array space that supports frozen tiles - tiles that cannot be modified during sampling.
This allows for creating partially constrained maps where some tiles are fixed and others
can be freely generated.
"""
class FrozenArraySpace(ArraySpace):
    """
    The constructor of the Frozen Array space.
    
    Parameters:
        dimensions(int[]): is an array or tuple that define the shape of the array.
        value(any): a space that is used to sample every unit of the array
        frozen_mask(np.ndarray, optional): boolean mask indicating which tiles are frozen
    """
    def __init__(self, dimensions, value, frozen_mask=None):
        super().__init__(dimensions, value)
        
        if not hasattr(dimensions, "__len__"):
            dimensions = [dimensions]
        self._dimensions = tuple(dimensions)
        
        # Initialize frozen mask
        if frozen_mask is not None:
            if not isinstance(frozen_mask, np.ndarray):
                frozen_mask = np.array(frozen_mask)
            if frozen_mask.shape != self._dimensions:
                raise ValueError(f"Frozen mask shape {frozen_mask.shape} does not match dimensions {self._dimensions}")
            self._frozen_mask = frozen_mask.copy()
        else:
            self._frozen_mask = np.zeros(self._dimensions, dtype=bool)
            
        # Store frozen values
        self._frozen_values = np.zeros(self._dimensions, dtype=object)
    
    def set_frozen(self, coordinates, value):
        """
        Mark a tile at given coordinates as frozen with a specific value.
        
        Parameters:
            coordinates(tuple): coordinates of the tile to freeze
            value(any): the value to freeze the tile at
        """
        if not isinstance(coordinates, tuple):
            coordinates = tuple(coordinates)
        
        # Validate coordinates
        if len(coordinates) != len(self._dimensions):
            raise ValueError(f"Coordinates {coordinates} do not match dimensions {self._dimensions}")
        
        for i, coord in enumerate(coordinates):
            if not (0 <= coord < self._dimensions[i]):
                raise ValueError(f"Coordinate {coord} out of bounds for dimension {i} (size {self._dimensions[i]})")
        
        # Validate value is within the space range
        if hasattr(self._value, 'contains') and not self._value.contains(value):
            raise ValueError(f"Value {value} is not valid for this space")
        
        self._frozen_mask[coordinates] = True
        self._frozen_values[coordinates] = value
    
    def set_frozen_region(self, coordinates_list, values_list):
        """
        Mark multiple tiles as frozen with their respective values.
        
        Parameters:
            coordinates_list(list): list of coordinate tuples
            values_list(list): list of values corresponding to each coordinate
        """
        if len(coordinates_list) != len(values_list):
            raise ValueError("Coordinates and values lists must have the same length")
        
        for coords, value in zip(coordinates_list, values_list):
            self.set_frozen(coords, value)
    
    def is_frozen(self, coordinates):
        """
        Check if a tile at given coordinates is frozen.
        
        Parameters:
            coordinates(tuple): coordinates to check
            
        Returns:
            bool: True if the tile is frozen, False otherwise
        """
        if not isinstance(coordinates, tuple):
            coordinates = tuple(coordinates)
        
        if len(coordinates) != len(self._dimensions):
            return False
        
        try:
            return self._frozen_mask[coordinates]
        except IndexError:
            return False
    
    def get_frozen_value(self, coordinates):
        """
        Get the frozen value at given coordinates.
        
        Parameters:
            coordinates(tuple): coordinates to get value from
            
        Returns:
            any: the frozen value at those coordinates, or None if not frozen
        """
        if not isinstance(coordinates, tuple):
            coordinates = tuple(coordinates)
        
        if not self.is_frozen(coordinates):
            return None
        
        return self._frozen_values[coordinates]
    
    def sample(self, content=None):
        """
        Sample from the space, respecting frozen tiles.
        
        Parameters:
            content(any, optional): existing content to preserve frozen tiles from
            
        Returns:
            any: sampled content with frozen tiles preserved
        """
        # Generate new sample normally
        new_sample = super().sample()
        
        # If we have existing content, preserve frozen tiles from it
        if content is not None:
            content_array = np.array(content)
            if content_array.shape == self._dimensions:
                # Copy frozen tiles from existing content
                frozen_coords = np.where(self._frozen_mask)
                for i in range(len(frozen_coords[0])):
                    coord_tuple = tuple(frozen_coords[j][i] for j in range(len(frozen_coords)))
                    if self.is_frozen(coord_tuple):
                        self._set_value_at_coords(new_sample, coord_tuple, content_array[coord_tuple])
        
        # Apply explicitly set frozen values
        frozen_coords = np.where(self._frozen_mask)
        for i in range(len(frozen_coords[0])):
            coord_tuple = tuple(frozen_coords[j][i] for j in range(len(frozen_coords)))
            frozen_value = self._frozen_values[coord_tuple]
            if frozen_value is not None:
                self._set_value_at_coords(new_sample, coord_tuple, frozen_value)
        
        return new_sample
    
    def _set_value_at_coords(self, array, coordinates, value):
        """
        Helper method to set a value at specific coordinates in a nested array structure.
        
        Parameters:
            array: the array structure to modify
            coordinates: tuple of coordinates
            value: value to set
        """
        if len(coordinates) == 1:
            array[coordinates[0]] = value
        else:
            self._set_value_at_coords(array[coordinates[0]], coordinates[1:], value)
    
    def sample_with_constraints(self, base_content=None, preserve_frozen=True):
        """
        Sample new content while optionally preserving frozen tiles from base content.
        
        Parameters:
            base_content(any, optional): base content to preserve frozen tiles from
            preserve_frozen(bool): whether to preserve frozen tiles
            
        Returns:
            any: sampled content
        """
        if preserve_frozen and base_content is not None:
            return self.sample(base_content)
        else:
            return self.sample()
    
    def get_frozen_mask(self):
        """
        Get a copy of the frozen mask.
        
        Returns:
            np.ndarray: boolean mask indicating frozen tiles
        """
        return self._frozen_mask.copy()
    
    def get_mutable_mask(self):
        """
        Get a boolean mask indicating which tiles can be modified.
        
        Returns:
            np.ndarray: boolean mask indicating mutable (non-frozen) tiles
        """
        return ~self._frozen_mask
    
    def clear_frozen(self):
        """
        Clear all frozen tiles, making all tiles mutable again.
        """
        self._frozen_mask.fill(False)
        self._frozen_values.fill(None)
    
    def freeze_tiles_by_value(self, content, target_values):
        """
        Freeze all tiles in content that match specific values.
        
        Parameters:
            content(np.ndarray): the content array to analyze
            target_values(list): list of values to freeze
        """
        content_array = np.array(content)
        if content_array.shape != self._dimensions:
            raise ValueError(f"Content shape {content_array.shape} does not match space dimensions {self._dimensions}")
        
        # Find all positions that match target values
        for target_value in target_values:
            positions = np.where(content_array == target_value)
            for i in range(len(positions[0])):
                coord_tuple = tuple(positions[j][i] for j in range(len(positions)))
                self.set_frozen(coord_tuple, target_value)
    
    def freeze_tiles_by_type(self, content, tile_types):
        """
        Freeze all tiles of specific types. This is an alias for freeze_tiles_by_value
        for better semantic clarity when working with tile-based problems.
        
        Parameters:
            content(np.ndarray): the content array to analyze
            tile_types(list): list of tile type values to freeze
        """
        self.freeze_tiles_by_value(content, tile_types)
    
    def freeze_random_tiles(self, probability=0.1, seed=None):
        """
        Randomly freeze tiles based on probability.
        First generates a random map, then freezes tiles randomly based on the probability.
        
        Parameters:
            probability(float): probability of freezing each tile (default 0.1)
            seed(int, optional): random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate a random sample first
        random_content = self.sample()
        random_array = np.array(random_content)
        
        # Create random mask based on probability
        random_mask = np.random.random(self._dimensions) < probability
        
        # Freeze tiles according to the mask
        frozen_coords = np.where(random_mask)
        for i in range(len(frozen_coords[0])):
            coord_tuple = tuple(frozen_coords[j][i] for j in range(len(frozen_coords)))
            tile_value = random_array[coord_tuple]
            self.set_frozen(coord_tuple, tile_value)
    
    def freeze_tiles_at_positions(self, tile_type, positions):
        """
        Freeze specific tile type at given positions.
        
        Parameters:
            tile_type(any): the tile type/value to freeze at positions
            positions(list): list of coordinate tuples where to place the frozen tiles
        """
        for position in positions:
            if not isinstance(position, tuple):
                position = tuple(position)
            self.set_frozen(position, tile_type)
    
    def freeze_all_tiles_of_types(self, content, tile_types):
        """
        Freeze all tiles of specified types in the given content.
        
        Parameters:
            content(np.ndarray): the content array to analyze
            tile_types(list): list of tile type values to freeze
        """
        content_array = np.array(content)
        if content_array.shape != self._dimensions:
            raise ValueError(f"Content shape {content_array.shape} does not match space dimensions {self._dimensions}")
        
        # Find all positions that match target tile types
        for tile_type in tile_types:
            positions = np.where(content_array == tile_type)
            for i in range(len(positions[0])):
                coord_tuple = tuple(positions[j][i] for j in range(len(positions)))
                self.set_frozen(coord_tuple, tile_type)