"""
Python implementation of Sparse Distributed Memory, a computational model of human 
memory introduced by neuroscientist Pentti Kanerva. 

This module implements the fundamental operations of Kanerva's Sparse Distributed
Memory (SDM) model, including writing, reading, and erasing memories, based on 
Hamming distance activation. 

Reference:
    Pentti Kanerva (1992). Sparse Distributed Memory and Related Models.

(c) 2025 Simon Wong
"""

import numpy as np 

class KanervaSDM: 
    """
    This class provides fundamental SDM functionality for storing and recalling memories. 
    Single letters in parentheses (e.g. (M) for number of locations) indicate notation 
    from Kanerva's original work. 

    """

    def __init__(self, 
                 ADDRESS_DIMENSION: int, 
                 MEMORY_DIMENSION: int, 
                 NUM_LOCATIONS: int, 
                 HAMMING_THRESHOLD: int, 
                 RANDOM_SEED:int = 42
                 ) -> None:
        """
        Initializes the Kanerva SDM.

        Args:
            ADDRESS_DIMENSION: Length of address vectors (N).
            MEMORY_DIMENSION: Length of memory vectors (U).
            NUM_LOCATIONS: Number of hard locations (M).
            HAMMING_THRESHOLD: Hamming distance threshold for activation (H).
            RANDOM_SEED: Seed for reproducible random generation of hard locations. 

        Raises:
            ValueError: If any dimension or threshold is non-positive.
        """
        if ADDRESS_DIMENSION <= 0 or MEMORY_DIMENSION <= 0 or NUM_LOCATIONS <= 0:
            raise ValueError("All dimensions must be positive integers.")
        if HAMMING_THRESHOLD < 0:
            raise ValueError("Activation threshold must be non-negative.")
        
        self.ADDRESS_DIMENSION = int(ADDRESS_DIMENSION)  # Length of addresses (N). 
        self.MEMORY_DIMENSION = int(MEMORY_DIMENSION)  # Length of memories (U). 
        self.NUM_LOCATIONS = int(NUM_LOCATIONS)  # Number of locations (M). 
        self.HAMMING_THRESHOLD = int(HAMMING_THRESHOLD)  # Hamming activation threshold (H). 

        rng = np.random.default_rng(RANDOM_SEED)

        self.address_matrix = rng.integers(
            0, 2, 
            size=(self.NUM_LOCATIONS, self.ADDRESS_DIMENSION), 
            dtype=np.int8
        )
        
        self.memory_matrix = np.zeros(
            (self.NUM_LOCATIONS, self.MEMORY_DIMENSION), 
            dtype=np.float32)  
         
        self.memory_count = 0  # Number of stored memories (T). 

    def _get_activated_locations(self, address: np.ndarray) -> np.ndarray:
        """
        Finds activated locations based on Hamming distance threshold (H). 

        Args:
            address: Target address vector (x) of shape (ADDRESS_DIMENSION,).

        Returns:
            Array of indices for activated locations (y).

        Raises:
            ValueError: If address shape doesn't match ADDRESS_DIMENSION.
        """
        hamming_distances = np.count_nonzero(self.address_matrix != address, axis=1)  # Vectorized Hamming distance. 
        return np.where(hamming_distances <= self.HAMMING_THRESHOLD)[0] 
    
    def _validate__vector(self, vector: np.ndarray, vector_name: str) -> None: 
        """
        Validates that an address vector or memory vector has the correct dimension 
        and contains only binary values. 

        
        Args:
            vector: Vector to validate.
            vector_name: Name of the vector for error message (either "address" or "memory"). 
        
        Raises:
            ValueError: If vector dimension is incorrect or contains non-binary values.
        """
        if vector_name == "address": 
            expected_dimension = self.ADDRESS_DIMENSION
        elif vector_name == "memory": 
            expected_dimension = self.MEMORY_DIMENSION

        if vector.shape != (expected_dimension,):
            raise ValueError(
                f"{vector_name} shape {vector.shape} doesn't match "
                f"expected ({expected_dimension},)"
            )
        
        if not np.all(np.isin(vector, [0, 1])):
            raise ValueError(f"{vector_name} must contain only 0s and 1s")
        
    def write(self, address: np.ndarray, memory: np.ndarray) -> None: 
        """
        Writes a memory to an address. 

        Args:
            address: Target address vector (x) of shape (ADDRESS_DIMENSION,).
            memory: Memory vector (w) of shape (MEMORY_DIMENSION,). 

        Raises:
            ValueError: If address or memory vectors are invalid. 

        """
        self._validate__vector(address, "address")
        self._validate__vector(memory, "memory")

        activated_locations = self._get_activated_locations(address)  # Activation vector (y). 

        polar_memory = 2 * memory - 1  # Convert memory (0 and 1) to polar memory (-1 and +1). 
        self.memory_matrix[activated_locations] += polar_memory  # Add or subtract one to activated locations in memory matrix (C). 
        self.memory_count += 1  # Increment number of stored memories (T). 

    def read(self, address: np.ndarray) -> np.ndarray: 
        """
        Reads a memory from an address. 

        Args:
            address: Target address vector (x) of shape (ADDRESS_DIMENSION,).

        Returns:
            Recalled memory vector (z) of shape (MEMORY_DIMENSION,).
            Returns all zeros if no locations are activated.

        Raises:
            ValueError: If address vector is invalid. 
        """
        self._validate__vector(address, "address")

        activated_locations = self._get_activated_locations(address)  # Activation vector (y). 

        if len(activated_locations) == 0:  # Failsafe in case no locations are activated. 
            return np.zeros(self.MEMORY_DIMENSION, dtype=np.uint8)
        
        locations_sum = self.memory_matrix[activated_locations].sum(axis=0)  # Sum all activated locations in memory matrix (s).  
        return (locations_sum >= 0).astype(np.int8)  # Memory vector (z) is binary vector of all location sum entries that are greater than zero. 
    
    def erase_memory(self) -> None: 
        """
        Erases memory matrix (C), but NOT address matrix (A), 
        so locations are preserved. 
        """
        self.memory_matrix.fill(0)
        self.memory_count = 0 