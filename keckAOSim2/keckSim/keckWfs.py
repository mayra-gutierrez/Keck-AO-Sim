import numpy as np
from OOPAO.ShackHartmann import ShackHartmann

class keckWfs:
    """
    Base class for Keck wavefront sensors. Subclasses should set up
    'self.wfs' with the appropriate ShackHartmann parameters.
    """

    def __init__(self, telescope, wfs_type):
        """
        Base constructor for keckWfs. Typically, you'll never call this
        directly; the 'create' factory method calls the appropriate
        subclass.
        """
        self.telescope = telescope
        self.wfs_type = wfs_type
        self.wfs = None

    @classmethod
    def create(cls, telescope, wfs_type, **kwargs):
        """
        Factory method that dispatches to one of the subclasses based on wfs_type.
        
        Usage:
            wfs = keckWfs.create(telescope, 'scao', print_properties=True)
        """
        wfs_type = wfs_type.lower()
        if wfs_type == 'scao':
            return scaoWfs(telescope, **kwargs)
        elif wfs_type == 'kapa':
            return kapaWfs(telescope, **kwargs)
        else:
            valid_types = ['scao', 'kapa']
            raise ValueError(f"Invalid WFS type '{wfs_type}'. Choose from {valid_types}.")

class scaoWfs(keckWfs):
    """SCAO system using a single Shack-Hartmann sensor."""
    def __init__(self, telescope, print_properties=False):
        super().__init__(telescope, 'scao')
        self.wfs = ShackHartmann(
            nSubap=20,          # 20 subapertures across diameter
            telescope=telescope,
            lightRatio=0.5,     # Example light ratio
            threshold_cog=0.01 # Center of gravity threshold
        )

class kapaWfs(keckWfs):
    """KAPA system using four identical Shack-Hartmann sensors."""
    def __init__(self, telescope, print_properties=False):
        super().__init__(telescope, 'kapa')
        self.wfs = [ShackHartmann(
            nSubap=20,          # 20 subapertures across diameter
            telescope=telescope,
            lightRatio=0.5,     # Example light ratio
            threshold_cog=0.01 # Center of gravity threshold
        ) for _ in range(4)]

# Example usage:
if __name__ == "__main__":
    
    from OOPAO.Telescope import Telescope
    from OOPAO.Source import Source
    # --- Source object ---
    src = Source(magnitude=10, optBand='I')  # Natural Guide Star in I band of magnitude 5
    # Create a mock telescope
    tel = Telescope(diameter=10.93, resolution=160, samplingTime=1/1000)
    # Couple the source to the telescope
    src * tel

    # Create different WFS types using the factory method
    scao = keckWfs.create(tel, 'scao')
    kapa = keckWfs.create(tel, 'kapa')

    # Demonstrate properties
    print(f"SCAO WFS nSubap: {scao.wfs.nSubap}")
    print(f"KAPA WFS count: {len(kapa.wfs)}")