import numpy as np
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Telescope import Telescope

class keckDm:
    """
    Base class for Keck deformable mirrors. Subclasses should set up
    'self.dm' with the appropriate DeformableMirror parameters.
    """

    def __init__(self, telescope, dm_type):
        """
        Base constructor for keckDm. Typically, you'll never call this
        directly; the 'create' factory method calls the appropriate
        subclass.
        """
        self.telescope = telescope
        self.dm_type = dm_type
        self.dm = None

    @classmethod
    def create(cls, telescope, dm_type, **kwargs):
        """
        Factory method that dispatches to one of the subclasses based on dm_type.
        
        Usage:
            dm = keckDm.create(telescope, 'haka', print_properties=True)
        """
        dm_type = dm_type.lower()
        if dm_type == 'haka':
            return hakaDm(telescope, **kwargs)
        elif dm_type == 'xinetics':
            return xineticsDm(telescope, **kwargs)
        elif dm_type == 'kasm':
            return kasmDm(telescope, **kwargs)
        else:
            valid_types = ['haka', 'xinetics', 'kasm']
            raise ValueError(f"Invalid DM type '{dm_type}'. Choose from {valid_types}.")

class hakaDm(keckDm):
    """Haka-style DM configuration using Fried geometry."""
    def __init__(self, telescope, print_properties=False):
        super().__init__(telescope, 'haka')
        self.dm = DeformableMirror(
            telescope=telescope,
            nSubap=60,          # 40 subapertures across diameter
            mechCoupling=0.1458,  # Mechanical coupling coefficient
            pitch=0.1825,       # Actuator spacing in meters (0.5 mm)
            coordinates=None,   # Use default Fried geometry
            modes=None,         # Default Gaussian influence
            print_dm_properties=print_properties,
            nJobs=30,
            nThreads=20
        )

class xineticsDm(keckDm):
    """Xinetics-style DM configuration with custom parameters."""
    def __init__(self, telescope, print_properties=False):
        super().__init__(telescope, 'xinetics')
        self.dm = DeformableMirror(
            telescope=telescope,
            nSubap=21,          # 32 subapertures across diameter
            mechCoupling=0.1458,  # Lower mechanical coupling
            pitch=0.5475,         # Larger actuator spacing (1 mm)
            coordinates=None,
            modes=None,
            print_dm_properties=print_properties,
            nJobs=30,
            nThreads=20
        )

class kasmDm(keckDm):
    """KASM-style DM configuration with custom geometry."""
    def __init__(self, telescope, print_properties=False):
        super().__init__(telescope, 'kasm')
        self.dm = DeformableMirror(
            telescope=telescope,
            nSubap=48,          # 48 subapertures across diameter
            mechCoupling=0.4,   # Higher mechanical coupling
            pitch=0.8e-3,       # Medium actuator spacing (0.8 mm)
            coordinates=None,
            modes=None,
            print_dm_properties=print_properties,
            nJobs=30,
            nThreads=20
        )

# Example usage:
if __name__ == "__main__":
    # Create a Telescope object (parameters depend on your system)
    tel = Telescope(resolution=100, diameter=10, centralObstruction=0.15)
    
    # Create different DM types using the factory method
    haka = keckDm.create(tel, 'haka')
    xinetics = keckDm.create(tel, 'xinetics')
    kasm = keckDm.create(tel, 'kasm')

    # Demonstrate properties
    print(f"Haka DM nAct: {haka.dm.nAct}")
    print(f"Xinetics pitch: {xinetics.dm.pitch*1e3} mm")
    print(f"KASM mechanical coupling: {kasm.dm.mechCoupling}")
