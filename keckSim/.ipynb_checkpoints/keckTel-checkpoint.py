import numpy as np
from OOPAO.Telescope import Telescope
from hcipy import ( make_hexagonal_grid,
                    hexagonal_aperture, 
                    make_segmented_aperture, 
                    make_spider,
                    Field,
                    make_pupil_grid,
                    circular_aperture,
                    evaluate_supersampled)
import matplotlib.pyplot as plt

class keckTel:
    """
    Base class for Keck telescope configurations.
    
    This class provides a single factory-like interface (`create`)
    for instantiating specialized telescope objects that use OOPAO’s
    Telescope under the hood.
    """
    def __init__(self, telescope_type):
        """
        You typically won't instantiate `keckTel` directly.
        Instead, call `keckTel.create(...)`.
        """
        self.telescope_type = telescope_type
        self.tel = None  # Will hold the OOPAO Telescope instance

    @property
    def pupil(self):
        """Easy access to the pupil property of the OOPAO Telescope."""
        return self.tel.pupil if self.tel else None

    @property
    def OPD(self):
        """Easy access to the OPD property of the OOPAO Telescope."""
        return self.tel.OPD if self.tel else None

    @property
    def PSF(self):
        """Easy access to the PSF property of the OOPAO Telescope."""
        return self.tel.PSF if self.tel else None

    @classmethod
    def create(cls, telescope_type='keck', resolution=256, samplingTime = 0.001, **kwargs):
        """
        Factory method to create a specialized Keck telescope instance.
        
        Usage example:
            keck = keckTel.create('keck', n_subaperture=20, diameter=10.93)
            # Then access OOPAO telescope:
            tel = keck.tel
        """
        telescope_type = telescope_type.lower()
        if telescope_type == 'keck':
            return keckStandard(resolution = resolution, 
                                samplingTime = samplingTime,
                                **kwargs)
        else:
            valid_types = ['keck']
            raise ValueError(f"Invalid telescope type '{telescope_type}'. "
                             f"Choose from {valid_types}.")

class keckStandard(keckTel):
    """
    A class that sets up the OOPAO Telescope object with default
    parameters appropriate for a ~10-meter Keck configuration,
    including a pupil mask from keckAperture.
    """
    def __init__(self,
                 resolution=256,
                 samplingTime = 0.001,
                 **kwargs):
        """
        Args:
            diameter (float): Telescope diameter in meters.
            n_subaperture (int): Number of subapertures across pupil (used to derive resolution).
            samplingTime (float): Sampling time of the AO loop in seconds.
            obs_ratio (float or None): Central obstruction ratio. If None, not applied.
            make_pupil (bool): If True, generate a Keck pupil mask using `make_keck_aperture()`.
            **kwargs: Additional parameters passed to the OOPAO `Telescope` constructor.
        """
        super().__init__("keck")
        
        #diameter of long side hexagonal element in meters
        D_subaperture_l = 1.8
        #diameter of short side hexagonal element in meters
        D_subaperture_s = D_subaperture_l * np.sqrt(3) / 2
        #mirror gap = 3 mm
        mirror_gap = 3e-3
        #Total mirror diameter
        diameter = 7 * D_subaperture_s + 6 * mirror_gap

        grid = make_pupil_grid(resolution, diameter=diameter)
        supersampling_factor = 1
        #keck_pupil = self.make_keck_aperture()(grid).shaped.astype(bool)
        pupil,segments = self.make_keck_aperture(**kwargs)
        keck_pupil = pupil(grid).shaped.astype(bool)
        self.keck_segments = evaluate_supersampled(segments, grid,supersampling_factor)
        self.samplingTime = samplingTime
        
        # Build the OOPAO Telescope object
        self.tel = Telescope(
            diameter=diameter,
            resolution=resolution,
            samplingTime=samplingTime,
            pupil=keck_pupil)
            #**kwargs
        #)
    
    def make_keck_aperture(with_secondary=True, with_spiders=True, return_segments=False):
        ''' Generates the Keck aperture.

        Parameters
        ----------
        with_secondary : Boolean

        with_spiders : Boolean

        return_segments : Boolean

        Returns
        -------
        func : Field

        segments:

        '''
        # Keck II aperture

        #diameter of long side hexagonal element in meters
        D_subaperture_l = 1.8

        #diameter of short side hexagonal element in meters
        D_subaperture_s = D_subaperture_l * np.sqrt(3) / 2

        #mirror gap = 3 mm
        mirror_gap = 3e-3

        #Secondary diameter
        D_obscuration = 2.6 # meter

        D_spider = 25.4e-3 #spiders are 1 inch wide

        #Total mirror diameter
        D_tot = 7 * D_subaperture_s + 6 * mirror_gap

        #the edges of the mirror are non-reflective, removing 2mm, i.e. 4 mm from every diameter
        D_subaperture_l = 1.8 - 4e-3
        D_subaperture_s = D_subaperture_l * np.sqrt(3) / 2

        #Creating a hexagonal grid for adding the mirrors
        n_rings = 3
        circum_diameter = D_tot / (2 * n_rings + 1)
        segment_positions = make_hexagonal_grid(circum_diameter, n_rings, pointy_top=True)

        # removing the central segment
        segment_positions = segment_positions.subset(segment_positions.x**2 + segment_positions.y**2>0)

        #Creating mirror segments:
        segment = hexagonal_aperture(D_subaperture_l)

        #Adding mirror segments:
        if return_segments:
            segmented_aperture, segments = make_segmented_aperture(segment, segment_positions, return_segments=True)
        else:
            segmented_aperture = make_segmented_aperture(segment, segment_positions)

        #Adding Spiders
        spider_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)+np.radians(30)

        secondary_obscuration = circular_aperture(D_obscuration)

        def func(grid):
            res = segmented_aperture(grid)

            if with_secondary:
                res *=  1 - secondary_obscuration(grid)

            if with_spiders:
                for angle in spider_angles:
                    x = D_tot * np.cos(angle)
                    y = D_tot * np.sin(angle)

                    res *= Field(make_spider((0, 0), (x, y), D_spider)(grid),grid)

            return Field(res, grid)

        if return_segments:
            return func, segments
        else:
            return func

# Example usage:
if __name__ == "__main__":
    # Create a Keck telescope using default parameters
    keck = keckTel.create('keck')

    # We now have an OOPAO Telescope instance in `keck.tel`
    print(f"Keck telescope sampling time: {keck.samplingTime}")
    print(f"Keck telescope resolution: {keck.tel.resolution}")
    w = keck.tel.pixelSize * keck.tel.resolution
    plt.imshow(keck.pupil, 
               cmap='inferno', 
               extent=[-w/2,w/2, 
                       -w/2,w/2])
    plt.xlabel("Pupil X [m]", size = 16)
    plt.ylabel("Pupil Y [m]", size = 16)
    plt.show()

    # You can override defaults:
    custom_keck = keckTel.create(
        'keck',
        resolution=64,
        samplingTime=1/1000
    )

    w = custom_keck.tel.pixelSize * custom_keck.tel.resolution
    plt.imshow(custom_keck.pupil, 
               cmap='inferno', 
               extent=[-w/2,w/2, 
                       -w/2,w/2])
    plt.xlabel("Pupil X [m]", size = 16)
    plt.ylabel("Pupil Y [m]", size = 16)
    plt.show()

    # We now have an OOPAO Telescope instance in `keck.tel`
    print(f"Custom Keck telescope sampling time: {keck.samplingTime}")
    print(f"Custom Keck telescope resolution: {keck.tel.resolution}")