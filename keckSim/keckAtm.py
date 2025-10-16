import numpy as np
from OOPAO.Atmosphere import Atmosphere

class keckAtm:
    """
    Base class for Keck atmosphere configurations. Subclasses
    should set up self.atm with an OOPAO Atmosphere instance
    and then call atm.initializeAtmosphere().
    """
    def __init__(self, telescope, atm_type):
        self.telescope = telescope
        self.atm_type = atm_type
        self.atm = None  # Will hold the OOPAO Atmosphere instance

    @classmethod
    def create(cls, telescope, atm_type='maunakea', **kwargs):
        """
        Factory method to create a specialized atmosphere instance.
        
        Usage:
            atm_obj = keckAtm.create(telescope=myTelescope, atm_type='maunakea')
            # Then access the OOPAO Atmosphere:
            my_atm = atm_obj.atm
        """
        atm_type = atm_type.lower()
        if atm_type == 'maunakea':
            return maunaKeaAtm(telescope, **kwargs)
        elif atm_type == 'ground':
            return groundAtm(telescope, **kwargs)
        elif atm_type == 'custom':
            return customAtm(telescope, **kwargs)
        else:
            valid_types = ['maunakea', 'ground', 'custom']
            raise ValueError(f"Invalid atmosphere type '{atm_type}'. "
                             f"Choose from {valid_types}.")

class maunaKeaAtm(keckAtm):
    """
    Represents a detailed multi-layer atmospheric profile
    for Mauna Kea.
    """
    def __init__(self,
                 telescope,
                 r0=0.16,
                 L0=50.0,
                 altitudes=np.array([0, 0.5,1,2,4,8,16])*1e3,
                 fractional_r0=[0.517, 0.119, 0.063, 0.061, 0.105,0.081,0.054],
                 wind_speeds=[6.8, 6.9,7.1,7.5,10.0,26.9,18.5],
                 wind_directions=[0, np.pi/2, np.pi/4, 3*np.pi/2, 8*np.pi/3,np.pi/8, np.pi]
                 ):
        super().__init__(telescope, 'maunakea')
        # Create the Atmosphere object
        self.atm = Atmosphere(
            telescope=self.telescope,
            r0=r0,
            L0=L0,
            fractionalR0=fractional_r0,
            altitude=altitudes,
            windSpeed=wind_speeds,
            windDirection=wind_directions
        )
        # Initialize the atmosphere
        self.atm.initializeAtmosphere(telescope=self.telescope)

class groundAtm(keckAtm):
    """
    Simple single-layer 'ground-layer' atmosphere.
    """
    def __init__(self,
                 telescope,
                 r0=0.2,
                 L0=25.0,
                 wind_speed=10,
                 wind_direction=0
                 ):
        super().__init__(telescope, 'ground')
        # Create the Atmosphere object (single layer at altitude=0)
        self.atm = Atmosphere(
            telescope=self.telescope,
            r0=r0,
            L0=L0,
            fractionalR0=[1.0],
            altitude=[0],
            windSpeed=[wind_speed],
            windDirection=[wind_direction]
        )
        # Initialize the atmosphere
        self.atm.initializeAtmosphere(telescope=self.telescope)

class customAtm(keckAtm):
    """
    An example of a 'custom' multi-layer atmosphere.
    Change defaults as desired.
    """
    def __init__(self,
                 telescope,
                 r0=0.15,
                 L0=30.0,
                 altitudes=[0, 5000],
                 fractional_r0=[0.7, 0.3],
                 wind_speeds=[5, 20],
                 wind_directions=[0, np.pi/2]
                 ):
        super().__init__(telescope, 'custom')
        self.atm = Atmosphere(
            telescope=self.telescope,
            r0=r0,
            L0=L0,
            fractionalR0=fractional_r0,
            altitude=altitudes,
            windSpeed=wind_speeds,
            windDirection=wind_directions
        )
        self.atm.initializeAtmosphere(telescope=self.telescope)


# Example Usage
if __name__ == "__main__":
    
    from OOPAO.Telescope import Telescope
    from OOPAO.Source import Source
    # --- Source object ---
    src = Source(magnitude=10, optBand='I')  # Natural Guide Star in I band of magnitude 5
    # Create a mock telescope
    tel = Telescope(diameter=10.93, resolution=160, samplingTime=1/1000)
    # Couple the source to the telescope
    src * tel

    # Create a Mauna Kea atmosphere
    mk_atm_obj = keckAtm.create(telescope=tel, atm_type='maunakea')
    mk_atm = mk_atm_obj.atm
    print("Mauna Kea atmosphere created with layers:")
    print("Altitude:", mk_atm.altitude)
    print("Fractional r0:", mk_atm.fractionalR0)

    # Create a ground-layer atmosphere
    ground_atm_obj = keckAtm.create(telescope=tel, atm_type='ground', wind_speed=12)
    print("Ground-layer atmosphere wind speed:", ground_atm_obj.atm.windSpeed)

    # Create a custom atmosphere
    custom_atm_obj = keckAtm.create(telescope=tel, atm_type='custom')
    print("Custom atmosphere altitudes:", custom_atm_obj.atm.altitude)
