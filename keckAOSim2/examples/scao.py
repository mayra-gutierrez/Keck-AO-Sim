from keckSim.keckWfs import keckWfs
from keckSim.keckDm import keckDm
from keckSim.keckAtm import keckAtm
from keckSim.keckTel import keckTel
from OOPAO.Source import Source

# Create a Keck telescope using the factory method
keck_telescope = keckTel.create('keck', resolution=256, samplingTime=1/1000)

# Create a Source object
src = Source(magnitude=10, optBand='I')  # Natural Guide Star in I band of magnitude 10

# Couple the source to the telescope
src * keck_telescope.tel

# Instantiate the SCAO WFS
scao_wfs = keckWfs.create(keck_telescope.tel, 'scao')

# Instantiate the Xinetics DM
xinetics_dm = keckDm.create(keck_telescope.tel, 'xinetics')

# Instantiate the Mauna Kea Atmosphere
mauna_kea_atm = keckAtm.create(keck_telescope.tel, 'maunakea')

# Print some properties to verify the setup
print(f"SCAO WFS nSubap: {scao_wfs.wfs.nSubap}")
print(f"Xinetics DM pitch: {xinetics_dm.dm.pitch * 1e3} mm")
print("Mauna Kea atmosphere altitudes:", mauna_kea_atm.atm.altitude)