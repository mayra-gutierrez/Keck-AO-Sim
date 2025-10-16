# Keck Adaptive Optics Simulation Library

Welcome to the Keck Adaptive Optics Simulation Library, an open-source project designed to simulate the components of the Keck adaptive optics (AO) system using the OOPAO simulation library. 
This library provides a comprehensive suite of tools to model and analyze the performance of AO systems, including telescope apertures, AO modes, wavefront sensors, and detectors.

## Features

- **Telescope Apertures**: Simulate the Keck telescope's segmented aperture with realistic parameters.
- **Adaptive Optics Modes**: Implementations of various AO modes such as KAPA, HAKA, and SCAO.
- **Wavefront Sensors**: Models for different types of wavefront sensors used in AO systems.
- **Deformable Mirrors**: Configurations for different deformable mirror types, including Haka, Xinetics, and KASM.
- **Atmospheric Models**: Simulate atmospheric conditions with customizable parameters.

## Installation

To use this library, you need to have Python installed on your system. You can install the required dependencies using pip:

```sh
pip install git+https://github.com/jacotay7/keckSim.git
```

or clone the repository:

```sh
git clone https://github.com/jacotay7/keckSim.git
cd keckSim
pip install .
```

Ensure you have the OOPAO library installed, as it is a core dependency for this simulation library.
Instructions for the OOPAO library can be found here: https://github.com/cheritier/OOPAO

## Usage

### Telescope Simulation

To create a Keck telescope instance:

```python
from keckSim.keckTel import keckTel

#Create a Keck telescope with default parameters
keck_telescope = keckTel.create('keck')
```

### Atmosphere Simulation

To simulate atmospheric conditions:

```python
from keckSim.keckAtm import keckAtm

#Create a Mauna Kea atmosphere
mauna_kea_atm = keckAtm.create(telescope=keck_telescope.tel, atm_type='maunakea')
```

### Deformable Mirror Simulation

To configure a deformable mirror:

```python
from keckSim.keckDm import keckDm

#Create a Haka-style deformable mirror
haka_dm = keckDm.create(telescope=keck_telescope.tel, dm_type='haka')
```
## Examples

The repository includes example scripts demonstrating how to set up and run simulations for different components of the Keck AO system. Check the `examples` directory for more detailed use cases.

## Contributing

We welcome contributions from the community! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or feedback, please open an issue on GitHub or contact Jacob Taylor: jtaylor@keck.hawaii.edu

---

Thank you for using the Keck Adaptive Optics Simulation Library! We hope it serves as a valuable tool for your research and development in adaptive optics systems.