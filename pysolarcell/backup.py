import sympy as sp
import numpy as np
import pandas as pd
import scipy.integrate

# Constants
h = 6.62607015e-34  # Js
c = 299792458  # m/s
k = 1.380649e-23  # J/K
q = 1.60217e-19  # C
sigma = 5.670374419e-8  # W/m^2/K^4
pi = np.pi
n_points = 100

AM15G = pd.read_csv('spectra/AM1.5G.txt', skiprows=1, names=['Wavelength', 'Spectral Irradiance', 'Photon Flux'])
# plt.plot(AM15G['Wavelength'], AM15G['Spectral Irradiance'])


class SolarCell:
    def __init__(self, Jsc, Voc, area=100, Rs=0.01, Rsh=5e5, T=298, n=1):
        """Class representing a solar cell

        :param Jsc: Ideal short circuit current density (mA/cm^2)
        :param Voc: Ideal open circuit voltage (V)
        :param area: Area of cell (cm^2)
        :param Rs: Series resistance (Ohm cm^2)
        :param Rsh: Shunt resistance (Ohm cm^2)
        :param T: Temperature
        :param n: Ideality factor
        """

        self.Isc = Jsc * area / 1000  # Ideal short circuit current in mA/cm^2
        self.Jsc = Jsc
        self.Voc = Voc  # Ideal open circuit voltage in V
        self.area = area  # Area in cm^2
        self.Rs = Rs / area
        self.Rsh = Rsh / area
        self.T = T
        self.n = n

        self.voltages = np.linspace(0, self.Voc, n_points)
        self.currents = np.linspace(0, self.Isc, n_points)
        self.I0 = self.Isc / (np.exp(q * self.Voc / (self.n * k * self.T)))

        self.found_currents = None
        self.found_voltages = None

    def find_current(self, voltages=None):
        """Returns an array of currents
        """

        if self.found_currents is not None:
            return self.found_currents

        if voltages is None:
            voltages = self.voltages

        I = sp.Symbol('I')
        result = np.zeros_like(voltages)
        for index, v in enumerate(voltages):
            guess = self.Isc - self.I0 * (sp.exp(q * v / (self.n * k * self.T)) - 1)
            result[index] = sp.nsolve(self.Isc
                                      - self.I0 * (sp.exp((v + I * self.Rs) / (self.n * k * self.T / q)) - 1)
                                      - (v + I * self.Rs) / self.Rsh - I, guess)

        self.found_currents = result
        return result

    def find_voltage(self, currents=None):
        """Returns an array of voltages
        """

        if self.found_voltages is not None:
            return self.found_voltages

        if currents is None:
            currents = self.currents

        V = sp.Symbol('V')
        result = np.zeros_like(currents)
        for index, i in enumerate(currents):
            guess = (self.n * k * self.T) / q * np.log(self.Isc / self.I0 + 1)
            result[index] = sp.nsolve(self.Isc
                                      - self.I0 * (sp.exp((V + i * self.Rs) / (self.n * k * self.T / q)) - 1)
                                      - (V + i * self.Rs) / self.Rsh - i, guess)

        self.found_voltages = result
        return result

    def iv(self):
        I = self.find_current()

        return self.voltages, self.ItoJ(I)

    def mpp(self):
        """Finds the maximum power point of the solar cell

        :return: Tuple of ((voltage, current), maximum power)
        """
        currents = self.find_current()
        max_index = np.argmax(self.voltages * currents)

        return (self.voltages[max_index], self.ItoJ(currents[max_index])), self.voltages[max_index] * currents[
            max_index]

    def ItoJ(self, current):
        return current * 1000 / self.area

    @staticmethod
    def mpp_from_iv(voltages, currents):
        """Finds the maximum power point from an IV curve

        :param voltages: Voltage points
        :param currents: Current points (mA/cm^2)
        :return: Tuple of ((voltage, current), maximum power)
        """
        max_index = np.argmax(voltages * currents)
        max_power = voltages[max_index] * currents[max_index]
        # print(f'Maximum Power: {max_power} W')

        return (voltages[max_index], currents[max_index]), max_power

    @staticmethod
    def add_parallel(cell1: 'SolarCell', cell2: 'SolarCell'):
        """Returns the IV curve of two solar cells in parallel

        :param cell1: First solar cell
        :param cell2: Second solar cell
        :return: Tuple of (voltages, currents)
        """

        voltages = np.linspace(0, min(cell1.Voc, cell2.Voc) + 0.05, n_points)
        currents = cell1.ItoJ(cell1.find_current(voltages)) + cell2.ItoJ(cell2.find_current(voltages))

        return voltages, currents

    @staticmethod
    def add_series(cell1: 'SolarCell', cell2: 'SolarCell'):
        """Returns the IV curve of two solar cells in series

        :param cell1: First solar cell
        :param cell2: Second solar cell
        :return: Tuple of (voltages, currents)
        """

        currents = np.linspace(0, min(cell1.Isc, cell2.Isc) + 0.005, n_points)
        voltages = cell1.find_voltage(currents) + cell2.find_voltage(currents)

        return voltages, cell1.ItoJ(currents)

    @classmethod
    def fromBandgap(Eg, upper_Eg=np.inf, area=100, Rs=0.01, Rsh=5e5, T=298, n=1):
        """Creates an ideal solar cell from its bandgap bandgap assuming an AM1.5G spectrum

        :param Eg: Bandgap energy (eV) must be more than 0.5 eV
        :param upper_Eg: Bandgap energy of the cell above
        :return: A solar cell with calculated Jsc, Voc from bandgap
        """

        Eg = Eg * q  # Energy in J
        upper_Eg = upper_Eg * q

        u = Eg / (k * T)
        J0 = (15 * q * sigma * T ** 4) / (k * pi ** 4) * \
             scipy.integrate.quad(lambda x: x ** 2 / (np.exp(x) - 1), u, 500)[
                 0] / 10  # Diode saturation current (mA/cm^2)

        lambda_g = h * c / Eg * 1e9  # Bandgap wavelength in nm
        lambda_upper = h * c / upper_Eg * 1e9  # Upper bandgap wavelength
        wavelengths = (AM15G['Wavelength'] < lambda_g) & (AM15G['Wavelength'] > lambda_upper)

        spectral_response = q * AM15G['Wavelength'][wavelengths] / (
                    h * c) * 1e-9  # Spectral response assuming EQE = 1 whenever above the bandgap energy and 0 otherwise (A/W)
        Jsc = scipy.integrate.trapezoid(AM15G['Spectral Irradiance'][wavelengths] * spectral_response,
                                        AM15G['Wavelength'][wavelengths]) / 10  # mA/cm^2

        Voc = n * k * T / q * np.log(Jsc / J0 + 1)

        solar_cell = SolarCell(Jsc, Voc, area=area, Rs=Rs, Rsh=Rsh, T=T, n=n)

        return solar_cell