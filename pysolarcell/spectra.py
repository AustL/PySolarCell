import importlib.resources
import pandas as pd
import matplotlib.pyplot as plt
import pySMARTS
import os
import scipy.integrate

from pandas.errors import EmptyDataError

spectra = importlib.resources.files('pysolarcell') / 'spectra'

AM15G_CONST = pd.read_csv(spectra / 'AM1.5G.txt', skiprows=1, sep='\t',
                          names=['Wavelength', 'AM0', 'Spectral Irradiance', 'AM1.5D'])
AM15D_CONST = pd.read_csv(spectra / 'AM1.5G.txt', skiprows=1, sep='\t',
                          names=['Wavelength', 'AM0', 'AM1.5G', 'Spectral Irradiance'])
AM0_CONST = pd.read_csv(spectra / 'AM1.5G.txt', skiprows=1, sep='\t',
                        names=['Wavelength','Spectral Irradiance','AM1.5G', 'AM1.5D'])
LED1_CONST = pd.read_csv(spectra / 'LED1.txt', skiprows=2, sep='\t',
                         names=['Wavelength','Spectral Irradiance'])
LED2_CONST = pd.read_csv(spectra / 'LED2.txt', skiprows=3, sep='\t',
                         names=['Wavelength','Spectral Irradiance'])

def AM15G():
    return Lamp(AM15G_CONST).copy()

def AM15D():
    return Lamp(AM15D_CONST).copy()

def AM0():
    return Lamp(AM0_CONST).copy()

def LED1():
    return Lamp(LED1_CONST).copy()

def LED2():
    return Lamp(LED2_CONST).copy()

class Lamp:
    def __init__(self, spectrum, wavelength_name='Wavelength', irradiance_name='Spectral Irradiance', direct_name='AM1.5D'):
        self.spectrum = spectrum  # Wavelength in nm and irradiance in W/m^2/nm
        self.wavelength_name = wavelength_name
        self.irradiance_name = irradiance_name
        self.direct_name = direct_name

    def plot(self, figax=None):
        if figax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig, ax = figax

        ax.plot(self.spectrum[self.wavelength_name], self.spectrum[self.irradiance_name])
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Spectral Irradiance ($W/m^2/nm$)')

        return fig, ax

    def wavelengths(self):
        return self.spectrum[self.wavelength_name]

    def irradiances(self):
        return self.spectrum[self.irradiance_name]

    def update(self, EQE):
        self.spectrum[self.irradiance_name] = self.spectrum[self.irradiance_name] * (1 - EQE)

    def copy(self):
        return Lamp(self.spectrum.copy(), wavelength_name=self.wavelength_name, irradiance_name=self.irradiance_name)

    def head(self, *args, **kwargs):
        return self.spectrum.head(*args, **kwargs)

    def modify_clarity(self, clarity):
        assert self.direct_name in self.spectrum.columns, 'Direct irradiances are required for clarity modification.'

        spectrum = pd.DataFrame()
        spectrum[self.wavelength_name] = self.spectrum[self.wavelength_name]
        spectrum[self.irradiance_name] = self.spectrum[self.irradiance_name] - self.spectrum[self.direct_name] * (1 - clarity)

        return Lamp(spectrum, self.wavelength_name, self.irradiance_name)

    def total_power(self):
        """ Calculates the total power in mW/cm^2

        :return: Total power in mW/cm^2
        """
        return scipy.integrate.trapezoid(self.irradiances(), self.wavelengths()) / 10

    @staticmethod
    def from_csv(filename, wavelength_name='Wavelength', irradiance_name='Spectral Irradiance', direct_name='AM1.5D', **kwargs):
        spectrum = pd.read_csv(filename, **kwargs)

        return Lamp(spectrum, wavelength_name, irradiance_name, direct_name)

    def to_csv(self, filename, *args):
        self.spectrum.to_csv(filename, *args)

    @staticmethod
    def from_smarts(year, month, day, hour, latitude, longitude, altitude=0, timezone=0):
        """ Returns a lamp from a time and place

        :param year: Year
        :param month: Month
        :param day: Day
        :param hour: Hour
        :param latitude: Latitude
        :param longitude: Longitude
        :param altitude: Altitude (km above sea level)
        :param timezone: Timezone (hours ahead of GMT)
        :return: Lamp
        """

        IOUT = '4 5'  # DNI and DHI
        YEAR = str(year)
        MONTH = str(month)
        DAY = str(day)
        HOUR = str(hour)
        LATIT = str(latitude)
        LONGIT = str(longitude)
        ALTIT = str(altitude)  # km above sea level
        ZONE = str(timezone)

        assert 'SMARTSPATH' in os.environ, 'Please add the SMARTS2 software to the SMARTSPATH environment variable.'

        try:
            spectrum = pySMARTS.SMARTSTimeLocation(IOUT, YEAR, MONTH, DAY, HOUR, LATIT, LONGIT, ALTIT, ZONE)
        except EmptyDataError:
            return None

        return Lamp(spectrum, 'Wvlgth', 'Global_horizn_irradiance', 'Direct_horizn_irradiance')