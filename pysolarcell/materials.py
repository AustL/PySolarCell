import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
import importlib.resources

materials = importlib.resources.files('pysolarcell') / 'materials'


class Material:
    def __init__(self, name, bandgap):
        self.name = name
        self.bandgap = bandgap  # eV
        self.n = lambda x: 1
        self.k = lambda x: 0

    def setN(self, wavelengths, n):
        """Set the real refractive index

        :param wavelengths: List of wavelengths (nm)
        :param n: Refractive index at each wavelength
        :return: Creates function n(lambda)
        """

        self.n = scipy.interpolate.interp1d(wavelengths, n, fill_value=1, bounds_error=False)

    def setK(self, wavelengths, k):
        """Set the extinction coefficient

        :param wavelengths: List of wavelengths (nm)
        :param k: Extinction coefficient at each wavelength
        :return: Creates function k(lambda)
        """

        self.k = scipy.interpolate.interp1d(wavelengths, k, fill_value=0, bounds_error=False)

    def plot(self):
        wavelengths = np.linspace(300, 1200, 500)
        plt.plot(wavelengths, self.n(wavelengths), label='n')
        plt.plot(wavelengths, self.k(wavelengths), label='k')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('n, k')
        plt.legend()
        plt.title(f'Optical Properties of {self.name}')
        plt.show()


def PEROVSKITE():
    n_data = pd.read_csv(materials / 'CsPbI3_PSK_n.txt', sep='\t', names=['Wavelength', 'n'])
    k_data = pd.read_csv(materials / 'CsPbI3_PSK_k.txt', sep='\t', names=['Wavelength', 'k'])

    material = Material('Perovskite', 1.65)
    material.setN(n_data['Wavelength'] * 1e3, n_data['n'])
    material.setK(k_data['Wavelength'] * 1e3, k_data['k'])

    return material


def SILICON():
    n_data = pd.read_csv(materials / 'Si_n.txt', sep='\t', names=['Wavelength', 'n'])
    k_data = pd.read_csv(materials / 'Si_n.txt', sep='\t', names=['Wavelength', 'k'])

    material = Material('Silicon', 1.12)
    material.setN(n_data['Wavelength'] * 1e3, n_data['n'])
    material.setK(k_data['Wavelength'] * 1e3, k_data['k'])

    return material

if __name__ == '__main__':
    perovskite = PEROVSKITE()
    perovskite.plot()
    silicon = SILICON()
    silicon.plot()