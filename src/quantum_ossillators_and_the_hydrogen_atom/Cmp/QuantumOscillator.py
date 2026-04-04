import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import eval_hermite
from math import factorial
from enum import Enum


class PlotType(Enum):
    WAVEFUNCTION = 0
    PROBABILITY = 1
    BOTH = 2


class QuantumOscillator:

    def __init__(self, mass, omega, hbar=1.0):
        if mass <= 0:
            raise ValueError("Mass must be positive.")
        if omega <= 0:
            raise ValueError("Omega cannot be negative.")
        if hbar <= 0:
            raise ValueError("Planck's constant must be positive.")

        self.mass = mass
        self.omega = omega
        self.hbar = hbar

        self.alpha = np.sqrt((self.mass * self.omega) / self.hbar)

    def energy_level(self, n):
        return self.hbar * self.omega * (n + 0.5)

    def potential_energy(self, x):
        return 0.5 * self.mass * self.omega ** 2 * x ** 2

    def wavefunction(self, n, x):
        '''
        This is of the non-dimentionalized version of the schrodinger equation....
        '''
        y = self.alpha * x

        val = float(2 ** n) * float(factorial(n))
        N_n = np.sqrt(self.alpha / np.sqrt(np.pi)) * (1.0 / np.sqrt(val))

        H_n = eval_hermite(n, y)

        envelope = np.exp(-0.5 * y ** 2)

        return N_n * H_n * envelope

    def probability_density(self, n, x):
        psi = self.wavefunction(n, x)
        return np.abs(psi) ** 2

    def plot_state(self, initial_n=1, x_span=(-5.0, 5.0), plot_type=PlotType.BOTH):
        if plot_type not in PlotType:
            raise ValueError(f"plot_type must be a valid PlotType Enum.")

        self.x_span = x_span
        self.plot_type = plot_type

        if plot_type == PlotType.BOTH:
            self.fig, (self.ax_top, self.ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.subplots_adjust(bottom=0.15)
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            plt.subplots_adjust(bottom=0.20)

        self._draw_plots(initial_n)

        ax_freq = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])

        self.freq_slider = Slider(
            ax=ax_freq,
            label='Quantum Number (n)',
            valmin=0,
            valmax=10,
            valinit=initial_n,
            valstep=1  # Snap to integers
        )

        self.freq_slider.on_changed(self.update)

        plt.show()

    def _draw_plots(self, n):
        n = int(n)
        x = np.linspace(self.x_span[0], self.x_span[1], 1000)
        V_x = self.potential_energy(x)
        E_n = self.energy_level(n)

        psi = self.wavefunction(n, x)
        prob = self.probability_density(n, x)

        if self.plot_type == PlotType.BOTH:
            self.ax_top.clear()
            self.ax_bot.clear()

            V_scaled_wave = V_x * (np.max(np.abs(psi)) / E_n) if E_n != 0 else V_x
            V_scaled_prob = V_x * (np.max(prob) / E_n) if E_n != 0 else V_x

            self.ax_top.plot(x, psi, color='steelblue', label=f'$\\psi_{{{n}}}(x)$', linewidth=2)
            self.ax_top.plot(x, V_scaled_wave, color='gray', linestyle='--', alpha=0.5, label='Scaled V(x)')
            self.ax_top.axhline(0, color='black', linewidth=0.8)
            self.ax_top.set_ylabel("Amplitude")
            self.ax_top.set_title(f"Quantum Harmonic Oscillator (n={n}) — Wavefunction")
            self.ax_top.legend(loc="upper right")
            self.ax_top.grid(True, alpha=0.3)

            # Bottom Plot: Probability Density
            self.ax_bot.fill_between(x, prob, color='skyblue', alpha=0.5)
            self.ax_bot.plot(x, prob, color='navy', label=f'$|\\psi_{{{n}}}(x)|^2$', linewidth=2)
            self.ax_bot.plot(x, V_scaled_prob, color='gray', linestyle='--', alpha=0.5, label='Scaled V(x)')
            self.ax_bot.axhline(0, color='black', linewidth=0.8)
            self.ax_bot.set_xlabel("Position (x)")
            self.ax_bot.set_ylabel("Probability Density")
            self.ax_bot.legend(loc="upper right")
            self.ax_bot.grid(True, alpha=0.3)

        else:
            self.ax.clear()

            if self.plot_type == PlotType.WAVEFUNCTION:
                V_scaled = V_x * (np.max(np.abs(psi)) / E_n) if E_n != 0 else V_x
                self.ax.plot(x, psi, color='steelblue', label=f'$\\psi_{{{n}}}(x)$', linewidth=2)
                self.ax.set_ylabel("Amplitude")
                self.ax.set_title(f"Quantum Harmonic Oscillator (n={n}) — Wavefunction")
            elif self.plot_type == PlotType.PROBABILITY:
                V_scaled = V_x * (np.max(prob) / E_n) if E_n != 0 else V_x
                self.ax.fill_between(x, prob, color='skyblue', alpha=0.5)
                self.ax.plot(x, prob, color='navy', label=f'$|\\psi_{{{n}}}(x)|^2$', linewidth=2)
                self.ax.set_ylabel("Probability Density")
                self.ax.set_title(f"Quantum Harmonic Oscillator (n={n}) — Probability Density")

            self.ax.plot(x, V_scaled, color='gray', linestyle='--', alpha=0.5, label='Scaled V(x)')
            self.ax.axhline(0, color='black', linewidth=0.8)
            self.ax.set_xlabel("Position (x)")
            self.ax.legend(loc="upper right")
            self.ax.grid(True, alpha=0.3)

    def update(self, val):
        self._draw_plots(val)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    oscillator = QuantumOscillator(mass=1.0, omega=1.0, hbar=1.0)
    oscillator.plot_state(initial_n=1, x_span=(-5.0, 5.0), plot_type=PlotType.BOTH)