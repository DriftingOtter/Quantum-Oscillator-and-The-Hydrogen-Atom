import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from enum import Enum


class Energy(Enum):
    KINETIC = 0
    POTENTIAL = 1
    BOTH = 2


class ClassicalOscillator:

    def __init__(self, mass=1.0, k=0.4, rel_tolerance=1e-10, abs_tolerance=1e-12):
        if mass <= 0:
            raise ValueError("mass must be positive.")
        if k < 0:
            raise ValueError("k cannot be negative.")

        self.mass = mass
        self.k = k
        self.rel_tolerance = rel_tolerance
        self.abs_tolerance = abs_tolerance

        self.A = np.array(
            [
                [0, 1],
                [-self.k / self.mass, 0],
            ]
        )

    def system_dynamics(self, t, x):
        return self.A @ x

    def hamiltonian(self, x):
        pos = x[0]
        v = x[1]
        p = self.mass * v
        KE = (p ** 2) / (2 * self.mass)
        PE = 0.5 * self.k * pos ** 2
        return KE + PE, KE, PE

    def solve_system(self, t0, tn, x0):
        if t0 > tn:
            raise ValueError("Time Constraints Invalid: t0 <= tn.")

        x0_vec = np.array([x0, 0.0])
        t_span = (t0, tn)
        t_eval = np.linspace(t0, tn, num=500, endpoint=True)

        solv = solve_ivp(self.system_dynamics, t_span, x0_vec, t_eval=t_eval,
                         method='RK45', rtol=self.rel_tolerance, atol=self.abs_tolerance)

        if solv.status == 0:
            return solv.y, solv.t
        else:
            return None, None

    def _compute_energy(self, x, t):
        n = len(t)
        H_vals = np.zeros(n)
        KE_vals = np.zeros(n)
        PE_vals = np.zeros(n)

        for i in range(n):
            H_vals[i], KE_vals[i], PE_vals[i] = self.hamiltonian(x[:, i])

        return H_vals, KE_vals, PE_vals

    def plot_interactive(self, t0=0.0, tn=10.0, initial_x0=4.0):
        self.t0 = t0
        self.tn = tn

        self.fig, (self.ax_top, self.ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(bottom=0.15)

        self._draw_plots(initial_x0)

        ax_x0 = self.fig.add_axes([0.15, 0.05, 0.7, 0.03])
        self.x0_slider = Slider(
            ax=ax_x0,
            label='Initial Disp. ($x_0$)',
            valmin=-10.0,
            valmax=10.0,
            valinit=initial_x0
        )

        self.x0_slider.on_changed(self.update)

        plt.show()

    def _draw_plots(self, x0):
        x, t = self.solve_system(self.t0, self.tn, x0)
        if x is None:
            return

        H_vals, KE_vals, PE_vals = self._compute_energy(x, t)
        drift = np.max(np.abs(H_vals - H_vals[0]))

        self.ax_top.clear()
        self.ax_bot.clear()

        self.ax_top.plot(t, x[0], color='steelblue', label='Displacement $x(t)$', linewidth=2)

        max_disp = np.max(np.abs(x[0]))
        max_pe = np.max(PE_vals)
        PE_scaled = PE_vals * (max_disp / max_pe) if max_pe > 0 else PE_vals

        self.ax_top.plot(t, PE_scaled, color='gray', linestyle='--', alpha=0.5, label='Scaled V(x)')

        self.ax_top.axhline(0, color='black', linewidth=0.8)
        self.ax_top.set_ylabel("Amplitude")
        self.ax_top.set_title("Classical Harmonic Oscillator — Displacement & Scaled Potential")
        self.ax_top.legend(loc="upper right")
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.set_ylim(-(max_disp+5), max_disp+5)

        self.ax_bot.plot(t, KE_vals, color='green', label='KE = $p^2/2m$', linestyle='--')
        self.ax_bot.plot(t, PE_vals, color='gold', label='PE = $1/2kx^2$', linestyle='--')

        self.ax_bot.set_xlabel("Time (t)")
        self.ax_bot.set_ylabel("Energy (J)")
        self.ax_bot.set_title("Classical Harmonic Oscillator — Energy Conservation")
        self.ax_bot.legend(loc="upper right")
        self.ax_bot.grid(True, alpha=0.3)

        self.ax_bot.set_ylim(0, max_pe*1.2)

    def update(self, val):
        self._draw_plots(val)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    oscillator = ClassicalOscillator(mass=1.0, k=2)
    oscillator.plot_interactive(t0=0.0, tn=5.0, initial_x0=4.0)