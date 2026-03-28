import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from enum import Enum

class Energy(Enum):
    KINETIC = 0
    POTENTIAL = 1
    BOTH = 2

class ClassicalOscillator:

    def __init__(self, mass, k, rel_tolerance=1e-10, abs_tolerance=1e-12):
        if mass <= 0:
            raise Exception("mass must be positive.")
        if k < 0:
            raise Exception("k cannot be negative.")

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
            raise Exception("Time Constraints Invalid: t0 <= tn.")

        x0 = np.array([x0, 0.0])
        t_span = (t0, tn)
        t_eval = np.linspace(t0, tn, num=500, endpoint=True)

        solv = solve_ivp(self.system_dynamics, t_span, x0, t_eval=t_eval,
                         method='RK45', rtol=self.rel_tolerance, atol=self.abs_tolerance)

        if solv.status == 0:
            return solv.y, solv.t
        else:
            return None

    def _compute_energy(self, x, t):
        n = len(t)
        H_vals  = np.zeros(n)
        KE_vals = np.zeros(n)
        PE_vals = np.zeros(n)

        for i in range(n):
            H_vals[i], KE_vals[i], PE_vals[i] = self.hamiltonian(x[:, i])

        return H_vals, KE_vals, PE_vals

    @staticmethod
    def plot_disp(solv):
        if solv is None:
            raise Exception("Solution provided cannot be <None>.")

        x, t = solv

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t, x[0], color='steelblue', label='x(t)', linewidth=2)

        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Displacement (x)")
        ax.set_title("Classical Harmonic Oscillator — Displacement")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_energy(self, solv, energy):
        if solv is None:
            raise Exception("Solution provided cannot be <None>.")

        x, t = solv
        H_vals, KE_vals, PE_vals = self._compute_energy(x, t)
        drift = np.max(np.abs(H_vals - H_vals[0]))

        fig, ax = plt.subplots(figsize=(10, 5))

        if energy == Energy.KINETIC:
            ax.plot(t, KE_vals, color='green', label='KE = p^2/2m', linestyle='--')
        if energy == Energy.POTENTIAL:
            ax.plot(t, PE_vals, color='gold',  label='PE = 1/2kx^2',  linestyle='--')
        if energy == Energy.BOTH:
            ax.plot(t, KE_vals, color='green', label='KE = p^2/2m', linestyle='--')
            ax.plot(t, PE_vals, color='gold', label='PE = 1/2kx^2', linestyle='--')

        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"Classical Harmonic Oscillator — Energy  |  max drift = {drift:.2e} J")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_all(self, solv, energy):
        if solv is None:
            raise Exception("Solution provided cannot be <None>.")

        x, t = solv
        H_vals, KE_vals, PE_vals = self._compute_energy(x, t)
        drift = np.max(np.abs(H_vals - H_vals[0]))

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax_top.plot(t, x[0], color='steelblue', label='x(t)', linewidth=2)
        ax_top.set_xlabel("Time (t)")
        ax_top.set_ylabel("Displacement (x)")
        ax_top.set_title("Classical Harmonic Oscillator — Displacement")
        ax_top.legend()
        ax_top.grid(True)

        if energy == Energy.KINETIC:
            ax_bot.plot(t, KE_vals, color='green', label='KE = p^2/2m', linestyle='--', linewidth=2)
        if energy == Energy.POTENTIAL:
            ax_bot.plot(t, PE_vals, color='gold',  label='PE = 1/2kx^2',  linestyle='--', linewidth=2)
        if energy == Energy.BOTH:
            ax_bot.plot(t, KE_vals, color='green', label='KE = p^2/2m', linestyle='--')
            ax_bot.plot(t, PE_vals, color='gold', label='PE = 1/2kx^2', linestyle='--')

        ax_bot.set_xlabel("Time (t)")
        ax_bot.set_ylabel("Energy (J)")
        ax_bot.set_title(f"Classical Harmonic Oscillator — Energy")
        ax_bot.legend()
        ax_bot.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    oscillator = ClassicalOscillator(mass=1.0, k=0.4)
    solution = oscillator.solve_system(0.0, 5.0, 5.0)
    oscillator.plot_all(solution, energy=Energy.POTENTIAL)