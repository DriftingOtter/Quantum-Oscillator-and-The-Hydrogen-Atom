import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from scipy.special import genlaguerre, factorial
from numba import njit, prange


@njit(parallel=True, cache=True)
def evaluate_laguerre(rho, coeffs):
    N = rho.shape[0]
    result = np.zeros(N)
    for i in prange(N):
        val = 0.0
        x = rho[i]
        for k in range(len(coeffs) - 1, -1, -1):
            val = val * x + coeffs[k]
        result[i] = val
    return result


@njit(parallel=True, cache=True)
def evaluate_radial_wavefunction(rho, l, laguerre_vals, N_nl):
    N = rho.shape[0]
    R = np.zeros(N)
    for i in prange(N):
        R[i] = N_nl * np.exp(-rho[i] / 2.0) * (rho[i] ** l) * laguerre_vals[i]
    return R


@njit(parallel=True, cache=True)
def evaluate_probability_density(r, R):
    N = r.shape[0]
    P = np.zeros(N)
    for i in prange(N):
        P[i] = r[i] ** 2 * R[i] ** 2
    return P


class HydrogenAtom:

    def __init__(self, Z=1, a0=1.0):
        if Z <= 0:
            raise ValueError("Atomic number Z must be positive.")
        if a0 <= 0:
            raise ValueError("Bohr radius must be positive.")

        self.Z  = Z
        self.a0 = a0
        self._cache = {}
        self._polar_mesh = None
        self._warmup()

    def _warmup(self):
        self._evaluate(1, 0, np.linspace(0.01, 5.0, 128))

    def _normalization_constant(self, n, l):
        factor = (2.0 * self.Z / (n * self.a0)) ** 3
        num    = float(factorial(n - l - 1))
        den    = 2.0 * n * float(factorial(n + l)) ** 3
        return np.sqrt(factor * num / den)

    def _laguerre_coefficients(self, n, l):
        L = genlaguerre(n - l - 1, 2 * l + 1)
        return np.array(L.c[::-1], dtype=np.float64)

    def _evaluate(self, n, l, r):
        key = (n, l, len(r))
        if key in self._cache:
            return self._cache[key]

        rho    = np.ascontiguousarray((2.0 * self.Z / (n * self.a0)) * r, dtype=np.float64)
        r_cont = np.ascontiguousarray(r, dtype=np.float64)
        N_nl   = self._normalization_constant(n, l)
        coeffs = self._laguerre_coefficients(n, l)

        laguerre_vals    = evaluate_laguerre(rho, coeffs)
        R                = evaluate_radial_wavefunction(rho, l, laguerre_vals, N_nl)
        P                = evaluate_probability_density(r_cont, R)
        self._cache[key] = (R, P)
        return R, P

    def _radial_domain(self, n):
        return (1e-4 * self.a0, 2.0 * n ** 2 * self.a0 * (n + 2))

    def _clamp_l(self, n, l):
        return int(max(0, min(l, n - 1)))

    def _orbital_label(self, n, l):
        shell_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        return f"{n}{shell_names.get(l, f'l={l}')}"

    def _build_polar_mesh(self, r, R, r_cutoff_scaled):
        theta = np.linspace(0, 2 * np.pi, 360)
        R_abs = np.abs(R)
        r_scaled = r / self.a0

        # clip to the effective probability region
        mask = r_scaled <= r_cutoff_scaled
        r_ds_full = r_scaled[mask]
        R_ds_full = R_abs[mask]

        step = max(1, len(r_ds_full) // 400)
        r_ds = r_ds_full[::step]
        R_ds = R_ds_full[::step]

        theta_grid, R_grid = np.meshgrid(theta, R_ds)
        r_grid = np.tile(r_ds, (len(theta), 1)).T
        return theta_grid, r_grid, R_grid

    def plot_state(self, initial_n=1, initial_l=0):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor('white')

        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[1, 1],
            hspace=0.38, wspace=0.35,
            left=0.08, right=0.95,
            top=0.92,   bottom=0.18
        )

        self.ax_amplitude = self.fig.add_subplot(gs[0, 0])
        self.ax_polar     = self.fig.add_subplot(gs[0, 1], projection='polar')
        self.ax_prob      = self.fig.add_subplot(gs[1, :])

        self._draw_plots(initial_n, initial_l)

        ax_n_slider = self.fig.add_axes([0.15, 0.09, 0.7, 0.025])
        ax_l_slider = self.fig.add_axes([0.15, 0.04, 0.7, 0.025])

        self.n_slider = Slider(ax_n_slider, 'n', 1, 15, valinit=initial_n, valstep=1)
        self.l_slider = Slider(ax_l_slider, 'l', 0, 14, valinit=initial_l, valstep=1)

        self.n_slider.on_changed(self.update)
        self.l_slider.on_changed(self.update)

        plt.show()

    def _draw_plots(self, n, l):
        n = int(n)
        l = self._clamp_l(n, l)

        r_min, r_max = self._radial_domain(n)
        r = np.linspace(r_min, r_max, 4000)
        R, P = self._evaluate(n, l, r)
        r_scaled = r / self.a0

        radial_nodes = n - l - 1
        orbital = self._orbital_label(n, l)
        self.fig.suptitle(
            f"Hydrogen Atom  |  $n={n}$,  $\\ell={l}$  ({orbital})  "
            f"|  radial nodes $= {radial_nodes}$",
            fontsize=12
        )

        self.ax_amplitude.clear()
        self.ax_amplitude.plot(r_scaled, R, color='steelblue',
                               label=f'$R_{{{n}{l}}}(r)$', linewidth=1.8)
        self.ax_amplitude.axhline(0, color='black', linewidth=0.7)
        self.ax_amplitude.set_xscale('log')
        self.ax_amplitude.set_xlabel(r'$r\,/\,a_0$  (log scale)')
        self.ax_amplitude.set_ylabel('Amplitude')
        self.ax_amplitude.set_title('Radial Wavefunction $R_{n\\ell}(r)$', fontsize=10)
        self.ax_amplitude.legend(loc='upper right', fontsize=8)
        self.ax_amplitude.grid(True, which='both', alpha=0.25)

        dr = r[1] - r[0]
        cumP = np.cumsum(P) * dr
        cumP /= cumP[-1]
        idx_cutoff = np.searchsorted(cumP, 0.999)
        r_cutoff_scaled = r_scaled[min(idx_cutoff, len(r_scaled) - 1)]
        r_polar_lim = r_cutoff_scaled * 1.08

        self.ax_polar.clear()
        theta_grid, r_grid, R_grid = self._build_polar_mesh(r, R, r_cutoff_scaled)
        vmin, vmax = R_grid.min(), R_grid.max()
        if vmax - vmin < 1e-14:
            vmax = vmin + 1e-14

        self._polar_mesh = self.ax_polar.pcolormesh(
            theta_grid, r_grid, R_grid,
            cmap='plasma', shading='auto',
            norm=Normalize(vmin=vmin, vmax=vmax)
        )
        self.ax_polar.set_ylim(0, r_polar_lim)
        self.ax_polar.set_title(r'Polar  $|R_{n\ell}(r)|$  vs $r/a_0$', fontsize=10, pad=12)
        self.ax_polar.tick_params(labelsize=7)

        self.ax_prob.clear()
        self.ax_prob.fill_between(r_scaled, P, color='skyblue', alpha=0.4)
        self.ax_prob.fill_between(-r_scaled, P, color='skyblue', alpha=0.4)
        self.ax_prob.plot(r_scaled, P, color='navy', linewidth=1.8,
                          label=f'$P(r)=r^2|R_{{{n}{l}}}(r)|^2$')
        self.ax_prob.plot(-r_scaled, P, color='navy', linewidth=1.8)
        self.ax_prob.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
        self.ax_prob.axhline(0, color='black', linewidth=0.7)

        r_max_scaled = r_scaled.max()
        self.ax_prob.set_xlim(-r_max_scaled, r_max_scaled)
        self.ax_prob.set_xlabel(r'$\pm\,r\,/\,a_0$')
        self.ax_prob.set_ylabel('Radial Probability $P(r)$')
        self.ax_prob.set_title('Radial Probability Distribution $P(r)$ — Cross-Section', fontsize=10)
        self.ax_prob.legend(loc='upper right', fontsize=8)
        self.ax_prob.grid(True, alpha=0.25)

    def update(self, val):
        n = int(self.n_slider.val)
        l = self._clamp_l(n, int(self.l_slider.val))
        if int(self.l_slider.val) != l:
            self.l_slider.set_val(l)
            return
        self._polar_mesh = None
        self._draw_plots(n, l)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    HydrogenAtom(Z=1, a0=1.0).plot_state(initial_n=1, initial_l=0)