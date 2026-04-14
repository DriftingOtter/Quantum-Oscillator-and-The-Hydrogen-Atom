# Quantum Oscillators and the Hydrogen Atom

This repository supports a written development of the quantum harmonic oscillator and the hydrogen atom by pairing the mathematics with **interactive figures** you can explore in Python. You can watch how stationary states of the oscillator depend on the quantum number, compare that picture with the motion of a **classical** harmonic oscillator solved as a differential equation, and then move to hydrogen: radial wavefunctions, radial probability, and how those shapes change when you vary \(n\) and \(\ell\). All of the graphics are built with **Matplotlib** and focus on **two-dimensional** plots—time series, radial curves, and a polar view of the radial amplitude—rather than a full three-dimensional rendering of atomic orbitals.

## What the programs do

**`QuantumOscillator.py`** — The one-dimensional quantum harmonic oscillator in closed form. The code evaluates the Hermite–Gaussian eigenfunctions \(\psi_n(x)\), plots the probability density \(|\psi_n(x)|^2\), and overlays the harmonic potential (scaled so it sits on the same vertical scale as the curves). A slider lets you step the quantum number **n** through the integers from **0 to 1000**, so you can sweep from the ground state to highly excited states and see how the oscillations and the classical turning region behave.

**`ClassicalOscillator.py`** — The same physical system in the classical limit: a mass on a spring, written as a linear ODE and integrated with **`scipy.integrate.solve_ivp`** and the **RK45** solver. You choose an initial displacement \(x_0\) with a slider; the plot shows the trajectory in time and how kinetic energy, potential energy, and the total exchange energy over a period. Together with the quantum script, this gives a direct side-by-side narrative for the harmonic problem.

**`HydrogenOrbital.py`** — The radial part of the nonrelativistic hydrogen problem. Radial wavefunctions \(R_{n\ell}(r)\) are built from the standard Coulomb normalization and associated Laguerre structure; you see the radial amplitude on a logarithmic radius axis, a polar colormap of \(|R_{n\ell}|\), and the radial probability \(P(r)=r^2|R_{n\ell}|^2\) shown as a symmetric cross-section about the nucleus. Sliders set **n** from **1 to 15** and **\(\ell\)** from **0** upward, with \(\ell\) limited by **\(\ell \le n-1\)** so only physical combinations appear. The heavier numerical work runs through **Numba** so the plots stay responsive as you change \(n\) and \(\ell\).

## Where everything lives

```
src/quantum_ossillators_and_the_hydrogen_atom/
  ClassicalOscillator.py
  HydrogenOrbital.py
  QuantumOscillator.py
  __init__.py
```

Python **≥ 3.14** and the full list of third-party packages are recorded in **`pyproject.toml`**.

## Installing dependencies

From the repository root, install with Poetry:

```bash
cd /path/to/Quantum_Ossillators_and_The_Hydrogen_Atom
poetry install
```

## Running the visualizations

Each file is meant to be run as a script from the repository root:

```bash
poetry run python src/quantum_ossillators_and_the_hydrogen_atom/QuantumOscillator.py
poetry run python src/quantum_ossillators_and_the_hydrogen_atom/ClassicalOscillator.py
poetry run python src/quantum_ossillators_and_the_hydrogen_atom/HydrogenOrbital.py
```

When a window opens, use the sliders along the bottom (or as labeled on the figure). For the quantum oscillator, move **Quantum Number (n)** to refresh \(\psi_n\), \(|\psi_n|^2\), and the scaled potential; by default you get the wavefunction on top and the probability density underneath. For the classical oscillator, move **Initial Disp. (\(x_0\))** to rerun the integration and update the trajectory and energy traces. For hydrogen, adjust **n** and **\(\ell\)** to see all three panels respond together—the radial line plot, the polar view of \(|R_{n\ell}|\), and the mirrored radial probability in \(r/a_0\).

## Note on importing this as a package

The `__init__.py` file re-exports the modules using short names. That works cleanly when you **run each `.py` file directly**, which is the intended way to launch these tools. If you later install the project and `import quantum_ossillators_and_the_hydrogen_atom` from another codebase, you may need to switch those imports to explicit package-relative or absolute imports so they resolve under your environment.
