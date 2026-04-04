# Quantum Oscillators and the Hydrogen Atom

This repository supports a **theoretical write-up** paired with an **interactive visualization** stack. The goal is to connect the quantum harmonic oscillator and hydrogen radial structure to classical limits as quantum numbers grow, with clean 2D radial plots (full 3D orbitals are out of scope).

**Current code** lives under `src/quantum_ossillators_and_the_hydrogen_atom/Cmp/`:

| Module | Role |
|--------|------|
| `QuantumOscillator.py` | 1D quantum harmonic oscillator: energies, wavefunctions via Hermite polynomials (`scipy.special.eval_hermite`), probability density, and a **matplotlib slider** for the quantum number \(n\) (0–10). Optional views: wavefunction, probability, or both (stacked). |
| `ClassicalOscillator.py` | Classical harmonic oscillator as a linear ODE solved with **`scipy.integrate.solve_ivp` (RK45)**; interactive plot with a slider for initial displacement \(x_0\), showing displacement vs. time and energy (KE/PE). |

Planned hydrogen radial plotters, correspondence-focused comparisons, and paper-facing derivations are **not yet implemented in this repo**; they are captured in the roadmap below.

---

## Setup

- **Python:** `>=3.14` (see `pyproject.toml`).
- **Dependencies:** NumPy, SciPy, Matplotlib (managed with [Poetry](https://python-poetry.org/)).

```bash
cd /path/to/Quantum_Ossillators_and_The_Hydrogen_Atom
poetry install
```

---

## Running the interactive demos

From the repository root, run either module as a script (each file has a `if __name__ == "__main__":` block):

```bash
poetry run python src/quantum_ossillators_and_the_hydrogen_atom/Cmp/QuantumOscillator.py
poetry run python src/quantum_ossillators_and_the_hydrogen_atom/Cmp/ClassicalOscillator.py
```

- **Quantum:** use the **Quantum Number (n)** slider; the figure can show \(\psi_n(x)\), \(|\psi_n(x)|^2\), and a scaled harmonic potential for context.
- **Classical:** use the **Initial Disp. (\(x_0\))** slider to re-simulate the trajectory.

---

## Project plan (math, code, synthesis)

Keeping **derivations** (paper) separate from **plotting code** keeps the work manageable.

### Phase 1 — Mathematical foundation and write-up

| Topic | Direction |
|-------|-----------|
| **Quantum harmonic oscillator** | Start from the 1D Schrödinger equation; use the **power series method** to derive Hermite polynomials and the ladder of solutions (differential-equations proficiency). |
| **Radial hydrogen atom** | Set up the **separated radial equation**; for the full analytic form, **cite known exact solutions** in terms of **associated Laguerre polynomials** rather than re-deriving asymptotic termination in full. |
| **Correspondence principle** | Narrative section: how both systems move toward **classical** behavior as **\(n \to \infty\)**. |
| **Angular momentum (\(l\))** | Contrast **fixed \(l\)** (radial nodes proliferate as \(n\) grows) with **\(l = n - 1\)** (“circular” states: no radial nodes, closer to classical circular orbits). |

### Phase 2 — Visualization engine (2D radial focus)

| Deliverable | Status in repo |
|-------------|----------------|
| **QHO plotter** — \(|\psi_n|^2\) vs position for chosen \(n\) | **Present** (`QuantumOscillator.py`; also wavefunction + slider). |
| **Hydrogen radial plotter** — radial probability for \((n,l)\) | **Planned** |
| **Interactive controller** — scrub \(n\) smoothly | **Partial** — integer \(n\) slider for QHO; extend/replicate for hydrogen. |
| **Hydrogen comparison view** — e.g. fixed \(l=0\) vs \(l=n-1\) side-by-side as \(n\) increases | **Planned** |

Optional: unify classical and quantum figures in a single “correspondence” narrative in the paper; the classical script already gives a concrete **classical** reference trajectory.

---

## Layout

```
src/quantum_ossillators_and_the_hydrogen_atom/
  Cmp/
    ClassicalOscillator.py   # classical HO, RK45, slider
    QuantumOscillator.py     # QHO, Hermite, probability, slider
  main.py                    # placeholder
```

Package metadata and dependencies: `pyproject.toml`.

---

## Note on imports

The top-level package `__init__.py` currently references modules that are not all present in the tree; for day-to-day use, **run the `Cmp` scripts directly** as above until the package layout is completed.
