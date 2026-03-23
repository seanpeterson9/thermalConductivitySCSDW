# Anisotropic Thermal Conductivity in Superconductors with Spin Density Waves (GPU-Accelerated)

GPU-accelerated numerical calculation of thermal conductivity in superconducting systems coexisting with anisotropic spin density wave (SDW) order using Bogoliubov quasiparticle formalism.

---

## Overview

This project models thermal transport in superconductors where superconductivity coexists with anisotropic spin density wave order. The calculation requires repeated diagonalization of Bogoliubov Hamiltonians across a dense momentum-space grid.

Because the problem involves solving a large number of small matrices rather than a single large system, standard GPU linear algebra libraries (cuBLAS) were not well suited to the workload. This project therefore implements a custom GPU-based eigenvector solver using PyCUDA.

---

## Methodology

1. **Self-Consistent Calculation of Coexisting Order Parameters**
     - SDW order parameter (M) calculated in the absence of SC order parameter ($\Delta$)
     - SDW order parameter in the absence of SC used as an initial guess in presence of superconductivity
     - Coexesting SDW and SC order parameters calculated self-consistently from mean-field Hamiltonian
2. **Calculation of integration grid**
     - For anisotropic SDW ordering where Fermi surface nesting occurs along one direction and doesn't along the other, the underlying electronic structure from which superconductivity emerges is essentially the SDW state where nesting occurs and the normal tight-binding state where nesting doesn't occur
     - Diagonalizing these Hamiltonians in their respective regions yields the eigenvalues of the bands, which can be used to generate polar grids where the radial k-values are determined from where the eignenvalues are equivalent to the requisite $E$-value required to complete our grid where the energy can go from $-E_\text{cutoff}$ to $E_{cutoff}$.
     - The Hamiltonians in the presence of superconductivity are diagonalized for every point along this $(\theta,E)$-grid
4. **Calculation of Bogoliubov Quasiparticle lifetimes**
     - For every point in the $(\theta,E)$-grid generated in the previous step, quasiparticle lifetimes need to be calculated in order to calculate their contribution to the thermal conductivity tensor
     - This involves integrating each point over every other point in the grid, and diagonalizing their respective Hamiltonians to calculate the impurity-scattering probability from Fermi's Golden rule
     - The eigenvectors of these Hamiltonians were calculated numerically on GPUs in order to speed-up this integration process
6. **Calculation of Thermal Conductivity Tensor**
     - Contributions to the thermal conductivity tensor were then summed over the entire integration-grid for a particular temperature
     - Thermal conductivity tensor was diagonalized relative to find the components parallel and perpendicular to the nesting vector of the SDW ordering
   
---

## Results

- Captures anisotropic thermal transport behavior relative to the nesting vector of SDW ordering
- Provides a computational framework consistent with phenomenology observed in cuprate SCs
- Demonstrates scalability improvements via parallel execution

---

## Skills Demonstrated

- Tight-binding and Bogoliubov-de Gennes modeling
- Numerical integration in momentum space
- Scientific computing in Python (NumPy, SciPy)
- Parallel computing with multiprocessing (GPU-based)
- Performance optimization of numerical workflows 

---

## Notes

This code was developed as part of research into unconventional superconductivity and is intended as a demonstration of numerical modeling techniques rather than a production software package.

