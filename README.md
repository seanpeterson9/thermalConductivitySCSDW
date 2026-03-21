# Anisotropic Thermal Conductivity in Superconductors with Spin Density Waves (GPU-Accelerated)

GPU-accelerated numerical calculation of thermal conductivity in superconducting systems coexisting with anisotropic spin density wave (SDW) order using Bogoliubov quasiparticle formalism.

---

## Overview

This project models thermal transport in superconductors where superconductivity coexists with anisotropic spin density wave order. The calculation requires repeated diagonalization of Bogoliubov Hamiltonians across a dense momentum-space grid.

Because the problem involves solving a large number of small matrices rather than a single large system, standard GPU linear algebra libraries (cuBLAS) were not well suited to the workload. This project therefore implements a custom GPU-based eigenvector solver using PyCUDA.

---

## Methodology

1. **Self-Consistent Calculation of Coexisting Order Parameters**
     -
2. **Calculation of integration grid**

3. **Calculation of Bogoliubov Quasiparticle lifetimes**

4. **Calculation of Thermal Conductivity Tensor**
