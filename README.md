# quvac

Quvac (from quantum vacuum, pronounced as qu-ack 🐸) allows to calculate quantum vacuum signals produced during light-by-light scattering.

Documentation is available [here](https://maxbalrog.github.io/quantum-vacuum/).

## Installation

Is is recommended to create a separate Python environment for this package, e.g.

```bash
    micromamba create -n quvac python=3.12
```

After cloning the git repository it could be simply installed with

```bash
    pip install quantum-vacuum
```

After successfull installation run ``pytest`` to make sure the installation was
successfull (it takes some time).

```bash
    pytest
```

## Contribution

If you noticed a bug or have a feature request, open a new [issue](https://github.com/maxbalrog/quantum-vacuum/issues).

## References

[1] F. Karbstein, and R. Shaisultanov. "Stimulated photon emission from the vacuum." PRD 91.11 (2015): 113002.

[2] A. Blinne, et al. "All-optical signatures of quantum vacuum nonlinearities in generic laser fields." PRD 99.1 (2019): 016006.
