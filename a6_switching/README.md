# A6 Python switching evidence — public source scope

The associated V24 evidence uses a 25-kHz ideal-switch/diode DCO implementation for a selected A6 disagreement, with 16 and 32 RK4 subdivisions and +0.2% initial DC-link-voltage perturbation. The global VSC/PLL model remains averaged and the grid algebraic.

The complete public-safe source selection has been assembled separately but has **not yet been uploaded to this directory**. The currently available `scientific-requirements.txt` records the numerical environment. Do not interpret this folder as a standalone executable reproduction of A6 until the source/input/verification files are committed and independently checked.

A6 is not an independent guard-validation case. Native Simulink A6 remains unresolved; no full EMT, HIL, Floquet, or hardware verification is claimed.
