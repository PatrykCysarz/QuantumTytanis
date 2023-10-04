# QUANTUM TYTANIS

Created for HackYeah 2023

Application for visualising Lorenz & Rossler attractors with addition of quantum uncertainty principle. Without Qiskit usage application would yield almost the same result every time. Colors and starting values of attractor set with usage of quantum circuit. Each axis of image is mapped to different color - R, G or B. Each simulation step is affected by quantum uncertainty. 

Usage:

`python3 tytanis.py`

Command line parameters and their default values:

`--steps=100` iterations count

`--alg=lorenz` attractor generation algoritm - `lorenz` or `rossler`

`--s=10 --r=28 --b=2.667` default lorenz parameters

`--ra=0.2 --rb=0.2 --rc=5.7` default rossler parameters


![image](https://raw.githubusercontent.com/PatrykCysarz/QuantumTytanis/main/out_0.png)
