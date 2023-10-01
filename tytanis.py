from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit_aer import AerSimulator
from numpy import pi, sqrt
import random
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import array as arr
import getopt
import sys

steps = 100
alg="lorenz"
s=10
r=28
b=2.667
lw=1
ra = 0.2
rb = 0.2
rc = 5.7

argv = sys.argv[1:]
short_opts = "st:"
long_opts = ["steps=", "s=", "r=", "b=", "lw=", "ra=", "rb=", "rc=", "alg="]

try:
	args, opts = getopt.getopt(argv, short_opts, long_opts)
except getopt.error as err:
	print (str(err))
  
## Loop through arguments
for current_argument, current_value in args:
    if current_argument in ("-st", "--steps"):
        steps = int(current_value)
    if current_argument in ("--alg"):
        alg = current_value
    elif current_argument in ("--s"):
        s = int(current_value)
    
    elif current_argument in ("--r"):
        r = int(current_value)
    
    elif current_argument in ("--b"):
        b = float(current_value)
    elif current_argument in ("--lw"):
        lw = float(current_value)
    elif current_argument in ("--ra"):
        ra = float(current_value)
    elif current_argument in ("--rb"):
        rb = float(current_value)
    elif current_argument in ("--rc"):
        rc = float(current_value)


def quantum_shit(x, y, z):
    simulator = AerSimulator()
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    if (x > 20):
        print('asd');
    circuit.measure([0, 1], [0, 1])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=40)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts['00'] / counts['11']

def quantum_shit2(position, scale, shift):
    print(position, scale)
    simulator = AerSimulator()
    qreg_q = QuantumRegister(1, 'q')
    creg_c = ClassicalRegister(1, 'c')
    color_circuit = QuantumCircuit(qreg_q, creg_c)
    color_circuit.reset(qreg_q[0])
    color_circuit.x(qreg_q[0])
    color_circuit.rx(pi/(2 + ((position / scale)*4)), qreg_q[0])
    color_circuit.measure(qreg_q[0], creg_c[0])
    compiled_circuit = transpile(color_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=100)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
 
    print(counts)
 
    if '0' not in counts:
        counts['0'] = 1
 
    if '1' not in counts:
        counts['1'] = 1
 
    result = counts['1'] * 0.01
    result = abs(pow(result, 2) - 0.5) * 2

    return result % 1

quantum_shit(1, 2, 3)

red_shift = random.uniform(0, 1)
green_shift = random.uniform(0, 1)
blue_shift = random.uniform(0, 1)

def lorenz(xyz):
    x, y, z = xyz

    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot]), [quantum_shit2(x_dot, 50, red_shift), quantum_shit2(abs(y_dot), 30, green_shift), quantum_shit2(abs(z_dot), 30, blue_shift)]

def rossler(xyz):
    x, y, z = xyz
    x_dot = -y - z
    y_dot = x + ra * y
    z_dot = rb + z * (x - rc)
    return np.array([x_dot, y_dot, z_dot]), [quantum_shit2(x_dot, 20, red_shift), quantum_shit2(abs(y_dot), 15, green_shift), quantum_shit2(abs(z_dot), 10, blue_shift)]

dt = 0.01

xyzs = np.empty((steps + 1, 3))
xyzs[0] = (0., 1. + quantum_shit(1, 0,0), 1.05 + quantum_shit(1, 0, 0))
colors = []

print(alg)

for i in range(steps):
    if(alg == "lorenz"):
        result = lorenz(xyzs[i])
    elif(alg == "rossler"):
        result = rossler(xyzs[i])
    colors.append(result[1])
    xyzs[i + 1] = xyzs[i] + result[0] * dt

colors.append(np.random.rand(3))


plt.style.use('dark_background')
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "#000000",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#000000",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

# Plot
ax = plt.figure(figsize=(5, 5)).add_subplot(projection='3d')


ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('#000000')
ax.yaxis.pane.set_edgecolor('#000000')
ax.zaxis.pane.set_edgecolor('#000000')
ax.grid(False)
ax.set_axis_off()


ax.scatter(*xyzs.T, lw=lw, s=0.3, c=np.array(colors))
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.savefig("out/quantum" + str(time.time()) + ".png", dpi=1000)
