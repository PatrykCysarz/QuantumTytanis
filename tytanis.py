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

argv = sys.argv[1:]
short_opts = "st:"
long_opts = ["steps="]

try:
	args, opts = getopt.getopt(argv, short_opts, long_opts)
except getopt.error as err:
	print (str(err))
  
## Loop through arguments
for current_argument, current_value in args:
    if current_argument in ("-st", "--steps"):
        steps = int(current_value)


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

def quantum_shit2(position, scale):
    simulator = AerSimulator()
    qreg_q = QuantumRegister(1, 'q')
    creg_c = ClassicalRegister(1, 'c')
    color_circuit = QuantumCircuit(qreg_q, creg_c)
    color_circuit.reset(qreg_q[0])
    color_circuit.x(qreg_q[0])
    color_circuit.rx(pi/(2 + (position / scale)), qreg_q[0])
    color_circuit.measure(qreg_q[0], creg_c[0])
    compiled_circuit = transpile(color_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=100)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
 
    print(counts)
 
    if '0' not in counts:
        counts['0'] = 0.001
 
    if '1' not in counts:
        counts['1'] = 0.001
 
    result = counts['0'] / counts['1']
 
    if result > 1:
        result = 1
 
    print(result)
 
    return result

quantum_shit(1, 2, 3)


def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    colors = []

    asd = sqrt(quantum_shit(x, y, z))
    x_dot = s * (y - x) * asd
    y_dot = r * x - y - x * z * asd
    z_dot = x * y - b * z * asd
    return np.array([x_dot, y_dot, z_dot]), [quantum_shit2(x_dot, 100), quantum_shit2(y_dot, 40), quantum_shit2(z_dot, 40)]


dt = 0.01

xyzs = np.empty((steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (0., 1., 1.05)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
colors = []

for i in range(steps):
    lorenz_result = lorenz(xyzs[i])
    colors.append(lorenz_result[1])
    xyzs[i + 1] = xyzs[i] + lorenz_result[0] * dt

colors.append(np.random.rand(3))


plt.style.use('dark_background')
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "#333333",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

# Plot
ax = plt.figure().add_subplot(projection='3d')


ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('#000000')
ax.yaxis.pane.set_edgecolor('#000000')
ax.zaxis.pane.set_edgecolor('#000000')
ax.grid(False)
ax.set_axis_off()


ax.scatter(*xyzs.T, lw=0.5, s=0.4, c=np.array(colors))
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

# plt.scatter(x, y, 1, "#ff0000", cmap="coolwarm")
# plt.colorbar()
plt.savefig("quantum" + str(time.time()) + ".png")

# plt.scatter(x_cercle, y_cercle, c=tab_cercle[1], s=1, cmap="coolwarm")
# plt.scatter(x, y, c=tab[1], s=1, cmap="coolwarm")
# plt.colorbar()
# plt.savefig("quantum" + str(time.time()) + ".png")


# qc_cercle.draw(output="mpl", filename="test.png")
# print(qc_cercle)
