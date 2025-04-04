import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'serif', 
    'font.serif': ['Compute Modern'], 
    'axes.unicode_minus': False, 
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{bm} \newcommand{\probP}{\text{I\kern-0.15em P}}'  
})

num_param_updates = 100000  # Example value
num_cycles = 1  # Example value
initial_p = 2  # Example value
end_p = 0.5 # Example value

x = np.linspace(0, 2 * np.pi * (num_cycles+0.5), num_param_updates + 1)
p = initial_p + (end_p - initial_p) * 0.5 * (1 - np.cos(x))

plt.figure(figsize=(5, 5))
plt.plot(x, p)
plt.xlabel('Parameter updates', fontsize=14)
plt.ylabel('p', fontsize=14)
plt.title('Cyclic p schedule', fontsize=14)
# plt.show()
plt.savefig('figures/visual/cyclic_annealing.png', dpi=300)