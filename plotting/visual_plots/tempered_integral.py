import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'serif', 
    'font.serif': ['Compute Modern'], 
    'axes.unicode_minus': False, 
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{bm} \newcommand{\probP}{\text{I\kern-0.15em P}}'  
})

p_list = [0.1, 0.35, 0.5, 1, 2, 4, 6, 10]
temp_cmap = plt.get_cmap('coolwarm')
temp_colors = [temp_cmap(i) for i in np.linspace(0, 1, len(p_list))]
num_temps = 100

# plt.figure(figsize=(12, 9)) 
# for p in p_list:
#     temp = np.linspace(0, 1, num_temps) ** p
#     label = "p = {}".format(p)
#     plt.plot(temp, label=label, color=temp_colors[p_list.index(p)])
# plt.xlabel(r"Schedule/Summation Index, $k$")
# plt.ylabel(r"Temperature, $t^{(k)}$")
# plt.title("Temperature Schedule", pad=70)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, columnspacing=1.5)
# plt.tight_layout()
# plt.savefig("results/temperature_schedule.png")


reduced_p = [0.35, 1, 4]
reduced_temps = [temp_colors[p_list.index(p)] for p in reduced_p]
num_temps = 30

def y_function(x):
    return (x - x ** 2) * np.exp(x)
t = np.linspace(0, 1, 100) 

for p, color in zip(reduced_p, reduced_temps):
    plt.figure(figsize=(4, 2.6))
    plt.plot(t, y_function(t), label=r"Integrand, $E_k$", color="maroon")
    plt.fill_between(t, y_function(t), alpha=0.2, color=color, label=r"Area")
    temps = np.linspace(0, 1, num_temps) ** p
    for i in range(num_temps):
        if i == 0:
            label = "Discretisation: p = {}".format(p) 
        else:
            label = None
        plt.vlines(temps[i], 0, y_function(temps[i]), color="black", label=label)
    plt.xlabel(r"Temperature $t_k$", fontsize=16)
    plt.ylabel(r"$E_k$", fontsize=16)
    # plt.title("Discretised Integral")
    plt.legend(loc="upper left", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/visual/temperature_schedule_{}.png".format(p))
    # plt.show()






