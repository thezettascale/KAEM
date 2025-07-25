import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Compute Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
            r"\newcommand{\probP}{\text{I\kern-0.15em P}}"
        ),
    }
)

p_list = [0.1, 0.25, 0.5, 1, 2, 4, 10]
temp_cmap = plt.get_cmap("coolwarm")
temp_colors = [temp_cmap(i) for i in np.linspace(0, 1, len(p_list))]
num_temps = 100

plt.figure(figsize=(7, 5))
for p in p_list:
    temp = np.linspace(0, 1, num_temps) ** p
    label = "p = {}".format(p)
    plt.plot(temp, label=label, color=temp_colors[p_list.index(p)])
plt.xlabel(r"Schedule/Summation Index, $k$", fontsize=16)
plt.ylabel(r"Temperature, $t_{k}$", fontsize=16)
plt.title("Temperature Schedule", fontsize=20)
plt.legend(loc="upper left", fontsize=14)
plt.savefig("figures/visual/temperature_schedule.png")
