import numpy as np
import matplotlib.pyplot as plt

# === Settings ===
depths = [7, 15]  # in meters
focal_lengths = {
    "Linear": 2448.228,
    "Wide": 2337.803
}
disparity_errors = [0.25, 0.5, 0.75, 1]  # in pixels

# Define plot colours (tints for each lens)
colors = {
    "Linear": ['#191970', '#0000FF', '#6495ED', '#ADD8E6'],  # tints of blue
    "Wide": ['#556B2F', '#228B22', '#ADFF2F', '#FFFF00']     # tints of green/yellow
}

for z in depths:
    plt.figure()

    if z == 7:
        baseline = np.arange(0.01, 1, 0.01)
        accuracy_max = 5  # cm
        ylim_max = 6
    else:
        baseline = np.arange(0.10, 3, 0.01)
        accuracy_max = 10  # cm
        ylim_max = 20
    plt.axhline(y=accuracy_max, color='grey', linestyle='--', label='REQ-EXP-01.2')
    plotted_point = False

    for lens_type, f in focal_lengths.items():
        for i, epsilon_d in enumerate(disparity_errors):
            epsilon_z = [(z**2 * epsilon_d) / (f * b) * 100 for b in baseline]  # cm
            label = f"{lens_type}, $\\epsilon_d$ = {epsilon_d}"
            plt.plot(baseline, epsilon_z, label=label, color=colors[lens_type][i])

            # Add black dot where Wide, ε_d=2 intersects accuracy line
            if lens_type == "Wide" and epsilon_d == 1 and not plotted_point:
                for b, ez in zip(baseline, epsilon_z):
                    if ez <= accuracy_max:
                        plt.plot(b, ez+0.03, 'ko', label='Minimum baseline')
                        print(f"baseline minimum: {round(b*2,2)} m")
                        plotted_point = True
                        break


    plt.ylim(0, ylim_max)
    plt.title(f"Estimated depth error ($\\epsilon_z$) at $z$ = {z} m")
    plt.xlabel("Baseline $b$ (m)")
    plt.ylabel("Depth error $\\epsilon_z$ (cm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
