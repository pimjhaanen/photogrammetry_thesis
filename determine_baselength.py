import numpy as np
import matplotlib.pyplot as plt

# === Settings ===
depths = [7]  # in meters
focal_lengths = {
    "Wide": 2337.803
}
disparity_errors = [0.25, 0.5, 0.75, 1]  # in pixels


for z in depths:
    plt.figure()


    baseline = np.arange(0.01, 1, 0.01)
    accuracy_max = 5  # cm
    ylim_max = 6

    plt.axhline(y=accuracy_max, color='grey', linestyle='--', label='REQ-EXP-01.2')
    plotted_point = False

    for lens_type, f in focal_lengths.items():
        for i, epsilon_d in enumerate(disparity_errors):
            epsilon_z = [(z**2 * epsilon_d) / (f * b) * 100 for b in baseline]  # cm
            label = f"{lens_type}, $\\epsilon_d$ = {epsilon_d}"
            plt.plot(baseline, epsilon_z, label=label)

            # Add black dot where Wide, ε_d=2 intersects accuracy line
            if lens_type == "Wide" and epsilon_d == 1 and not plotted_point:
                for b, ez in zip(baseline, epsilon_z):
                    if ez <= accuracy_max:
                        plt.plot(b, ez+0.02, 'ko', label='Minimum baseline')
                        print(f"baseline minimum: {round(b*2,2)} m")
                        plotted_point = True
                        break


    plt.ylim(0, ylim_max)
    plt.title(f"Estimated depth error ($\\epsilon_z$) at $z$ = {z} m")
    plt.xlabel("Baseline $B$ (m)")
    plt.ylabel("Depth error $\\epsilon_z$ (cm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
