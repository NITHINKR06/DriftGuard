import os
import matplotlib.pyplot as plt

def save_plot(filename, phase, dpi=300):
    base_path = f"../reports/figures/{phase}"
    os.makedirs(base_path, exist_ok=True)

    full_path = os.path.join(base_path, filename)

    plt.tight_layout()
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Plot saved → {full_path}")
