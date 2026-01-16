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

for feature in top_features:
    plt.figure(figsize=(6,4))
    plt.hist(
        X_df.iloc[true_negative_idx][feature],
        bins=50,
        alpha=0.6,
        label="True Normal"
    )
    plt.hist(
        X_df.iloc[false_positive_idx][feature],
        bins=50,
        alpha=0.6,
        label="False Positive"
    )
    plt.title(f"Feature {feature} Distribution")
    plt.legend()
    plt.grid()
    plt.show()
