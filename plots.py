import numpy as np
import matplotlib.pyplot as plt

def plot_preds_vs_target(mus: np.ndarray, targets: np.ndarray, vars: np.ndarray, filename):
    """
    preds: 1D array of your model predictions
    targets: 1D array of the true values (same length as preds)
    """
    stds = np.sqrt(vars).detach().cpu().numpy()
    x = np.arange(len(mus))

    plt.figure(figsize=(10, 6))
    plt.plot(x, mus, label="Predicted μ", color="C0")

    lower = mus - stds
    upper = mus + stds

    plt.fill_between(x, lower, upper,
                     color="C0", alpha=0.2,
                     label="±1sigma uncertainty")
    # plot true target
    plt.plot(x, targets, label="True value", color="C1", linestyle="--")

    plt.xlabel("Sample index")
    plt.ylabel("Price")
    plt.title(f"{filename}: predictions ± uncertainty vs. true")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}_with_uncertainty.png")
    plt.close()

def plot_loss(loss_list, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label="Loss", color="C0")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{filename}: Loss")
    plt.tight_layout()
    plt.savefig(f"{filename}_loss.png")
    plt.close()

