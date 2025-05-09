import numpy as np
import torch

class EarlyStopping:
    """
    Stop training when validation loss doesn’t improve after a given patience.
    Saves the best model to `best_model_path`.
    """
    def __init__(self, patience=20, delta=0.005, best_model_path='checkpoint.pth', verbose=False):
        self.patience = patience
        self.delta = delta
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, save=True):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if save:
                torch.save(model.state_dict(), self.best_model_path)
                if self.verbose:
                    print(f"Validation loss improved to {val_loss:.4f}. Saved model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")