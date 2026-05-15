# networksecurity/components/anomaly_detector.py
#
# PURPOSE:
# This component adds a second layer of threat detection using a PyTorch Autoencoder.
# 
# HOW IT WORKS:
# - An Autoencoder is a neural network that learns to COMPRESS and RECONSTRUCT normal traffic
# - It is trained ONLY on benign (normal) traffic samples
# - When it sees an attack, it cannot reconstruct it well → high reconstruction error
# - We set a THRESHOLD on reconstruction error → anything above = anomaly (attack)
#
# WHY THIS IS VALUABLE:
# - Your sklearn model does supervised classification (needs labels)
# - The Autoencoder does unsupervised anomaly detection (learns what normal looks like)
# - Two different approaches = more robust system = impressive to recruiters

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_object


# ─────────────────────────────────────────────
# AUTOENCODER ARCHITECTURE
# ─────────────────────────────────────────────

class AutoEncoder(nn.Module):
    """
    A simple feedforward Autoencoder with two parts:

    ENCODER: Compresses input features into a small latent representation
        Input(40) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(8)

    DECODER: Reconstructs the original input from the compressed representation
        Latent(8) → Linear(16) → ReLU → Linear(32) → ReLU → Linear(40)

    If the model is trained on normal traffic only, it learns to reconstruct
    normal traffic well but fails on attack traffic → high reconstruction error.

    Args:
        input_dim (int): Number of input features (matches your selected features count)
    """

    def __init__(self, input_dim: int):
        super(AutoEncoder, self).__init__()

        # ENCODER: progressively compresses features
        # 40 → 32 → 16 → 8 (bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),   # compress from input_dim to 32
            nn.ReLU(),                   # activation — adds non-linearity
            nn.Linear(32, 16),           # compress from 32 to 16
            nn.ReLU(),
            nn.Linear(16, 8)             # bottleneck — most compressed representation
        )

        # DECODER: reconstructs original features from compressed form
        # 8 → 16 → 32 → 40 (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),            # expand from 8 to 16
            nn.ReLU(),
            nn.Linear(16, 32),           # expand from 16 to 32
            nn.ReLU(),
            nn.Linear(32, input_dim)     # reconstruct back to original feature size
        )

    def forward(self, x):
        """
        Forward pass:
        1. Encode input to compressed latent vector
        2. Decode latent vector back to original size
        Returns reconstructed input — should be close to original if normal traffic
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ─────────────────────────────────────────────
# ANOMALY DETECTOR COMPONENT
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    Wraps the AutoEncoder with training, threshold calculation, and prediction logic.

    Workflow:
        1. Filter only BENIGN samples from training data
        2. Train AutoEncoder on those benign samples only
        3. Calculate reconstruction error on benign samples
        4. Set threshold = mean + 2*std of benign reconstruction errors
        5. At prediction time: error > threshold = ANOMALY (attack)

    Args:
        input_dim   (int)   : Number of features (should match selected features count)
        epochs      (int)   : Number of training epochs (default: 20)
        batch_size  (int)   : Samples per training batch (default: 256)
        lr          (float) : Learning rate for Adam optimizer (default: 0.001)
        threshold   (float) : Set automatically after training, or pass manually
    """

    def __init__(
        self,
        input_dim: int,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        threshold: float = None
    ):
        # Select GPU if available, otherwise CPU
        # Google Colab will use GPU automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"AnomalyDetector using device: {self.device}")

        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold = threshold

        # Initialize the autoencoder and move it to the selected device
        self.model = AutoEncoder(input_dim=input_dim).to(self.device)

        # MSE Loss: measures how different reconstruction is from original
        # Lower loss = better reconstruction = more normal traffic
        self.criterion = nn.MSELoss()

        # Adam optimizer: updates model weights during training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Autoencoder on BENIGN traffic only.

        Steps:
        1. Filter X_train to keep only rows where y_train == 0 (benign)
        2. Convert to PyTorch tensors
        3. Create DataLoader for batch processing
        4. Train for self.epochs epochs
        5. After training, compute reconstruction errors on benign data
        6. Set threshold = mean_error + 2 * std_error
           (anything above this is considered an anomaly)

        Args:
            X_train (np.ndarray): Scaled training features
            y_train (np.ndarray): Training labels (0=benign, 1=attack)
        """
        try:
            logging.info("Starting Autoencoder training on benign traffic only")

            # Step 1: Keep only benign samples for training
            # Autoencoder learns what NORMAL looks like
            benign_mask = y_train == 0
            X_benign = X_train[benign_mask]
            logging.info(f"Training on {X_benign.shape[0]} benign samples")

            # Step 2: Convert numpy array to PyTorch float tensor
            X_tensor = torch.FloatTensor(X_benign).to(self.device)

            # Step 3: Create DataLoader — handles batching automatically
            dataset = TensorDataset(X_tensor, X_tensor)  # input = target (reconstruction)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Step 4: Training loop
            self.model.train()  # set model to training mode
            for epoch in range(self.epochs):
                epoch_loss = 0

                for batch_X, batch_y in loader:
                    # Forward pass: reconstruct the input
                    reconstructed = self.model(batch_X)

                    # Calculate how different reconstruction is from original
                    loss = self.criterion(reconstructed, batch_y)

                    # Backward pass: update weights to reduce loss
                    self.optimizer.zero_grad()  # clear previous gradients
                    loss.backward()             # compute gradients
                    self.optimizer.step()       # update weights

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)

                # Log every 5 epochs to avoid too much output
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

            # Step 5: Calculate reconstruction errors on benign data
            # This tells us what a "normal" reconstruction error looks like
            errors = self._get_reconstruction_errors(X_benign)



            # Step 6: Set threshold = mean + 2 standard deviations
            # ~95% of benign traffic will be below this threshold
            # Anything above = likely an anomaly
            self.threshold = float(np.mean(errors) + 2 * np.std(errors))
            logging.info(f"Anomaly threshold set to: {self.threshold:.6f}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        




    def _get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Private helper method.
        Passes data through the trained Autoencoder and computes
        reconstruction error for each sample.

        Reconstruction error = mean squared difference between
        original input and reconstructed output.

        Higher error = input looks different from what model learned = anomaly.

        Args:
            X (np.ndarray): Input features to evaluate

        Returns:
            np.ndarray: Reconstruction error per sample
        """
        self.model.eval()  # set model to evaluation mode (disables dropout etc.)

        with torch.no_grad():  # disable gradient computation for speed
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)

            # Calculate per-sample MSE error
            # .cpu() moves tensor back from GPU to CPU
            # .numpy() converts tensor to numpy array
            errors = torch.mean(
                (X_tensor - reconstructed) ** 2, dim=1
            ).cpu().numpy()

        return errors




    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts whether each sample is an anomaly.

        Steps:
        1. Get reconstruction error for each sample
        2. Compare to threshold
        3. Return 1 (anomaly/attack) if error > threshold, else 0 (normal)

        Args:
            X (np.ndarray): Scaled input features

        Returns:
            np.ndarray: Binary predictions — 0=normal, 1=anomaly
        """
        try:
            if self.threshold is None:
                raise ValueError("Threshold not set. Run train() before predict().")

            errors = self._get_reconstruction_errors(X)

            # Binary prediction: 1 if error exceeds threshold, 0 otherwise
            predictions = (errors > self.threshold).astype(int)
            return predictions

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Public method to get raw reconstruction errors.
        Used by Streamlit dashboard to show anomaly scores to users.

        Args:
            X (np.ndarray): Scaled input features

        Returns:
            np.ndarray: Reconstruction error per sample (float)
        """
        try:
            return self._get_reconstruction_errors(X)
        except Exception as e:
            raise NetworkSecurityException(e, sys)




    def initiate_anomaly_detector(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: str = "final_model/autoencoder.pkl"
    ) -> dict:
        """
        Main method that orchestrates the full anomaly detection pipeline.
        Called from training_pipeline.py after model training.

        Steps:
        1. Train autoencoder on benign training samples
        2. Predict on test set
        3. Calculate accuracy metrics
        4. Save the trained autoencoder to disk
        5. Return results dict for logging and Streamlit display

        Args:
            X_train    : Scaled training features
            y_train    : Training labels
            X_test     : Scaled test features
            y_test     : Test labels
            save_path  : Where to save the trained autoencoder

        Returns:
            dict with threshold, accuracy, detection_rate, false_alarm_rate
        """
        try:
            logging.info("Initiating anomaly detector pipeline")

            # Train on benign data only
            self.train(X_train, y_train)

            # Predict on full test set (benign + attack)
            y_pred = self.predict(X_test)

            # Calculate metrics
            accuracy = float(np.mean(y_pred == y_test))

            # Detection rate: how many actual attacks did we catch?
            attack_mask = y_test == 1
            detection_rate = float(np.mean(y_pred[attack_mask] == 1)) if attack_mask.sum() > 0 else 0.0

            # False alarm rate: how many benign flagged as attack?
            benign_mask = y_test == 0
            false_alarm_rate = float(np.mean(y_pred[benign_mask] == 1)) if benign_mask.sum() > 0 else 0.0

            logging.info(f"Autoencoder Results:")
            logging.info(f"  Threshold      : {self.threshold:.6f}")
            logging.info(f"  Accuracy       : {accuracy:.4f}")
            logging.info(f"  Detection Rate : {detection_rate:.4f}")
            logging.info(f"  False Alarm    : {false_alarm_rate:.4f}")

            # Save the entire AnomalyDetector object (model + threshold)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_object(save_path, self)
            logging.info(f"Autoencoder saved to: {save_path}")

            return {
                "threshold": self.threshold,
                "accuracy": accuracy,
                "detection_rate": detection_rate,
                "false_alarm_rate": false_alarm_rate
            }

        except Exception as e:
            raise NetworkSecurityException(e, sys)