import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM Neural Network for time series prediction"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictor:
    """LSTM-based time series forecasting"""
    
    def __init__(self, sequence_length: int = 24, hidden_size: int = 64, num_layers: int = 2):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.last_train_time = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LSTM using device: {self.device}")
    
    def prepare_sequences(
        self,
        values: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(values) - sequence_length):
            X.append(values[i:i + sequence_length])
            y.append(values[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        timestamps: List[datetime],
        values: List[float],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """Train LSTM model on historical data"""
        try:
            values_array = np.array(values).reshape(-1, 1)
            
            if len(values_array) < self.sequence_length + 10:
                raise ValueError(f"Need at least {self.sequence_length + 10} data points to train LSTM")
            
            # Normalize data
            values_scaled = self.scaler.fit_transform(values_array)
            
            # Prepare sequences
            X, y = self.prepare_sequences(values_scaled, self.sequence_length)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Initialize model
            self.model = LSTMNetwork(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            train_losses = []
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i + batch_size]
                    batch_y = y_tensor[i:i + batch_size]
                    
                    # Forward pass
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                train_losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            self.last_train_time = datetime.utcnow()
            
            # Calculate metrics
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).squeeze()
                mse = criterion(predictions, y_tensor).item()
                mae = torch.mean(torch.abs(predictions - y_tensor)).item()
            
            # Inverse transform for actual scale
            mae_actual = mae * (self.scaler.data_max_[0] - self.scaler.data_min_[0])
            
            logger.info(f"LSTM model trained - MSE: {mse:.6f}, MAE: {mae_actual:.2f}")
            
            return {
                'mse': float(mse),
                'mae': float(mae_actual),
                'final_loss': float(train_losses[-1]),
                'training_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(
        self,
        recent_values: List[float],
        forecast_horizon: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Generate predictions
        
        Returns:
            predictions, lower_bounds, upper_bounds
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            self.model.eval()
            
            # Prepare initial sequence
            values_array = np.array(recent_values[-self.sequence_length:]).reshape(-1, 1)
            values_scaled = self.scaler.transform(values_array)
            
            predictions = []
            current_sequence = values_scaled.copy()
            
            # Generate predictions iteratively
            with torch.no_grad():
                for _ in range(forecast_horizon):
                    # Prepare input
                    X = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
                    
                    # Predict next value
                    pred_scaled = self.model(X).cpu().numpy()[0, 0]
                    predictions.append(pred_scaled)
                    
                    # Update sequence (sliding window)
                    current_sequence = np.vstack([current_sequence[1:], [[pred_scaled]]])
            
            # Inverse transform predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            predictions_actual = self.scaler.inverse_transform(predictions_array).flatten()
            
            # Ensure non-negative predictions
            predictions_actual = np.maximum(predictions_actual, 0)
            
            # Calculate confidence intervals (simple approach using historical variance)
            historical_std = np.std(recent_values)
            lower_bounds = predictions_actual - 1.96 * historical_std
            upper_bounds = predictions_actual + 1.96 * historical_std
            
            lower_bounds = np.maximum(lower_bounds, 0)
            upper_bounds = np.maximum(upper_bounds, 0)
            
            return (
                predictions_actual.tolist(),
                lower_bounds.tolist(),
                upper_bounds.tolist()
            )
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            raise
    
    def detect_anomalies(
        self,
        values: List[float],
        threshold: float = 2.5
    ) -> List[int]:
        """Detect anomalies using reconstruction error"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            values_array = np.array(values).reshape(-1, 1)
            values_scaled = self.scaler.transform(values_array)
            
            X, y = self.prepare_sequences(values_scaled, self.sequence_length)
            X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).squeeze()
                errors = torch.abs(predictions - y_tensor).cpu().numpy()
            
            # Calculate threshold based on mean and std of errors
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            anomaly_threshold = mean_error + threshold * std_error
            
            # Find anomaly indices (adjusted for sequence offset)
            anomaly_indices = [
                i + self.sequence_length 
                for i, error in enumerate(errors) 
                if error > anomaly_threshold
            ]
            
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"LSTM anomaly detection failed: {e}")
            raise
    
    def save_model(self, path: str):
        """Save model state"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.scaler = checkpoint['scaler']
        
        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
    
    def needs_retraining(self, max_age_hours: int = 24) -> bool:
        """Check if model needs retraining based on age"""
        if not self.is_trained or not self.last_train_time:
            return True
        
        age = datetime.utcnow() - self.last_train_time
        return age > timedelta(hours=max_age_hours)