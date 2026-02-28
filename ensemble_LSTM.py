import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MultiModalWeatherDataset(Dataset):
    # Reload the Methods
    def __init__(self, X_seq, X_current, y, mode='dual'):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_current = torch.FloatTensor(X_current) if X_current is not None else None
        self.y = torch.FloatTensor(y)
        self.mode = mode
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        if self.mode == 'dual' and self.X_current is not None:
            return self.X_seq[idx], self.X_current[idx], self.y[idx]
        else:
            return self.X_seq[idx], self.y[idx]

class AdaptiveLSTM(nn.Module):
    # 
    def __init__(self, seq_input_size, current_input_size=0, hidden_size=64, num_layers=2):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.current_input_size = current_input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(seq_input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        if current_input_size > 0:
            self.current_fc = nn.Sequential(
                nn.Linear(current_input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # fusion_input_size = hidden_size + 32
        # else:
        #     fusion_input_size = hidden_size
    
        if current_input_size > 0:
            self.fusion_layers_with_current = nn.Sequential(
                nn.Linear(hidden_size + 32, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        self.fusion_layers_lstm_only = nn.Sequential(
            nn.Linear(hidden_size, 64), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_seq, x_current=None):
        lstm_out, _ = self.lstm(x_seq)
        lstm_features = lstm_out[:, -1, :]  # 取最后一个时间步
    
        if x_current is not None and self.current_input_size > 0:
            current_features = self.current_fc(x_current)
            combined = torch.cat([lstm_features, current_features], dim=1)
            output = self.fusion_layers_with_current(combined)
        else:
            output = self.fusion_layers_lstm_only(lstm_features)
        return output

class AdaptiveWeatherEnsemble:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length = 14
        
        self.lstm_model = None
        # self.xgb_model = ScenarioAwareXGBoost()
        
        self.weight_predictor = None
    
    def load_data(self):
        data = pd.read_csv('weather_prediction_dataset.csv')
        
        target_cols = [col for col in data.columns if 'BASEL_temp_mean' in col or 'BASEL_temp' in col]
        if not target_cols:
            temp_cols = [col for col in data.columns if 'temp_mean' in col]
            target_col = temp_cols[0] if temp_cols else 'BASEL_temp_mean'
        else:
            target_col = target_cols[0]
        
        feature_cols = []
        for col in data.columns:
            if col not in ['DATE', 'MONTH'] and col != target_col:
                if data[col].dtype in ['float64', 'int64']:
                    feature_cols.append(col)
        
        data = data.dropna()
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        print(f"Target: {target_col}")
        print(f"Features: {len(feature_cols)}")
        print(f"Data shape: {X.shape}")
        
        return X, y, feature_cols, target_col
    
    def create_dual_sequences(self, X, y):
        X_seq, X_current, y_seq = [], [], []
        
        for i in range(self.seq_length, len(X)):
            X_seq.append(X[i-self.seq_length:i])
            X_current.append(X[i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(X_current), np.array(y_seq)
    
    def create_weight_predictor(self, input_size):
        return nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2), nn.Softmax(dim=1)).to(self.device)
    
    def train(self, X, y):
        train_size = int(0.8 * len(X))
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        X_train_seq, X_train_current, y_train_seq = self.create_dual_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, X_test_current, y_test_seq = self.create_dual_sequences(X_test_scaled, y_test_scaled)
        
        self.lstm_model = AdaptiveLSTM(seq_input_size=X_train_seq.shape[2], current_input_size=X_train_current.shape[1], hidden_size=64, num_layers=2).to(self.device)
        
        self._train_lstm(X_train_seq, X_train_current, y_train_seq)
        self.xgb_model.fit(X_train_seq, X_train_current, y_train_seq)
        self.weight_predictor = self.create_weight_predictor(X_train_current.shape[1] + 10)  # +10 for sequence stats
        self._train_weight_predictor(X_train_seq, X_train_current, y_train_seq)
        
        return X_test_seq, X_test_current, y_test_seq
    
    def _train_lstm(self, X_seq, X_current, y):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        dataset = MultiModalWeatherDataset(X_seq, X_current, y, mode='dual')
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.lstm_model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_seq, batch_current, batch_y in dataloader:
                batch_seq = batch_seq.to(self.device)
                batch_current = batch_current.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_seq, batch_current).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"LSTM Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _train_weight_predictor(self, X_seq, X_current, y):
        lstm_preds = self._get_lstm_predictions(X_seq, X_current)
        xgb_preds = self.xgb_model.predict(X_seq, X_current)
        
        weight_features = []
        for i in range(len(X_seq)):
            seq_stats = np.concatenate([np.mean(X_seq[i], axis=0)[:6], np.std(X_seq[i], axis=0)[:6]])
            # feature engineering
            current_features = X_current[i]
            combined_features = np.concatenate([current_features, seq_stats])
            weight_features.append(combined_features)
        
        weight_features = np.array(weight_features)
        
        lstm_errors = np.abs(lstm_preds - y)
        xgb_errors = np.abs(xgb_preds - y)
        
        optimal_weights = []
        for i in range(len(y)):
            if lstm_errors[i] + xgb_errors[i] == 0:
                w_lstm, w_xgb = 0.5, 0.5
            else:
                w_xgb = lstm_errors[i] / (lstm_errors[i] + xgb_errors[i])
                w_lstm = 1 - w_xgb
            optimal_weights.append([w_lstm, w_xgb])
        
        optimal_weights = np.array(optimal_weights)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.weight_predictor.parameters(), lr=0.001)
        weight_features_tensor = torch.FloatTensor(weight_features).to(self.device)
        optimal_weights_tensor = torch.FloatTensor(optimal_weights).to(self.device)
        
        for epoch in range(50):
            optimizer.zero_grad()
            predicted_weights = self.weight_predictor(weight_features_tensor)
            loss = criterion(predicted_weights, optimal_weights_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Weight Predictor Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _get_lstm_predictions(self, X_seq, X_current):
        self.lstm_model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_seq), 64):
                batch_seq = torch.FloatTensor(X_seq[i:i+64]).to(self.device)
                batch_current = torch.FloatTensor(X_current[i:i+64]).to(self.device) if X_current is not None else None
                pred = self.lstm_model(batch_seq, batch_current)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions).flatten()
    
    def predict(self, X_seq, X_current=None, mode='adaptive'):
        # if mode == 'temporal':
        #     X_current = None

        lstm_preds = self._get_lstm_predictions(X_seq, X_current)
        xgb_preds = self.xgb_model.predict(X_seq, X_current)
        
        if mode == 'adaptive' and X_current is not None and self.weight_predictor is not None:
            weight_features = []
            for i in range(len(X_seq)):
                seq_stats = np.concatenate([
                    np.mean(X_seq[i], axis=0)[:5],
                    np.std(X_seq[i], axis=0)[:5]
                ])
                current_features = X_current[i]
                combined_features = np.concatenate([current_features, seq_stats])
                weight_features.append(combined_features)
            
            weight_features = torch.FloatTensor(weight_features).to(self.device)
            with torch.no_grad():
                weights = self.weight_predictor(weight_features).cpu().numpy()
            
            ensemble_preds = weights[:, 0] * lstm_preds + weights[:, 1] * xgb_preds
        else:
            if X_current is not None:
                ensemble_preds = 0.4 * lstm_preds + 0.6 * xgb_preds
            else:
                ensemble_preds = 0.7 * lstm_preds + 0.3 * xgb_preds
        
        return lstm_preds, xgb_preds, ensemble_preds

def main():
    predictor = AdaptiveWeatherEnsemble()
    X, y, _, _ = predictor.load_data()
    X_test_seq, X_test_current, y_test_seq = predictor.train(X, y)
    results, _ = predictor.evaluate_scenarios(X_test_seq, X_test_current, y_test_seq)
    
    # predictor.plot_scenario_comparison(results)

    best_ensemble_r2 = max(results[s]['ensemble_r2'] for s in results.keys())
    # best_scenario = [s for s in results.keys() if results[s]['ensemble_r2'] == best_ensemble_r2][0]
    
    print(f"Ensemble R² Score: {best_ensemble_r2:.4f}")

if __name__ == "__main__":
    main()