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

class WeatherDataset(Dataset):
    # reload
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.15)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        # self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class WeatherPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length = 14
        
    def load_data(self):
        data = pd.read_csv('weather_prediction_dataset.csv')
        
        target_cols = [col for col in data.columns if 'DE_BILT_temp_mean' in col or 'DE_BILT_temp' in col]
        if not target_cols:
            temp_cols = [col for col in data.columns if 'temp_mean' in col]
            target_col = temp_cols[0] if temp_cols else 'DE_BILT_temp_mean'
        else:
            target_col = target_cols[0]
        
        if target_col not in data.columns:
            target_col = 'DE_BILT_temp_mean'
            
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
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X)):
            X_seq.append(X[i-self.seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X, y):
        train_size = int(0.8 * len(X))
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        self.model = WeatherLSTM(X_train_seq.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        train_dataset = WeatherDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        best_loss = float('inf')
        patience = 0
        
        for _ in range(100):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
                torch.save(self.model.state_dict(), 'best_weather_model.pth')
            else:
                patience += 1
                
            if patience > 15:
                break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            self.model.load_state_dict(torch.load('best_weather_model.pth'))
        except:
            pass
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200, 
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        X_xgb_train = X_train_scaled[self.seq_length:]
        y_xgb_train = y_train_scaled[self.seq_length:]
        self.xgb_model.fit(X_xgb_train, y_xgb_train)
        
        return X_test_seq, y_test_seq, X_test_scaled
    
    def predict(self, X_test_seq, X_test_scaled):
        self.model.eval()
        lstm_preds = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_seq), 64):
                batch = torch.FloatTensor(X_test_seq[i:i+64]).to(self.device)
                pred = self.model(batch)
                lstm_preds.extend(pred.cpu().numpy())
        
        lstm_preds = np.array(lstm_preds).flatten()
        
        X_xgb_test = X_test_scaled[self.seq_length:]
        xgb_preds = self.xgb_model.predict(X_xgb_test)
        
        lstm_preds_rescaled = self.scaler_y.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()
        xgb_preds_rescaled = self.scaler_y.inverse_transform(xgb_preds.reshape(-1, 1)).flatten()
        
        ensemble_weight = 0.45
        ensemble_preds = ensemble_weight * lstm_preds + (1 - ensemble_weight) * xgb_preds
        ensemble_preds_rescaled = self.scaler_y.inverse_transform(ensemble_preds.reshape(-1, 1)).flatten()
        
        return lstm_preds_rescaled, xgb_preds_rescaled, ensemble_preds_rescaled
    
    def evaluate(self, y_true_scaled, lstm_preds, xgb_preds, ensemble_preds):
        y_true = self.scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        
        lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_preds))
        xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_preds))
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
        
        lstm_r2 = r2_score(y_true, lstm_preds)
        xgb_r2 = r2_score(y_true, xgb_preds)
        ensemble_r2 = r2_score(y_true, ensemble_preds)
        
        lstm_mae = mean_absolute_error(y_true, lstm_preds)
        xgb_mae = mean_absolute_error(y_true, xgb_preds)
        ensemble_mae = mean_absolute_error(y_true, ensemble_preds)
        
        return {
            'lstm_rmse': lstm_rmse, 'xgb_rmse': xgb_rmse, 'ensemble_rmse': ensemble_rmse,
            'lstm_r2': lstm_r2, 'xgb_r2': xgb_r2, 'ensemble_r2': ensemble_r2,
            'lstm_mae': lstm_mae, 'xgb_mae': xgb_mae, 'ensemble_mae': ensemble_mae,
            'y_true': y_true
        }
    
    def plot_results(self, metrics, lstm_preds, xgb_preds, ensemble_preds):
        y_true = metrics['y_true']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Weather Prediction Results', fontsize=16)
        
        min_temp = min(np.min(y_true), np.min(lstm_preds), np.min(xgb_preds), np.min(ensemble_preds))
        max_temp = max(np.max(y_true), np.max(lstm_preds), np.max(xgb_preds), np.max(ensemble_preds))
        
        axes[0,0].scatter(y_true, lstm_preds, alpha=0.6, color='blue', s=20)
        axes[0,0].plot([min_temp, max_temp], [min_temp, max_temp], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Temperature (°C)')
        axes[0,0].set_ylabel('Predicted Temperature (°C)')
        axes[0,0].set_title(f'Enhanced LSTM Model\nRMSE: {metrics["lstm_rmse"]:.2f}, R²: {metrics["lstm_r2"]:.4f}')
        axes[0,0].legend(['Perfect Prediction', 'LSTM Predictions'])
        
        axes[0,1].scatter(y_true, xgb_preds, alpha=0.6, color='green', s=20)
        axes[0,1].plot([min_temp, max_temp], [min_temp, max_temp], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Temperature (°C)')
        axes[0,1].set_ylabel('Predicted Temperature (°C)')
        axes[0,1].set_title(f'Enhanced XGBoost Model\nRMSE: {metrics["xgb_rmse"]:.2f}, R²: {metrics["xgb_r2"]:.4f}')
        axes[0,1].legend(['Perfect Prediction', 'XGBoost Predictions'])
        
        axes[0,2].scatter(y_true, ensemble_preds, alpha=0.6, color='purple', s=20)
        axes[0,2].plot([min_temp, max_temp], [min_temp, max_temp], 'r--', lw=2)
        axes[0,2].set_xlabel('Actual Temperature (°C)')
        axes[0,2].set_ylabel('Predicted Temperature (°C)')
        axes[0,2].set_title(f'Optimized Ensemble Model\nRMSE: {metrics["ensemble_rmse"]:.2f}, R²: {metrics["ensemble_r2"]:.4f}')
        axes[0,2].legend(['Perfect Prediction', 'Ensemble Predictions'])
        
        x_range = range(len(y_true))
        axes[1,0].plot(x_range, y_true, label='Actual Temperature', color='red', linewidth=2)
        axes[1,0].plot(x_range, lstm_preds, label='Enhanced LSTM', color='blue', alpha=0.7)
        axes[1,0].set_xlabel('Days')
        axes[1,0].set_ylabel('Temperature (°C)')
        axes[1,0].set_title('Enhanced LSTM Time Series')
        axes[1,0].legend()
        
        axes[1,1].plot(x_range, y_true, label='Actual Temperature', color='red', linewidth=2)
        axes[1,1].plot(x_range, xgb_preds, label='Enhanced XGBoost', color='green', alpha=0.7)
        axes[1,1].set_xlabel('Days')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].set_title('Enhanced XGBoost Time Series')
        axes[1,1].legend()
        
        axes[1,2].plot(x_range, y_true, label='Actual Temperature', color='red', linewidth=2)
        axes[1,2].plot(x_range, ensemble_preds, label='Optimized Ensemble', color='purple', alpha=0.7)
        axes[1,2].set_xlabel('Days')
        axes[1,2].set_ylabel('Temperature (°C)')
        axes[1,2].set_title('Optimized Ensemble Time Series')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.show()
        
        metrics_df = pd.DataFrame({
            'Model': ['Enhanced LSTM', 'Enhanced XGBoost', 'Optimized Ensemble'],
            'RMSE': [metrics['lstm_rmse'], metrics['xgb_rmse'], metrics['ensemble_rmse']],
            'R²': [metrics['lstm_r2'], metrics['xgb_r2'], metrics['ensemble_r2']],
            'MAE': [metrics['lstm_mae'], metrics['xgb_mae'], metrics['ensemble_mae']]
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['#1f77b4', '#2ca02c', '#9467bd']
        
        bars1 = axes[0].bar(metrics_df['Model'], metrics_df['RMSE'], color=colors)
        axes[0].set_title('Root Mean Square Error (RMSE)')
        axes[0].set_ylabel('RMSE (°C)')
        axes[0].set_ylim(0, max(metrics_df['RMSE']) * 1.1)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        bars2 = axes[1].bar(metrics_df['Model'], metrics_df['R²'], color=colors)
        axes[1].set_title('R-squared Score')
        axes[1].set_ylabel('R²')
        axes[1].set_ylim(0.9, 1.0)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        bars3 = axes[2].bar(metrics_df['Model'], metrics_df['MAE'], color=colors)
        axes[2].set_title('Mean Absolute Error (MAE)')
        axes[2].set_ylabel('MAE (°C)')
        axes[2].set_ylim(0, max(metrics_df['MAE']) * 1.1)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df

def main():
    predictor = WeatherPredictor()
    
    X, y, feature_cols, target_col = predictor.load_data()
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Temperature Distribution', fontweight='bold')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 2)
    data_for_viz = pd.DataFrame(X[:, :10])
    correlation_matrix = data_for_viz.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix', fontweight='bold')
    
    plt.subplot(2, 3, 3)
    plt.plot(y[:365], linewidth=1.5, color='darkblue')
    plt.title('Temperature Trend (First Year)', fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Temperature (°C)')
    
    plt.subplot(2, 3, 4)
    monthly_temps = []
    for month in range(1, 13):
        month_mask = np.arange(len(y)) % 365 // 30 + 1 == month
        if np.any(month_mask):
            monthly_temps.append(y[month_mask])
    
    if monthly_temps:
        box_plot = plt.boxplot(monthly_temps[:12], patch_artist=True)
        for patch, color in zip(box_plot['boxes'], plt.cm.Set3(np.linspace(0, 1, 12))):
            patch.set_facecolor(color)
        plt.title('Temperature by Month', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
    
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(range(len(y[:200])), y[:200], alpha=0.6, c=y[:200], cmap='coolwarm', s=30)
    plt.colorbar(scatter, label='Temperature (°C)')
    plt.title('Temperature Scatter (First 200 days)', fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Temperature (°C)')
    
    plt.subplot(2, 3, 6)
    rolling_mean = pd.Series(y).rolling(30).mean()
    rolling_std = pd.Series(y).rolling(30).std()
    plt.plot(rolling_mean[:365], label='30-Day Mean', linewidth=2, color='darkgreen')
    plt.fill_between(range(len(rolling_mean[:365])), rolling_mean[:365] - rolling_std[:365], rolling_mean[:365] + rolling_std[:365], alpha=0.3, label='±1 Std Dev', color='lightgreen')
    plt.title('30-Day Rolling Statistics', fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Training enhanced models...")
    X_test_seq, y_test_seq, X_test_scaled = predictor.train(X, y)
    
    print("Making predictions with optimized ensemble...")
    lstm_preds, xgb_preds, ensemble_preds = predictor.predict(X_test_seq, X_test_scaled)
    
    metrics = predictor.evaluate(y_test_seq, lstm_preds, xgb_preds, ensemble_preds)
    
    metrics_df = predictor.plot_results(metrics, lstm_preds, xgb_preds, ensemble_preds)
    
    print("\nEnhanced Performance Summary:")
    print("="*60)
    print(metrics_df.round(4))
    
    best_model_idx = metrics_df['R²'].idxmax()
    print(f"\nBest Model: {metrics_df.loc[best_model_idx, 'Model']}")
    print(f"Best R² Score: {metrics_df['R²'].max():.4f}")
    print(f"Best RMSE: {metrics_df['RMSE'].min():.4f}")
    
    ensemble_score = metrics_df.loc[metrics_df['Model']=='Optimized Ensemble', 'R²'].values[0]
    print(f"Ensemble R² Score: {ensemble_score:.4f}")
    
    if ensemble_score == metrics_df['R²'].max():
        print("🎉 Ensemble model achieved the best performance!")
    else:
        print(f"Ensemble performance: {ensemble_score:.4f} vs Best: {metrics_df['R²'].max():.4f}")

if __name__ == "__main__":
    main()