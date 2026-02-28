import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
warnings.filterwarnings('ignore')

class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HouseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1):
        super(HouseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(8)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        
        out = self.relu(self.batch_norm1(self.fc1(out)))
        out = self.dropout(out)
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.output(out)
        return out

class HousePricePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.seq_length = 5
        self.model = None
        self.xgb_model = None
        self.model_path = 'best_house_model.pth'
        
    def load_and_analyze_data(self, file_path):
        df = pd.read_csv(file_path)
        
        print("Dataset Shape:", df.shape)
        print("\nColumn Names:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nData Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        return df
    
    def feature_engineering(self, df):
        df = df.copy()
        
        df = df.dropna()
        
        if 'PRICE' in df.columns:
            target_col = 'PRICE'
        elif 'price' in df.columns:
            target_col = 'price'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = numeric_cols[-1]
        
        print(f"Using target column: {target_col}")
        
        df = df[df[target_col] > 0]
        
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[target_col] < (Q1 - 1.5 * IQR)) | (df[target_col] > (Q3 + 1.5 * IQR)))]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                try:
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                except:
                    df[col] = 0
            else:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0
        
        for col in numeric_cols:
            if col != target_col:
                df[col] = df[col].fillna(df[col].median())
                
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    df.loc[inf_mask, col] = df[col][~inf_mask].median()
        
        if len(numeric_cols) > 3:
            feature_correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            print("\nTop Feature Correlations with Target:")
            print(feature_correlations.head(10))
        
        if 'BEDS' in df.columns and 'BATH' in df.columns:
            df['bed_bath_ratio'] = df['BEDS'] / (df['BATH'] + 0.1)
        
        if 'PROPERTYSQFT' in df.columns:
            df['price_per_sqft'] = df[target_col] / (df['PROPERTYSQFT'] + 1)
            df['log_sqft'] = np.log1p(df['PROPERTYSQFT'])
        
        if 'BEDS' in df.columns:
            df['beds_squared'] = df['BEDS'] ** 2
        
        feature_cols = [col for col in df.columns if col != target_col]
        return df, feature_cols, target_col
    
    def create_sequences(self, data, target, feature_cols):
        sequences = []
        targets = []
        
        if len(data) < self.seq_length:
            X_simple = data[feature_cols].values
            y_simple = target.values
            return X_simple, y_simple
        
        for i in range(self.seq_length, len(data)):
            seq = data[feature_cols].iloc[i-self.seq_length:i].values
            sequences.append(seq)
            targets.append(target.iloc[i])
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, df, feature_cols, target_col, is_training=True):
        if is_training:
            features_scaled = self.feature_scaler.fit_transform(df[feature_cols])
            target_scaled = self.price_scaler.fit_transform(df[[target_col]])
        else:
            features_scaled = self.feature_scaler.transform(df[feature_cols])
            target_scaled = self.price_scaler.transform(df[[target_col]])
        
        df_scaled = pd.DataFrame(features_scaled, columns=feature_cols)
        target_series = pd.Series(target_scaled.flatten())
        
        X_seq, y_seq = self.create_sequences(df_scaled, target_series, feature_cols)
        
        if len(X_seq.shape) == 2:
            X_seq = X_seq.reshape(-1, 1, X_seq.shape[1])
        
        X_xgb = features_scaled[self.seq_length:] if len(features_scaled) > self.seq_length else features_scaled
        y_xgb = target_scaled.flatten()[self.seq_length:] if len(target_scaled) > self.seq_length else target_scaled.flatten()
        
        return X_seq, y_seq, X_xgb, y_xgb
    
    def train(self, df, feature_cols, target_col):
        X_seq, y_seq, X_xgb, y_xgb = self.prepare_data(df, feature_cols, target_col, is_training=True)
        
        if len(X_seq) == 0:
            print("Not enough data for sequence training")
            return
        
        print(f"Training data shapes: X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
        
        input_size = X_seq.shape[2] if len(X_seq.shape) == 3 else X_seq.shape[1]
        self.model = HouseLSTM(input_size=input_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        dataset = HouseDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)//4), shuffle=True)
        
        self.model.train()
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        print("Starting LSTM training...")
        for epoch in range(100):
            total_loss = 0
            batch_count = 0
            
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                
                if len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)
                if len(y_batch.shape) == 0:
                    y_batch = y_batch.unsqueeze(0)
                
                loss = criterion(outputs, y_batch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            if batch_count == 0:
                continue
                
            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, self.model_path)
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"LSTM training completed. Best loss: {best_loss:.6f}")
        else:
            print("LSTM training failed - using random model")
        
        print("Training XGBoost...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        try:
            self.xgb_model.fit(X_xgb, y_xgb)
            print("XGBoost training completed.")
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            self.xgb_model = None
    
    def predict(self, df, feature_cols, target_col):
        X_seq, y_seq, X_xgb, y_xgb = self.prepare_data(df, feature_cols, target_col, is_training=False)
        
        if len(X_seq) == 0:
            print("Not enough data for prediction")
            return None, None, None, None
        
        self.model.eval()
        lstm_preds = []
        
        with torch.no_grad():
            batch_size = min(32, len(X_seq))
            for i in range(0, len(X_seq), batch_size):
                batch = torch.FloatTensor(X_seq[i:i+batch_size]).to(self.device)
                pred = self.model(batch)
                lstm_preds.extend(pred.cpu().numpy())
        
        lstm_preds = np.array(lstm_preds).flatten()
        
        lstm_preds = np.nan_to_num(lstm_preds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if self.xgb_model is not None:
            try:
                xgb_preds = self.xgb_model.predict(X_xgb)
                xgb_preds = np.nan_to_num(xgb_preds, nan=0.0, posinf=1.0, neginf=-1.0)
            except:
                xgb_preds = np.zeros_like(lstm_preds)
        else:
            xgb_preds = np.zeros_like(lstm_preds)
        
        ensemble_preds = 0.7 * lstm_preds + 0.3 * xgb_preds
        
        try:
            lstm_preds_rescaled = self.price_scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()
            xgb_preds_rescaled = self.price_scaler.inverse_transform(xgb_preds.reshape(-1, 1)).flatten()
            ensemble_preds_rescaled = self.price_scaler.inverse_transform(ensemble_preds.reshape(-1, 1)).flatten()
        except:
            lstm_preds_rescaled = lstm_preds
            xgb_preds_rescaled = xgb_preds
            ensemble_preds_rescaled = ensemble_preds
        
        lstm_preds_rescaled = np.nan_to_num(lstm_preds_rescaled, nan=0.0, posinf=1e6, neginf=0.0)
        xgb_preds_rescaled = np.nan_to_num(xgb_preds_rescaled, nan=0.0, posinf=1e6, neginf=0.0)
        ensemble_preds_rescaled = np.nan_to_num(ensemble_preds_rescaled, nan=0.0, posinf=1e6, neginf=0.0)
        
        return lstm_preds_rescaled, xgb_preds_rescaled, ensemble_preds_rescaled, y_seq
    
    def evaluate(self, y_true_scaled, lstm_preds, xgb_preds, ensemble_preds):
        try:
            y_true = self.price_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        except:
            y_true = y_true_scaled
        
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=0.0)
        
        min_len = min(len(y_true), len(lstm_preds), len(xgb_preds), len(ensemble_preds))
        y_true = y_true[:min_len]
        lstm_preds = lstm_preds[:min_len]
        xgb_preds = xgb_preds[:min_len]
        ensemble_preds = ensemble_preds[:min_len]
        
        try:
            lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_preds))
            xgb_rmse = np.sqrt(mean_squared_error(y_true, xgb_preds))
            ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))
        except:
            lstm_rmse = float('inf')
            xgb_rmse = float('inf')
            ensemble_rmse = float('inf')
        
        try:
            lstm_r2 = max(0, r2_score(y_true, lstm_preds))
            xgb_r2 = max(0, r2_score(y_true, xgb_preds))
            ensemble_r2 = max(0, r2_score(y_true, ensemble_preds))
        except:
            lstm_r2 = 0.0
            xgb_r2 = 0.0
            ensemble_r2 = 0.0
        
        try:
            lstm_mae = mean_absolute_error(y_true, lstm_preds)
            xgb_mae = mean_absolute_error(y_true, xgb_preds)
            ensemble_mae = mean_absolute_error(y_true, ensemble_preds)
        except:
            lstm_mae = float('inf')
            xgb_mae = float('inf')
            ensemble_mae = float('inf')
        
        return {
            'lstm_rmse': lstm_rmse, 'xgb_rmse': xgb_rmse, 'ensemble_rmse': ensemble_rmse,
            'lstm_r2': lstm_r2, 'xgb_r2': xgb_r2, 'ensemble_r2': ensemble_r2,
            'lstm_mae': lstm_mae, 'xgb_mae': xgb_mae, 'ensemble_mae': ensemble_mae,
            'y_true': y_true
        }
    
    def plot_results(self, metrics, lstm_preds, xgb_preds, ensemble_preds):
        y_true = metrics['y_true']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('House Price Prediction Results', fontsize=16)
        
        max_val = max(np.max(y_true), np.max(lstm_preds), np.max(xgb_preds), np.max(ensemble_preds))
        
        axes[0,0].scatter(y_true, lstm_preds, alpha=0.6, color='blue', s=20)
        axes[0,0].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0,0].set_xlabel('True Price')
        axes[0,0].set_ylabel('Predicted Price')
        axes[0,0].set_title(f'LSTM Model\nRMSE: {metrics["lstm_rmse"]:.0f}, R²: {metrics["lstm_r2"]:.3f}')
        axes[0,0].set_xlim(0, max_val)
        axes[0,0].set_ylim(0, max_val)
        axes[0,0].legend(['Perfect Prediction', 'LSTM Predictions'])
        
        axes[0,1].scatter(y_true, xgb_preds, alpha=0.6, color='green', s=20)
        axes[0,1].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0,1].set_xlabel('True Price')
        axes[0,1].set_ylabel('Predicted Price')
        axes[0,1].set_title(f'XGBoost Model\nRMSE: {metrics["xgb_rmse"]:.0f}, R²: {metrics["xgb_r2"]:.3f}')
        axes[0,1].set_xlim(0, max_val)
        axes[0,1].set_ylim(0, max_val)
        axes[0,1].legend(['Perfect Prediction', 'XGBoost Predictions'])
        
        axes[0,2].scatter(y_true, ensemble_preds, alpha=0.6, color='purple', s=20)
        axes[0,2].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0,2].set_xlabel('True Price')
        axes[0,2].set_ylabel('Predicted Price')
        axes[0,2].set_title(f'Ensemble Model\nRMSE: {metrics["ensemble_rmse"]:.0f}, R²: {metrics["ensemble_r2"]:.3f}')
        axes[0,2].set_xlim(0, max_val)
        axes[0,2].set_ylim(0, max_val)
        axes[0,2].legend(['Perfect Prediction', 'Ensemble Predictions'])
        
        x_range = range(len(y_true))
        axes[1,0].plot(x_range, y_true, label='True Price', color='red', linewidth=2)
        axes[1,0].plot(x_range, lstm_preds, label='LSTM Prediction', color='blue', alpha=0.7)
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].set_ylabel('Price')
        axes[1,0].set_title('LSTM Time Series Prediction')
        axes[1,0].legend()
        axes[1,0].set_ylim(0, max_val)
        
        axes[1,1].plot(x_range, y_true, label='True Price', color='red', linewidth=2)
        axes[1,1].plot(x_range, xgb_preds, label='XGBoost Prediction', color='green', alpha=0.7)
        axes[1,1].set_xlabel('Sample Index')
        axes[1,1].set_ylabel('Price')
        axes[1,1].set_title('XGBoost Time Series Prediction')
        axes[1,1].legend()
        axes[1,1].set_ylim(0, max_val)
        
        axes[1,2].plot(x_range, y_true, label='True Price', color='red', linewidth=2)
        axes[1,2].plot(x_range, ensemble_preds, label='Ensemble Prediction', color='purple', alpha=0.7)
        axes[1,2].set_xlabel('Sample Index')
        axes[1,2].set_ylabel('Price')
        axes[1,2].set_title('Ensemble Time Series Prediction')
        axes[1,2].legend()
        axes[1,2].set_ylim(0, max_val)
        
        plt.tight_layout()
        plt.show()
        
        metrics_df = pd.DataFrame({
            'Model': ['LSTM', 'XGBoost', 'Ensemble'],
            'RMSE': [metrics['lstm_rmse'], metrics['xgb_rmse'], metrics['ensemble_rmse']],
            'R²': [metrics['lstm_r2'], metrics['xgb_r2'], metrics['ensemble_r2']],
            'MAE': [metrics['lstm_mae'], metrics['xgb_mae'], metrics['ensemble_mae']]
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(metrics_df['Model'], metrics_df['RMSE'], color=['blue', 'green', 'purple'])
        axes[0].set_title('Root Mean Square Error (RMSE)')
        axes[0].set_ylabel('RMSE')
        
        axes[1].bar(metrics_df['Model'], metrics_df['R²'], color=['blue', 'green', 'purple'])
        axes[1].set_title('R-squared Score')
        axes[1].set_ylabel('R²')
        
        axes[2].bar(metrics_df['Model'], metrics_df['MAE'], color=['blue', 'green', 'purple'])
        axes[2].set_title('Mean Absolute Error (MAE)')
        axes[2].set_ylabel('MAE')
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df

def main():
    predictor = HousePricePredictor()
    
    df = predictor.load_and_analyze_data('NY-House-Dataset.csv')
    
    df_processed, feature_cols, target_col = predictor.feature_engineering(df)
    
    print(f"\nTarget column: {target_col}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Processed data shape: {df_processed.shape}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(df_processed[target_col], bins=50, alpha=0.7, color='skyblue')
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 2)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns[:6]
    correlation_matrix = df_processed[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    plt.subplot(2, 3, 3)
    if len(numeric_cols) > 1:
        top_features = df_processed[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)[1:6]
        if len(top_features) > 0:
            plt.barh(range(len(top_features)), top_features.values)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.title('Top Feature Correlations')
            plt.xlabel('Correlation with Price')
    
    if 'PROPERTYSQFT' in df_processed.columns:
        plt.subplot(2, 3, 4)
        plt.scatter(df_processed['PROPERTYSQFT'], df_processed[target_col], alpha=0.5)
        plt.title('Price vs Property Size')
        plt.xlabel('Property Square Feet')
        plt.ylabel('Price')
    
    if 'BEDS' in df_processed.columns:
        plt.subplot(2, 3, 5)
        bed_counts = sorted(df_processed['BEDS'].unique())[:5]
        data_for_boxplot = [df_processed[df_processed['BEDS']==i][target_col] for i in bed_counts]
        plt.boxplot(data_for_boxplot)
        plt.title('Price vs Number of Bedrooms')
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('Price')
    
    plt.subplot(2, 3, 6)
    sample_size = min(200, len(df_processed))
    plt.plot(df_processed[target_col].values[:sample_size])
    plt.title(f'Price Trend (First {sample_size} samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()
    
    train_size = int(len(df_processed) * 0.8)
    train_data = df_processed.iloc[:train_size].copy()
    test_data = df_processed.iloc[train_size:].copy()
    
    print(f"\nTraining data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")
    
    predictor.train(train_data, feature_cols, target_col)
    
    lstm_preds, xgb_preds, ensemble_preds, y_test_scaled = predictor.predict(test_data, feature_cols, target_col)
    
    if lstm_preds is not None:
        metrics = predictor.evaluate(y_test_scaled, lstm_preds, xgb_preds, ensemble_preds)
        
        metrics_df = predictor.plot_results(metrics, lstm_preds, xgb_preds, ensemble_preds)
        
        print("\nFinal Performance Summary:")
        print("="*50)
        print(metrics_df.round(4))
        
        valid_r2 = metrics_df['R²'][~np.isinf(metrics_df['RMSE'])]
        if len(valid_r2) > 0:
            best_idx = valid_r2.idxmax()
            print(f"\nBest Model: {metrics_df.loc[best_idx, 'Model']}")
            print(f"Best R² Score: {metrics_df.loc[best_idx, 'R²']:.4f}")
            print(f"Best RMSE: {metrics_df.loc[best_idx, 'RMSE']:.0f}")
        else:
            print("\nAll models failed to produce valid results")
    
    if os.path.exists('best_house_model.pth'):
        os.remove('best_house_model.pth')

if __name__ == "__main__":
    main()