import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualAttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        self.norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        residual = x
        out = self.attention(x)
        out = self.norm(out + residual)
        return out

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=3, dropout=0.3):
        super(EnhancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size//2, 3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size//2, 6, batch_first=True)
        self.lstm3 = nn.LSTM(input_size, hidden_size//2, 9, batch_first=True)
        
        self.attention_blocks = nn.ModuleList([
            ResidualAttentionBlock(hidden_size//2, hidden_size//4)
            for _ in range(3)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3//2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(x)
        out3, _ = self.lstm3(x)
        
        out1 = self.attention_blocks[0](out1[:, -1, :])
        out2 = self.attention_blocks[1](out2[:, -1, :])
        out3 = self.attention_blocks[2](out3[:, -1, :])
        
        combined = torch.cat([out1, out2, out3], dim=1)
        return self.fusion(combined)

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HybridStockPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = 40
        self.lstm = EnhancedLSTM(input_size=13).to(self.device)
        self.direction_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.scalers = {}
        
    def calculate_technical_indicators(self, df):
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()

        for window in [7, 21]:
            df[f'ma{window}'] = df['close'].rolling(window=window).mean()
            df[f'std{window}'] = df['close'].rolling(window=window).std()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_upper'] = df['bb_middle'] + 2*df['close'].rolling(window=20).std()
        df['bb_lower'] = df['bb_middle'] - 2*df['close'].rolling(window=20).std()
        
        df['momentum'] = df['close'].diff(periods=5)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df

    def prepare_data(self, df):
        df = self.calculate_technical_indicators(df)
        
        features = ['open', 'high', 'low', 'close', 'volume',
                   'ma7', 'ma21', 'rsi', 'macd', 'signal',
                   'momentum', 'volume_ratio', 'returns']
        
        scaled_data = {}
        for feature in features:
            if feature not in self.scalers:
                self.scalers[feature] = MinMaxScaler()
            scaled_data[feature] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            ).reshape(-1)
        
        df_scaled = pd.DataFrame(scaled_data)
        
        X, y = [], []
        X_direction = []
        direction = []
        
        for i in range(len(df_scaled) - self.seq_length):
            X.append(df_scaled.iloc[i:i+self.seq_length].values)
            y.append(df_scaled['close'].iloc[i+self.seq_length])
            
            current_features = df_scaled.iloc[i+self.seq_length-1]
            X_direction.append(current_features)
            direction.append(1 if df['close'].iloc[i+self.seq_length] > df['close'].iloc[i+self.seq_length-1] else 0)
        
        return (np.array(X), np.array(y), 
                np.array(X_direction), np.array(direction))

    def train(self, train_data, epochs=100):
        X_train, y_train, X_dir_train, dir_train = self.prepare_data(train_data)
        
        train_dataset = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.lstm.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        
        best_loss = float('inf')
        for epoch in range(epochs):
            self.lstm.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                pred = self.lstm(X_batch)
                
                loss = nn.MSELoss()(pred.squeeze(), y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.lstm.state_dict(), 'best_lstm_model.pth')
        
        self.direction_model.fit(X_dir_train, dir_train)

    def predict(self, test_data):
        X_test, y_test, X_dir_test, _ = self.prepare_data(test_data)
        test_dataset = StockDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        self.lstm.eval()
        price_predictions = []
        
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                pred = self.lstm(X_batch)
                price_predictions.extend(pred.cpu().numpy())
        
        price_predictions = np.array(price_predictions).reshape(-1)
        price_predictions = self.scalers['close'].inverse_transform(
            price_predictions.reshape(-1, 1)
        ).reshape(-1)
        
        direction_pred = self.direction_model.predict(X_dir_test)
        
        return price_predictions, direction_pred

    def evaluate(self, test_data, predictions, direction_pred):
        actual = test_data['close'].values[self.seq_length:]
        
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        actual_direction = np.diff(actual) > 0
        actual_direction = np.append(actual_direction, actual_direction[-1])
        direction_accuracy = np.mean(direction_pred == actual_direction)
        
        print(f"价格RMSE: {rmse:.6f}")
        print(f"价格MAE: {mae:.6f}")
        print(f"价格R²: {r2:.4f}")
        print(f"方向准确率: {direction_accuracy:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'mape': mape
        }

    def plot_predictions(self, test_data, predictions, symbol):
        plt.figure(figsize=(15, 7))
        plt.plot(test_data.index[self.seq_length:], test_data['close'][self.seq_length:], label='actually', color='blue', alpha=0.7)
        plt.plot(test_data.index[self.seq_length:], predictions, label='predictions value', color='red', alpha=0.7)
        
        plt.title(f'{symbol} price predictions', fontsize=14)
        plt.xlabel('date', fontsize=12)
        plt.ylabel('price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        error = np.abs(test_data['close'][self.seq_length:].values - predictions)
        plt.fill_between(test_data.index[self.seq_length:], predictions - error, predictions + error, alpha=0.2, color='gray', label='预测误差范围')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    df = pd.read_csv('all_stocks_5yr.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    results = {}
    for symbol in stock_symbols:
        stock_data = df[df['Name'] == symbol].copy()
        stock_data.set_index('date', inplace=True)

        train_size = int(len(stock_data) * 0.8)
        train_data = stock_data[:train_size]
        test_data = stock_data[train_size:]

        predictor = HybridStockPredictor()
        predictor.train(train_data)
        predictions, direction_pred = predictor.predict(test_data)
           
        metrics = predictor.evaluate(test_data, predictions, direction_pred)
        predictor.plot_predictions(test_data, predictions, symbol)
        
        results[symbol] = metrics

    for symbol, metrics in results.items():
        print(f"\n{symbol}:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"directions accuracys: {metrics['direction_accuracy']:.4f}")

if __name__ == "__main__":
    main()