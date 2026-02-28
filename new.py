import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class Config:
    sequence_length = 30       # 使用30天数据预测
    predict_horizon = 5        # 预测未来5天
    hidden_dim = 128
    num_layers = 3
    batch_size = 256
    learning_rate = 0.001
    epochs = 5
    dropout = 0.2

class PriceRiskProcessor:
    def __init__(self, config):
        self.config = config
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.volatility_scaler = StandardScaler()
        
    def load_and_preprocess(self, csv_path):
        
        df = pd.read_csv(csv_path)
        print(f"原始数据: {df.shape}")
        
        # 1. 基础特征工程
        df = self._create_features(df)
        
        # 2. 创建风险指标
        df = self._create_risk_indicators(df)
        
        # 3. 创建预测序列
        sequences, targets, risk_targets, risk_features = self._create_prediction_sequences(df)
        
        print(f"生成序列: {len(sequences)}")
        return sequences, targets, risk_targets, risk_features
    
    def _create_features(self, df):
        """创建价格预测特征"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Name', 'date']).reset_index(drop=True)
        
        # 1. 价格特征
        df['return_1d'] = df.groupby('Name')['close'].pct_change()
        df['return_5d'] = df.groupby('Name')['close'].pct_change(5)
        df['return_10d'] = df.groupby('Name')['close'].pct_change(10)
        
        # 2. 技术指标
        for window in [5, 10, 20, 30]:
            # 移动平均
            df[f'ma_{window}'] = df.groupby('Name')['close'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
            
            # 价格通道
            df[f'high_{window}'] = df.groupby('Name')['high'].rolling(window).max().reset_index(0, drop=True)
            df[f'low_{window}'] = df.groupby('Name')['low'].rolling(window).min().reset_index(0, drop=True)
            df[f'channel_position_{window}'] = (df['close'] - df[f'low_{window}']) / (df[f'high_{window}'] - df[f'low_{window}'] + 1e-8)
        
        # 3. 动量指标
        df['momentum_5'] = df.groupby('Name')['close'].pct_change(5)
        df['momentum_10'] = df.groupby('Name')['close'].pct_change(10)
        df['momentum_20'] = df.groupby('Name')['close'].pct_change(20)
        
        # 4. 成交量指标
        df['volume_ma_10'] = df.groupby('Name')['volume'].rolling(10).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume'] / (df['volume_ma_10'] + 1)
        
        # 5. 价格-成交量关系
        df['vwap_5'] = (df.groupby('Name')[['close', 'volume']].apply(
            lambda x: (x['close'] * x['volume']).rolling(5).sum() / x['volume'].rolling(5).sum()
        ).reset_index(0, drop=True))
        df['price_vwap_ratio'] = df['close'] / df['vwap_5']
        
        return df
    
    def _create_risk_indicators(self, df):
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = (
                df.groupby('Name')['return_1d'].rolling(window).std().reset_index(0, drop=True)
            )
        
        # 2. VaR指标（Value at Risk）
        def calculate_var(returns, confidence=0.05):
            if len(returns) > 0:
                return returns.quantile(confidence)
            else:
                return 0.0
        
        for window in [10, 20]:
            df[f'var_95_{window}'] = (
                df.groupby('Name')['return_1d'].rolling(window).apply(
                    lambda x: calculate_var(x, 0.05)
                ).reset_index(0, drop=True)
            )
        
        # 3. 最大回撤
        def max_drawdown(prices):
            if len(prices) == 0:
                return 0.0
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        
        df['max_drawdown_20'] = (
            df.groupby('Name')['close'].rolling(20).apply(max_drawdown).reset_index(0, drop=True)
        )
        
        # 4. 异常检测指标
        # Z-score
        df['price_zscore_20'] = (
            df.groupby('Name')['close'].apply(
                lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
            ).reset_index(0, drop=True)
        )
        
        # 5. 未来风险标签（实际用于验证风险预测）
        df['future_volatility'] = (
            df.groupby('Name')['return_1d'].shift(-self.config.predict_horizon)
            .rolling(self.config.predict_horizon).std().reset_index(0, drop=True)
        )
        
        # 未来最大下跌
        df['future_max_decline'] = (
            df.groupby('Name')['close'].apply(
                lambda x: -(x.shift(-self.config.predict_horizon).rolling(self.config.predict_horizon).min() / x - 1)
            ).reset_index(0, drop=True)
        )
        
        # 风险事件标签（未来是否发生异常波动）
        df['risk_event'] = (
            (df['future_volatility'] > df['volatility_20'] * 2) |  # 波动率翻倍
            (df['future_max_decline'] > 0.1)  # 或下跌超过10%
        ).astype(int)
        
        return df
    
    def _create_prediction_sequences(self, df):
        price_features = ['return_1d', 'return_5d', 'return_10d'] + \
                        [f'price_ma_ratio_{w}' for w in [5, 10, 20, 30]] + \
                        [f'channel_position_{w}' for w in [5, 10, 20, 30]] + \
                        ['momentum_5', 'momentum_10', 'momentum_20'] + \
                        ['volume_ratio', 'price_vwap_ratio']
        
        risk_features = [f'volatility_{w}' for w in [5, 10, 20]] + \
                       [f'var_95_{w}' for w in [10, 20]] + \
                       ['max_drawdown_20', 'price_zscore_20']
        
        sequences = []
        price_targets = []
        risk_targets = []
        current_risk_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date').reset_index(drop=True)
            group = group.dropna().reset_index(drop=True)
            
            if len(group) >= self.config.sequence_length + self.config.predict_horizon + 5:
                
                # 标准化特征
                price_feat = group[price_features].values
                risk_feat = group[risk_features].values
                
                # 检查是否有有效数据
                if price_feat.shape[0] > 0 and risk_feat.shape[0] > 0:
                    price_feat_scaled = StandardScaler().fit_transform(price_feat)
                    risk_feat_scaled = StandardScaler().fit_transform(risk_feat)
                    
                    # 价格目标（使用价格变化率）
                    prices = group['close'].values
                    
                    for i in range(len(price_feat_scaled) - self.config.sequence_length - self.config.predict_horizon):
                        # 输入序列
                        seq_features = price_feat_scaled[i:i + self.config.sequence_length]
                        seq_risk = risk_feat_scaled[i:i + self.config.sequence_length]
                        
                        # 组合特征
                        combined_seq = np.concatenate([seq_features, seq_risk], axis=1)
                        sequences.append(combined_seq)
                        
                        # 价格预测目标（未来5天的价格变化率）
                        current_price = prices[i + self.config.sequence_length - 1]
                        future_prices = prices[i + self.config.sequence_length:i + self.config.sequence_length + self.config.predict_horizon]
                        
                        if len(future_prices) == self.config.predict_horizon:
                            # 使用价格变化率作为目标
                            price_changes = (future_prices - current_price) / current_price
                            price_targets.append(price_changes)
                            
                            # 风险预测目标
                            future_vol = group.iloc[i + self.config.sequence_length]['future_volatility']
                            future_decline = group.iloc[i + self.config.sequence_length]['future_max_decline']
                            risk_event = group.iloc[i + self.config.sequence_length]['risk_event']
                            
                            # 处理NaN值
                            if np.isnan(future_vol):
                                future_vol = 0.0
                            if np.isnan(future_decline):
                                future_decline = 0.0
                            
                            risk_target = {
                                'future_volatility': future_vol,
                                'future_max_decline': future_decline,
                                'risk_event': risk_event
                            }
                            risk_targets.append(risk_target)
                            
                            # 当前风险特征
                            current_risk = risk_feat_scaled[i + self.config.sequence_length - 1]
                            current_risk_features.append(current_risk)
        
        return (np.array(sequences, dtype=np.float32),
                np.array(price_targets, dtype=np.float32),
                risk_targets,
                np.array(current_risk_features, dtype=np.float32))

class PriceRiskDataset(Dataset):
    def __init__(self, sequences, price_targets, risk_targets):
        self.sequences = torch.FloatTensor(sequences)
        self.price_targets = torch.FloatTensor(price_targets)
        
        # 处理风险目标
        self.volatility_targets = torch.FloatTensor([r['future_volatility'] for r in risk_targets])
        self.decline_targets = torch.FloatTensor([r['future_max_decline'] for r in risk_targets])
        self.risk_event_targets = torch.LongTensor([r['risk_event'] for r in risk_targets])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], 
                self.price_targets[idx],
                self.volatility_targets[idx],
                self.decline_targets[idx],
                self.risk_event_targets[idx])

class PriceRiskModel(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.lstm = nn.LSTM(
            config.hidden_dim, 
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            config.hidden_dim * 2, 
            num_heads=8, 
            dropout=config.dropout,
            batch_first=True
        )
        self.feature_dim = config.hidden_dim * 2
        self.price_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.predict_horizon)  # 预测未来5天价格变化
        )

        self.risk_predictor = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU()
        )

        self.volatility_head = nn.Linear(config.hidden_dim // 2, 1)  # 预测波动率
        self.decline_head = nn.Linear(config.hidden_dim // 2, 1)     # 预测最大下跌
        self.risk_event_head = nn.Linear(config.hidden_dim // 2, 2)  # 预测风险事件（二分类）
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x_reshaped = x.view(-1, input_dim)
        features = self.feature_extractor(x_reshaped)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        global_features = torch.mean(attn_out, dim=1)  # [batch, feature_dim]
        price_pred = self.price_predictor(global_features)
        risk_features = self.risk_predictor(global_features)
        volatility_pred = self.volatility_head(risk_features)
        decline_pred = self.decline_head(risk_features)
        risk_event_pred = self.risk_event_head(risk_features)
        
        return {
            'price': price_pred,
            'volatility': volatility_pred.squeeze(-1),
            'decline': decline_pred.squeeze(-1),
            'risk_event': risk_event_pred,
            'attention_weights': attn_weights
        }

class PriceRiskTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using devices: {self.device}")
        
    def train(self, csv_path):
        processor = PriceRiskProcessor(self.config)
        sequences, price_targets, risk_targets, risk_features = processor.load_and_preprocess(csv_path)
        n_samples = len(sequences)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        X_train = sequences[:train_end]
        price_train = price_targets[:train_end]
        risk_train = risk_targets[:train_end]
        
        X_val = sequences[train_end:val_end]
        price_val = price_targets[train_end:val_end]
        risk_val = risk_targets[train_end:val_end]
        
        X_test = sequences[val_end:]
        price_test = price_targets[val_end:]
        risk_test = risk_targets[val_end:]

        print(f"training set: {len(X_train)}, validations set: {len(X_val)}, test set: {len(X_test)}")
        
        train_dataset = PriceRiskDataset(X_train, price_train, risk_train)
        val_dataset = PriceRiskDataset(X_val, price_val, risk_val)
        test_dataset = PriceRiskDataset(X_test, price_test, risk_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        input_dim = sequences.shape[-1]
        
        self.model = PriceRiskModel(self.config, input_dim).to(self.device)
        self._train_model(train_loader, val_loader)
        self._test_model(test_loader)
        
        return self.model, processor
    
    def _train_model(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            train_price_loss = 0
            train_risk_loss = 0
            
            for batch in train_loader:
                seq, price_target, vol_target, decline_target, risk_event_target = [x.to(self.device) for x in batch]
                
                optimizer.zero_grad()
                outputs = self.model(seq)
                price_loss = F.mse_loss(outputs['price'], price_target)
                volatility_loss = F.mse_loss(outputs['volatility'], vol_target)
                decline_loss = F.mse_loss(outputs['decline'], decline_target)
                risk_event_loss = F.cross_entropy(outputs['risk_event'], risk_event_target)
                
                total_loss = (price_loss + 0.5 * volatility_loss + 0.5 * decline_loss + 0.3 * risk_event_loss)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_price_loss += price_loss.item()
                train_risk_loss += (volatility_loss.item() + decline_loss.item() + risk_event_loss.item())
            
            self.model.eval()
            val_loss = 0
            val_price_loss = 0
            val_risk_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    seq, price_target, vol_target, decline_target, risk_event_target = [x.to(self.device) for x in batch]
                    
                    outputs = self.model(seq)
                    
                    price_loss = F.mse_loss(outputs['price'], price_target)
                    volatility_loss = F.mse_loss(outputs['volatility'], vol_target)
                    decline_loss = F.mse_loss(outputs['decline'], decline_target)
                    risk_event_loss = F.cross_entropy(outputs['risk_event'], risk_event_target)
                    
                    total_loss = (price_loss + 
                                0.5 * volatility_loss + 
                                0.5 * decline_loss + 
                                0.3 * risk_event_loss)
                    
                    val_loss += total_loss.item()
                    val_price_loss += price_loss.item()
                    val_risk_loss += (volatility_loss.item() + decline_loss.item() + risk_event_loss.item())
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_price = train_price_loss / len(train_loader)
            avg_val_price = val_price_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_price_risk_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {avg_train_loss:.6f} (Price: {avg_train_price:.6f}) | "
                      f"Val: {avg_val_loss:.6f} (Price: {avg_val_price:.6f})")
            
            if patience_counter >= patience:
                print(f"早停于第{epoch+1}轮")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_price_risk_model.pth'))
        
        # 绘制训练曲线
        self._plot_training_curves(train_losses, val_losses)
    
    def _test_model(self, test_loader):
        
        self.model.eval()
        all_price_preds = []
        all_price_targets = []
        all_vol_preds = []
        all_vol_targets = []
        all_decline_preds = []
        all_decline_targets = []
        all_risk_preds = []
        all_risk_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                seq, price_target, vol_target, decline_target, risk_event_target = [x.to(self.device) for x in batch]
                
                outputs = self.model(seq)
                
                all_price_preds.append(outputs['price'].cpu().numpy())
                all_price_targets.append(price_target.cpu().numpy())
                
                all_vol_preds.append(outputs['volatility'].cpu().numpy())
                all_vol_targets.append(vol_target.cpu().numpy())
                
                all_decline_preds.append(outputs['decline'].cpu().numpy())
                all_decline_targets.append(decline_target.cpu().numpy())
                
                risk_pred_probs = F.softmax(outputs['risk_event'], dim=1)
                all_risk_preds.append(risk_pred_probs[:, 1].cpu().numpy())  # 风险事件概率
                all_risk_targets.append(risk_event_target.cpu().numpy())
        
        # 合并结果
        price_preds = np.vstack(all_price_preds)
        price_targets = np.vstack(all_price_targets)
        vol_preds = np.concatenate(all_vol_preds)
        vol_targets = np.concatenate(all_vol_targets)
        decline_preds = np.concatenate(all_decline_preds)
        decline_targets = np.concatenate(all_decline_targets)
        risk_preds = np.concatenate(all_risk_preds)
        risk_targets = np.concatenate(all_risk_targets)
        
        # 价格预测评估
        self._evaluate_price_prediction(price_preds, price_targets)
        
        # 风险预测评估
        self._evaluate_risk_prediction(vol_preds, vol_targets, decline_preds, decline_targets, risk_preds, risk_targets)
        
        # 可视化结果
        self._visualize_results(price_preds, price_targets, vol_preds, vol_targets, risk_preds, risk_targets)
    
    def _evaluate_price_prediction(self, preds, targets):
        for day in range(self.config.predict_horizon):
            day_preds = preds[:, day]
            day_targets = targets[:, day]
            
            # 移除无效值
            valid_mask = ~(np.isnan(day_preds) | np.isnan(day_targets))
            day_preds = day_preds[valid_mask]
            day_targets = day_targets[valid_mask]
            
            if len(day_preds) > 0:
                rmse = np.sqrt(mean_squared_error(day_targets, day_preds))
                mae = mean_absolute_error(day_targets, day_preds)
                
                # 方向准确率
                pred_direction = np.sign(day_preds)
                target_direction = np.sign(day_targets)
                direction_acc = np.mean(pred_direction == target_direction)
                
                # R²分数
                if len(np.unique(day_targets)) > 1:  # 避免方差为0
                    r2 = r2_score(day_targets, day_preds)
                else:
                    r2 = 0.0
                
                print(f"  第{day+1}天: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}, 方向准确率={direction_acc:.4f}")
    
    def _evaluate_risk_prediction(self, vol_preds, vol_targets, decline_preds, decline_targets, risk_preds, risk_targets):
        print("EVALUATING HIGH RISKS")
        # 移除无效值
        vol_valid = ~(np.isnan(vol_preds) | np.isnan(vol_targets))
        decline_valid = ~(np.isnan(decline_preds) | np.isnan(decline_targets))
        risk_valid = ~(np.isnan(risk_preds) | np.isnan(risk_targets))
        
        # 波动率预测
        if vol_valid.sum() > 0:
            vol_rmse = np.sqrt(mean_squared_error(vol_targets[vol_valid], vol_preds[vol_valid]))
            vol_mae = mean_absolute_error(vol_targets[vol_valid], vol_preds[vol_valid])
            if len(np.unique(vol_targets[vol_valid])) > 1:
                vol_r2 = r2_score(vol_targets[vol_valid], vol_preds[vol_valid])
            else:
                vol_r2 = 0.0
            print(f"  波动率预测: RMSE={vol_rmse:.6f}, MAE={vol_mae:.6f}, R²={vol_r2:.4f}")
        
        # 最大下跌预测
        if decline_valid.sum() > 0:
            decline_rmse = np.sqrt(mean_squared_error(decline_targets[decline_valid], decline_preds[decline_valid]))
            decline_mae = mean_absolute_error(decline_targets[decline_valid], decline_preds[decline_valid])
            if len(np.unique(decline_targets[decline_valid])) > 1:
                decline_r2 = r2_score(decline_targets[decline_valid], decline_preds[decline_valid])
            else:
                decline_r2 = 0.0
            print(f"  最大下跌预测: RMSE={decline_rmse:.6f}, MAE={decline_mae:.6f}, R²={decline_r2:.4f}")
        
        # 风险事件预测
        if risk_valid.sum() > 0 and len(np.unique(risk_targets[risk_valid])) > 1:
            try:
                risk_auc = roc_auc_score(risk_targets[risk_valid], risk_preds[risk_valid])
                risk_binary_preds = (risk_preds[risk_valid] > 0.5).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    risk_targets[risk_valid], risk_binary_preds, average='binary'
                )
                print(f"  风险事件预测: AUC={risk_auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            except:
                print("  风险事件预测: 数据不足以计算AUC")
    
    def _plot_training_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='train_Loss', alpha=0.8)
        plt.plot(val_losses, label='validations_Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('LOSS')
        plt.title('training processing')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def _visualize_results(self, price_preds, price_targets, vol_preds, vol_targets, risk_preds, risk_targets):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        valid_mask = ~(np.isnan(price_preds[:, 0]) | np.isnan(price_targets[:, 0]))
        if valid_mask.sum() > 100:  # 如果有足够的有效数据点
            sample_size = min(1000, valid_mask.sum())
            sample_idx = np.random.choice(np.where(valid_mask)[0], sample_size, replace=False)
            axes[0, 0].scatter(price_targets[sample_idx, 0], price_preds[sample_idx, 0], alpha=0.5)
            min_val = min(price_targets[sample_idx, 0].min(), price_preds[sample_idx, 0].min())
            max_val = max(price_targets[sample_idx, 0].max(), price_preds[sample_idx, 0].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='perfect predictions')
            axes[0, 0].set_xlabel('actually prices')
            axes[0, 0].set_ylabel('predictions prices')
            axes[0, 0].set_title('price predictions(day 1)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 波动率预测
        vol_valid = ~(np.isnan(vol_preds) | np.isnan(vol_targets))
        if vol_valid.sum() > 100:
            sample_size = min(1000, vol_valid.sum())
            sample_idx = np.random.choice(np.where(vol_valid)[0], sample_size, replace=False)
            axes[0, 1].scatter(vol_targets[sample_idx], vol_preds[sample_idx], alpha=0.5)
            min_val = min(vol_targets[sample_idx].min(), vol_preds[sample_idx].min())
            max_val = max(vol_targets[sample_idx].max(), vol_preds[sample_idx].max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='perfect predictions')
            axes[0, 1].set_xlabel('actually')
            axes[0, 1].set_ylabel('predictions')
            axes[0, 1].set_title('predictions of waves')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 风险事件ROC曲线
        risk_valid = ~(np.isnan(risk_preds) | np.isnan(risk_targets))
        if risk_valid.sum() > 100 and len(np.unique(risk_targets[risk_valid])) > 1:
            try:
                fpr, tpr, _ = roc_curve(risk_targets[risk_valid], risk_preds[risk_valid])
                auc_score = roc_auc_score(risk_targets[risk_valid], risk_preds[risk_valid])
                axes[1, 0].plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
                axes[1, 0].plot([0, 1], [0, 1], 'r--', label='random prediction')
                axes[1, 0].set_xlabel('false positive')
                axes[1, 0].set_ylabel('true positive')
                axes[1, 0].set_title('ROC')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            except:
                axes[1, 0].text(0.5, 0.5, '无法绘制ROC曲线\n数据不足', ha='center', va='center')
                axes[1, 0].set_title('ROC VENTURE')
        
        # 风险概率分布
        if len(risk_targets) > 0:
            no_risk_mask = risk_targets == 0
            has_risk_mask = risk_targets == 1
            
            if no_risk_mask.sum() > 0:
                axes[1, 1].hist(risk_preds[no_risk_mask], bins=50, alpha=0.5, label='无风险', density=True)
            if has_risk_mask.sum() > 0:
                axes[1, 1].hist(risk_preds[has_risk_mask], bins=50, alpha=0.5, label='有风险', density=True)
            
            axes[1, 1].set_xlabel('风险概率')
            axes[1, 1].set_ylabel('密度')
            axes[1, 1].set_title('风险概率分布')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    config = Config()
    trainer = PriceRiskTrainer(config)
    
    csv_path = "D:\\Vs Code\\all_stocks_5yr.csv"
    
    try:
        model, processor = trainer.train(csv_path)
        torch.save({
            'model_state_dict': model.state_dict(),
            'processor': processor,
            'config': config
        }, 'price_risk_model.pth')

        print("- RMSE: 越小越好，表示价格预测误差")
        print("- R²: 越接近1越好, 表示解释方差比例")
        print("- 风险AUC: 越接近1越好, 表示风险识别能力")
        print("- 方向准确率: 涨跌方向预测正确率")
        
    except Exception as e:
        print(f"ERRORS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
"""
原始数据: (619040, 7)
生成序列: 584084
training set: 408858, validations set: 87613, test set: 87613
Epoch   5 | Train: 0.064604 (Price: 0.000776) | Val: 0.068380 (Price: 0.000655)
Epoch  10 | Train: 0.044202 (Price: 0.000767) | Val: 0.085237 (Price: 0.000651)
Epoch  15 | Train: 0.032645 (Price: 0.000759) | Val: 0.094201 (Price: 0.000652)
Epoch  20 | Train: 0.022357 (Price: 0.000751) | Val: 0.111091 (Price: 0.000651)
早停于第24轮
训练完成

  第1天: RMSE=0.016017, MAE=0.010603, R²=-0.0146, 方向准确率=0.4723
  第2天: RMSE=0.022463, MAE=0.015025, R²=-0.0024, 方向准确率=0.5111
  第3天: RMSE=0.027373, MAE=0.018562, R²=0.0005, 方向准确率=0.5305
  第4天: RMSE=0.031446, MAE=0.021571, R²=0.0013, 方向准确率=0.5338
  第5天: RMSE=0.035006, MAE=0.024178, R²=-0.0023, 方向准确率=0.5489
EVALUATING HIGH RISKS
  波动率预测: RMSE=0.009554, MAE=0.006528, R²=-0.0143
  最大下跌预测: RMSE=0.025939, MAE=0.017215, R²=-0.0020
  风险事件预测: AUC=0.7207, Precision=0.0000, Recall=0.0000, F1=0.0000
- RMSE: 越小越好，表示价格预测误差
- R²: 越接近1越好, 表示解释方差比例
- 风险AUC: 越接近1越好, 表示风险识别能力
- 方向准确率: 涨跌方向预测正确率
"""