from torch import nn
import torch

class OptimizedAdaptiveEnsemble:
    """简化的自适应集成模型"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length = 14
        
        # 核心组件
        self.lstm_model = None
        self.xgb_model = ScenarioAwareXGBoost()
        self.weight_predictor = None
    
    def predict(self, X_seq, X_current):
        """自适应预测（移除其他模式）"""
        # 获取基础预测
        lstm_preds = self._get_lstm_predictions(X_seq, X_current)
        xgb_preds = self.xgb_model.predict(X_seq, X_current)
        
        # 自适应权重预测
        weight_features = self._prepare_weight_features(X_seq, X_current)
        weight_features_tensor = torch.FloatTensor(weight_features).to(self.device)
        
        with torch.no_grad():
            weights = self.weight_predictor(weight_features_tensor).cpu().numpy()
        
        # 自适应集成
        ensemble_preds = weights[:, 0] * lstm_preds + weights[:, 1] * xgb_preds
        
        return lstm_preds, xgb_preds, ensemble_preds
    
    def _prepare_weight_features(self, X_seq, X_current):
        """准备权重预测器的特征"""
        weight_features = []
        for i in range(len(X_seq)):
            # 序列统计特征
            seq_stats = np.concatenate([
                np.mean(X_seq[i], axis=0)[:5],
                np.std(X_seq[i], axis=0)[:5]
            ])
            # 当前条件特征
            current_features = X_current[i]
            # 组合特征
            combined_features = np.concatenate([current_features, seq_stats])
            weight_features.append(combined_features)
        
        return np.array(weight_features)