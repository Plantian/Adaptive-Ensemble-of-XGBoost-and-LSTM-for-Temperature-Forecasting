<div align="center">

# ⛅ Adaptive Ensemble of XGBoost and LSTM for Temperature Forecasting

**基于 XGBoost 与 LSTM 自适应集成的高精度气象温度预测模型**

[English](#) • [简体中文](#) • [Dataset](https://github.com/florian-huber/weather_prediction_dataset) 

</div>

---

## 📖 项目简介 (Overview)

本项目开源了论文 **《Adaptive Ensemble of XGBoost and LSTM for Temperature Forecasting》** 中的核心算法。我们提出了一种创新的**自适应集成学习方法**，巧妙地结合了：
- 🧠 **LSTM (长短期记忆网络):** 擅长捕捉长期的时间序列依赖与季节性模式。
- 🌳 **XGBoost (极端梯度提升树):** 擅长处理复杂的特征交互与非线性关系。

传统集成模型通常使用固定的权重，而本项目的**最大亮点**在于引入了一个 **动态权重学习模块 (Adaptive Weight Learning Module)**，它能根据历史统计特征和当前气象条件，实时为单个模型分配最优权重。在欧洲五大城市的 10 年气象数据集上，该模型的预测决定系数 ($R^2$) 始终**稳定超过 0.99**。

## ✨ 核心特性 (Key Features)

- **自适应权重分配机制**：根据气象条件动态调整 XGBoost 和 LSTM 的输出权重，告别手动调参。
- **卓越的预测精度**：在温带大陆性/海洋性气候下表现惊艳，RMSE 极低，高度贴合真实温度曲线。
- **双通道架构**：兼顾长期时序依赖 (Temporal Trends) 与并发气象特征交互 (Feature Interactions)。
- **高鲁棒性**：在长达 10 年（3650天）的多维气象数据集上经过充分验证。

---

## 🚀 快速开始 (Getting Started)

### 克隆仓库
```bash
git clone https://github.com/YourUsername/Adaptive-Ensemble-Weather-Forecast.git
cd Adaptive-Ensemble-Weather-Forecast          
```
---

## 📖 引用

```bib
@inproceedings{Ye2026,
  title={Adaptive Ensemble of XGBoost and LSTM for Temperature Forecasting},
  author={Mingcheng Ye},
  year={2026},
  booktitle={Proceedings of the 2025 International Conference on Hybrid Commerce, Human Capital, and Economic Dynamics (ICHCH 2025)},
  pages={237-246},
  issn={2352-5428},
  isbn={978-2-38476-585-0},
  url={https://doi.org/10.2991/978-2-38476-585-0_28},
  doi={10.2991/978-2-38476-585-0_28},
  publisher={Atlantis Press}
}
```
