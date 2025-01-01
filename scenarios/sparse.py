import numpy as np
from .base import BaseScenario

class SparseRegressionScenario(BaseScenario):
    """稀疏回归场景"""
    
    def __init__(self, random_seed=42):
        super().__init__("Sparse Regression", random_seed)
        self.n_features = 150
        self.n_active = 4
        self.noise_std = 0.8
        
        # 设置真实参数捏
        np.random.seed(random_seed)
        self.beta_true = np.zeros(self.n_features)
        
        # 随机选择4个位置
        active_features = np.random.choice(self.n_features, 
                                         size=self.n_active, 
                                         replace=False)
        
        # 生成2个正值和2个负值，分别在[2,3]和[-3,-2]区间
        magnitudes = np.concatenate([
            np.random.uniform(2, 3, 2),    
            np.random.uniform(-3, -2, 2)   
        ])
        
        self.beta_true[active_features] = magnitudes
    
    def generate_data(self, n_samples=100):
        """生成稀疏基准训练数据"""
        return self._generate_samples(n_samples)
        
    def generate_test_data(self, n_samples=50):
        """生成稀疏基准测试数据"""
        return self._generate_samples(n_samples)
        
    def _generate_samples(self, n_samples):
        """内部方法：生成稀疏基准样本"""
        X = np.random.randn(n_samples, self.n_features)
        y = X @ self.beta_true + np.random.normal(0, self.noise_std, n_samples)
        return X, y
    
    def get_true_params(self):
        """返回真实参数"""
        return self.beta_true
