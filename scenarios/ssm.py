import numpy as np
# import matplotlib.pyplot as plt
from .base import BaseScenario

class SSMScenario(BaseScenario):
    """非线性状态空间模型场景"""
    
    def __init__(self, random_seed=42):
        super().__init__("SSM", random_seed)
        #真实参数捏
        self.true_params = {
            'Q': 10.0,  
            'R': 1.0   
        }
        
    def state_transition(self, x_prev, t):
        """状态转移函数"""
        return (0.5 * x_prev + 
                25 * x_prev / (1 + x_prev ** 2) + 
                8 * np.cos(1.2 * t))
                
    def observation(self, x):
        """观测函数"""
        return x ** 2 / 20
        
    def generate_data(self, n_samples=100):
        """生成状态空间模型训练数据"""
        return self._generate_samples(n_samples)
        
    def generate_test_data(self, n_samples=50):
        """生成状态空间模型测试数据"""
        return self._generate_samples(n_samples)
        
    def _generate_samples(self, n_samples):
        """内部方法：生成状态空间模型样本"""
        x = np.zeros(n_samples)  # 状态序列
        y = np.zeros(n_samples)  # 观测序列
        
        # 生成初始状态
        x[0] = np.random.normal(0, np.sqrt(self.true_params['Q']))
        
        # 生成状态序列和观测序列
        for t in range(1, n_samples):
            # 状态转移
            x[t] = (self.state_transition(x[t-1], t) + 
                   np.random.normal(0, np.sqrt(self.true_params['Q'])))
            
        # 生成观测
        y = self.observation(x) + np.random.normal(
            0, np.sqrt(self.true_params['R']), n_samples)
            
        # 为INLA创建合适的时间点，但不影响模型动态
        times = np.linspace(0, 100, n_samples)  # 使用更大的时间间隔
            
        return {
            'states': x,
            'observations': y,
            'times': times  # 只用于INLA的拟合
        }
    def get_true_params(self):
        """获取真实参数"""
        return self.true_params
