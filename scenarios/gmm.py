import numpy as np
from scipy.stats import multivariate_normal
from .base import BaseScenario

class GMMScenario(BaseScenario):
    """高斯混合模型场景"""
    
    def __init__(self, random_seed=42):
        super().__init__("GMM", random_seed)
        #真实参数捏
        self.true_params = {
            'mu1': np.array([-2, -2]),
            'mu2': np.array([2, 2]),
            'sigma1': np.array([[1.0, 0.5], [0.5, 1.0]]),
            'sigma2': np.array([[1.5, -0.5], [-0.5, 1.5]]),
            'weights': np.array([0.3, 0.7])
        }
        
    def generate_data(self, n_samples=1000):
        """生成GMM训练数据"""
        return self._generate_samples(n_samples)
        
    def generate_test_data(self, n_samples=500):
        """生成GMM测试数据"""
        return self._generate_samples(n_samples)
        
    def _generate_samples(self, n_samples):
        """内部方法：生成GMM样本"""
        #根据混合权重确定每个分量的样本数
        n1 = np.random.binomial(n_samples, self.true_params['weights'][0])
        n2 = n_samples - n1
        
        #从两个高斯分量生成样本
        samples1 = np.random.multivariate_normal(
            self.true_params['mu1'], 
            self.true_params['sigma1'], 
            n1
        )
        samples2 = np.random.multivariate_normal(
            self.true_params['mu2'], 
            self.true_params['sigma2'], 
            n2
        )
        #合并并打乱样本
        samples = np.vstack([samples1, samples2])
        np.random.shuffle(samples)
        
        return {'observations': samples}
        
    def get_true_params(self):
        """获取真实参数"""
        return self.true_params
