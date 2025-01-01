from abc import ABC, abstractmethod
import numpy as np

class BaseScenario(ABC):
    """所有场景的基类"""
    
    def __init__(self, name, random_seed=42):
        """
        参数:
        name: str, 场景名称
        random_seed: int, 随机种子，用于复现结果
        """
        self.name = name
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.true_params = None
        
    @abstractmethod
    def generate_data(self, n_samples=1000):
        """生成数据
        
        参数:
        n_samples: int, 样本数量
        
        返回:
        dict: 包含生成数据的字典
        """
        pass
        
    @abstractmethod
    def get_true_params(self):
        """获取真实参数
        
        返回:
        dict: 包含真实参数的字典
        """
        return self.true_params
