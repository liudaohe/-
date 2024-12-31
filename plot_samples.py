"""
用于可视化基准数据和采样结果的模块
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

try:
    from scenarios.gmm import GMMScenario
    from scenarios.ssm import SSMScenario
    from scenarios.sparse import SparseRegressionScenario
    from models.numerical.inla import INLAOptimizer
    from models.optimization.gflownet import GFlowNetOptimizer
    from models.optimization.vi import VISampler
except ImportError as e:
    print(f"导入错误: {e}")
    raise

# 设置全局绘图样式
plt.style.use('classic')  # 使用经典样式
mpl.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']  # 设置中文字体
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.axisbelow'] = True  # 网格线在底层

# 创建visualization目录（如果不存在）
if not os.path.exists('visualization'):
    os.makedirs('visualization')

def plot_benchmark_samples():
    """绘制三个基准场景的真实样本分布"""
    try:
        plt.figure(figsize=(20, 6))
        
        # GMM基准
        plt.subplot(131)
        gmm = GMMScenario(random_seed=42)
        data = gmm.generate_data(n_samples=10000)
        X = data['observations']
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=3, c='#1f77b4', label='样本点')
        plt.title('GMM基准样本', pad=15)
        plt.xlabel('X1', labelpad=10)
        plt.ylabel('X2', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # SSM基准
        plt.subplot(132)
        ssm = SSMScenario(random_seed=42)
        data = ssm.generate_data(n_samples=10000)
        t = np.arange(len(data['observations']))
        plt.plot(t, data['states'], '-', color='#2ecc71', alpha=0.8, linewidth=2, label='隐状态')
        plt.scatter(t, data['observations'], c='#e74c3c', alpha=0.4, s=2, label='观测')
        plt.title('SSM基准样本', pad=15)
        plt.xlabel('时间', labelpad=10)
        plt.ylabel('值', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Sparse基准
        plt.subplot(133)
        sparse = SparseRegressionScenario(random_seed=42)
        X, y = sparse.generate_data(n_samples=10000)
        plt.scatter(X[:, 0], y, alpha=0.6, s=3, c='#9b59b6', label='样本点')
        plt.title('Sparse基准样本', pad=15)
        plt.xlabel('X', labelpad=10)
        plt.ylabel('y', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout(pad=3.0)
        plt.savefig('visualization/benchmark_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("基准样本可视化完成")
    except Exception as e:
        print(f"基准样本可视化错误: {e}")
        raise

def plot_gmm_comparison():
    """绘制GMM基准与INLA采样的对比"""
    try:
        plt.figure(figsize=(16, 7))
        
        # 生成基准数据
        gmm = GMMScenario(random_seed=42)
        data = gmm.generate_data(n_samples=10000)
        X_true = data['observations']
        
        # INLA采样
        inla = INLAOptimizer(random_seed=42)
        samples = inla.fit_gmm(data, n_iterations=10000)
        
        # 从后验分布生成样本
        X_samples = []
        for i in range(10000):
            k = np.random.choice(2, p=samples['pi_samples'][i % len(samples['pi_samples'])])
            if k == 0:
                mu = samples['mu1_samples'][i % len(samples['mu1_samples'])]
                sigma = samples['sigma1_samples'][i % len(samples['sigma1_samples'])]
            else:
                mu = samples['mu2_samples'][i % len(samples['mu2_samples'])]
                sigma = samples['sigma2_samples'][i % len(samples['sigma2_samples'])]
            X_samples.append(np.random.multivariate_normal(mu, sigma))
        X_samples = np.array(X_samples)
        
        # 绘制对比图
        plt.subplot(121)
        plt.scatter(X_true[:, 0], X_true[:, 1], alpha=0.6, s=3, c='#3498db', label='真实样本')
        plt.title('GMM真实样本', pad=15)
        plt.xlabel('X1', labelpad=10)
        plt.ylabel('X2', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(122)
        plt.scatter(X_samples[:, 0], X_samples[:, 1], alpha=0.6, s=3, c='#e74c3c', label='INLA样本')
        plt.title('INLA生成样本', pad=15)
        plt.xlabel('X1', labelpad=10)
        plt.ylabel('X2', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout(pad=3.0)
        plt.savefig('visualization/gmm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("GMM对比可视化完成")
    except Exception as e:
        print(f"GMM对比可视化错误: {e}")
        raise

def plot_ssm_qr_comparison():
    """绘制SSM中Q和R参数的采样结果对比"""
    try:
        plt.figure(figsize=(16, 7))
        
        # 生成基准数据
        ssm = SSMScenario(random_seed=42)
        data = ssm.generate_data(n_samples=10000)
        true_Q = ssm.true_params['Q']  # 从SSMScenario实例获取真实的Q值
        true_R = ssm.true_params['R']  # 从SSMScenario实例获取真实的R值
        
        # GFlowNet采样
        gfn = GFlowNetOptimizer(random_seed=42)
        samples = gfn.fit_ssm(data, n_iterations=10000)
        
        # 绘制Q的采样分布
        plt.subplot(121)
        plt.hist(samples['Q_samples'], bins=50, alpha=0.6, color='#3498db', label='GFlowNet采样')
        plt.axvline(x=true_Q, color='red', linestyle='--', linewidth=2, label=f'真实值 (Q={true_Q})')
        plt.title('Q参数采样分布', pad=15)
        plt.xlabel('Q值', labelpad=10)
        plt.ylabel('频数', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 绘制R的采样分布
        plt.subplot(122)
        plt.hist(samples['R_samples'], bins=50, alpha=0.6, color='#3498db', label='GFlowNet采样')
        plt.axvline(x=true_R, color='red', linestyle='--', linewidth=2, label=f'真实值 (R={true_R})')
        plt.title('R参数采样分布', pad=15)
        plt.xlabel('R值', labelpad=10)
        plt.ylabel('频数', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout(pad=3.0)
        plt.savefig('visualization/ssm_qr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("SSM Q/R参数对比可视化完成")
    except Exception as e:
        print(f"SSM Q/R参数对比可视化错误: {e}")
        raise

def plot_sparse_comparison():
    """绘制Sparse基准与VI采样的对比"""
    try:
        plt.figure(figsize=(16, 7))
        
        # 生成基准数据
        sparse = SparseRegressionScenario(random_seed=42)
        X, y = sparse.generate_data(n_samples=10000)
        
        # VI采样
        vi = VISampler(random_seed=42)
        samples = vi.fit_sparse((X, y), n_iterations=10000)
        
        # 使用beta_samples生成采样点
        beta_samples = samples['beta_samples']
        y_samples = np.array([X @ beta for beta in beta_samples])  # shape: [n_samples, n_points]
        
        # 绘制对比图
        plt.subplot(121)
        plt.scatter(X[:, 0], y, alpha=0.6, s=3, c='#1f77b4', label='真实样本')
        plt.title('Sparse真实样本', pad=15)
        plt.xlabel('X', labelpad=10)
        plt.ylabel('y', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(122)
        # 随机选择一组采样结果展示
        sample_idx = np.random.randint(len(beta_samples))
        plt.scatter(X[:, 0], y_samples[sample_idx], alpha=0.6, s=3, c='#e74c3c', label='VI采样')
        plt.title('VI采样结果', pad=15)
        plt.xlabel('X', labelpad=10)
        plt.ylabel('y', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout(pad=3.0)
        plt.savefig('visualization/sparse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Sparse对比可视化完成")
    except Exception as e:
        print(f"Sparse对比可视化错误: {e}")
        raise

def main():
    """主函数"""
    try:
        print("开始生成可视化图像...")
        
        # 创建visualization目录
        if not os.path.exists('visualization'):
            os.makedirs('visualization')
        
        # 绘制所有图像
        plot_benchmark_samples()
        plot_gmm_comparison()
        plot_ssm_qr_comparison()
        plot_sparse_comparison()
        
        print("所有可视化图像生成完成！")
    except Exception as e:
        print(f"主程序错误: {e}")
        raise

if __name__ == "__main__":
    main() 