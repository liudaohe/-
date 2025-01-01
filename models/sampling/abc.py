import numpy as np
from ..base import BaseAlgorithm
from ..initialization import GMMPrior, SSMPrior, SparsePrior

class ABCSampler(BaseAlgorithm):
    """ABC (Approximate Bayesian Computation) 采样算法"""
    
    def __init__(self, random_seed=42):
        super().__init__("ABC", random_seed)
        self.n_particles = 100
        self.min_acceptance_rate = 0.01  # 最小接受率
        self.max_tries_multiplier = 10   # 最大尝试次数倍数
        np.random.seed(random_seed)

    
    def _compute_gmm_statistics(self, X):
        """计算GMM的充分统计量
        
        返回：均值、协方差、中位数、四分位数等统计量
        """
        stats = {}
        # 基本统计量
        stats['mean'] = np.mean(X, axis=0)
        stats['median'] = np.median(X, axis=0)
        stats['cov'] = np.cov(X.T)
        
        # Robust统计量
        stats['q1'] = np.percentile(X, 25, axis=0)
        stats['q3'] = np.percentile(X, 75, axis=0)
        stats['iqr'] = stats['q3'] - stats['q1']
        
        # 高阶统计量
        centered_X = X - stats['mean']
        stats['skewness'] = np.mean(centered_X**3, axis=0) / (np.std(X, axis=0)**3)
        stats['kurtosis'] = np.mean(centered_X**4, axis=0) / (np.std(X, axis=0)**4)
        
        # 相关性统计量
        if X.shape[1] > 1:
            stats['correlation'] = np.corrcoef(X.T)
        
        return stats
    
    def _compute_distance(self, stats1, stats2):
        """计算两组统计量之间的距离"""
        # 基本统计量的距离
        d_mean = np.linalg.norm(stats1['mean'] - stats2['mean']) / (np.linalg.norm(stats1['mean']) + 1e-8)
        d_median = np.linalg.norm(stats1['median'] - stats2['median']) / (np.linalg.norm(stats1['median']) + 1e-8)
        d_cov = np.linalg.norm(stats1['cov'] - stats2['cov'], ord='fro') / (np.linalg.norm(stats1['cov'], ord='fro') + 1e-8)
        
        # Robust统计量的距离
        d_iqr = np.linalg.norm(stats1['iqr'] - stats2['iqr']) / (np.linalg.norm(stats1['iqr']) + 1e-8)
        
        # 高阶统计量的距离
        d_skew = np.linalg.norm(stats1['skewness'] - stats2['skewness']) / (np.linalg.norm(stats1['skewness']) + 1e-8)
        d_kurt = np.linalg.norm(stats1['kurtosis'] - stats2['kurtosis']) / (np.linalg.norm(stats1['kurtosis']) + 1e-8)
        
        # 相关性的距离（如果存在）
        d_corr = 0
        if 'correlation' in stats1 and 'correlation' in stats2:
            d_corr = np.linalg.norm(stats1['correlation'] - stats2['correlation'], ord='fro') / (np.linalg.norm(stats1['correlation'], ord='fro') + 1e-8)
        
        # 加权求和，更重视robust统计量
        weights = np.array([0.2, 0.2, 0.15, 0.2, 0.1, 0.1, 0.05])
        return np.dot(weights, [d_mean, d_median, d_cov, d_iqr, d_skew, d_kurt, d_corr])
    
    def _adaptive_epsilon(self, X):
        """自适应确定阈值epsilon"""
        # 使用IQR (四分位距)
        q1 = np.percentile(X, 25)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        
        # 使用中位数绝对偏差(MAD)
        mad = np.median(np.abs(X - np.median(X)))
        
        # 结合IQR和MAD
        epsilon = 2.0 * (0.7 * iqr + 0.3 * mad)
        
        return epsilon
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的ABC采样实现"""
        def _fit():
            X = data['observations']
            n_samples, dim = X.shape
            
            # 使用先验对象
            prior = GMMPrior(dim)
            
            # 计算观测数据的统计量
            obs_stats = self._compute_gmm_statistics(X)
            
            # 存储采样结果
            mu1_samples = np.zeros((n_iterations, dim))
            mu2_samples = np.zeros((n_iterations, dim))
            sigma1_samples = np.zeros((n_iterations, dim, dim))
            sigma2_samples = np.zeros((n_iterations, dim, dim))
            pi_samples = np.zeros((n_iterations, 2))
            
            # 设置阈值
            epsilon = self._adaptive_epsilon(X)
            
            # ABC采样
            accepted = 0
            iteration = 0
            while accepted < n_iterations:
                # 从先验分布采样参数
                mu1 = prior.mu_prior.rvs()
                mu2 = prior.mu_prior.rvs()
                sigma1 = np.linalg.inv(prior.sigma_prior.rvs())  # Wishart生成精度矩阵的逆
                sigma2 = np.linalg.inv(prior.sigma_prior.rvs())
                pi = prior.pi_prior.rvs().ravel()  
                
                # 生成模拟数据
                z = np.random.binomial(1, pi[1], size=n_samples)
                X_sim = np.zeros((n_samples, dim))
                X_sim[z==0] = np.random.multivariate_normal(mu1, sigma1, size=np.sum(z==0))
                X_sim[z==1] = np.random.multivariate_normal(mu2, sigma2, size=np.sum(z==1))
                
                # 计算模拟数据的统计量
                sim_stats = self._compute_gmm_statistics(X_sim)
                
                # 计算距离并决定是否接受
                distance = self._compute_distance(obs_stats, sim_stats)
                
                if distance < epsilon:
                    mu1_samples[accepted] = mu1
                    mu2_samples[accepted] = mu2
                    sigma1_samples[accepted] = sigma1
                    sigma2_samples[accepted] = sigma2
                    pi_samples[accepted] = pi
                    accepted += 1
                    
                    if accepted % 100 == 0:
                        print(f"已接受 {accepted}/{n_iterations} 个样本")
                
                iteration += 1
                if iteration > n_iterations * 100:  
                    print("接受率过低，提前停止")
                    break
            
            # 计算接受率
            acceptance_rate = accepted / iteration
            
            return {
                'mu1_samples': mu1_samples[n_burnin:accepted],
                'mu2_samples': mu2_samples[n_burnin:accepted],
                'sigma1_samples': sigma1_samples[n_burnin:accepted],
                'sigma2_samples': sigma2_samples[n_burnin:accepted],
                'pi_samples': pi_samples[n_burnin:accepted],
                'n_iterations': accepted,
                'n_burnin': n_burnin,
                'acceptance_rate': acceptance_rate
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def _compute_ssm_statistics(self, y):
        """计算状态空间模型的统计量"""
        stats = {}
        stats['mean'] = np.mean(y)
        stats['std'] = np.std(y)
        stats['acf'] = self._compute_acf(y, nlags=5)
        stats['diff_std'] = np.std(np.diff(y))
        return stats
    
    def _compute_acf(self, y, nlags):
        """计算自相关系数"""
        acf = np.zeros(nlags)
        y_centered = y - np.mean(y)
        var = np.var(y)
        for i in range(nlags):
            acf[i] = np.sum(y_centered[i:] * y_centered[:-i if i>0 else None]) / (len(y) * var)
        return acf
    
    def _compute_ssm_distance(self, stats1, stats2):
        """计算SSM统计量之间的距离，添加标准化"""
        d1 = abs(stats1['mean'] - stats2['mean']) / (abs(stats1['mean']) + 1e-10)
        d2 = abs(stats1['std'] - stats2['std']) / (stats1['std'] + 1e-10)
        d3 = np.linalg.norm(stats1['acf'] - stats2['acf']) / (np.linalg.norm(stats1['acf']) + 1e-10)
        d4 = abs(stats1['diff_std'] - stats2['diff_std']) / (stats1['diff_std'] + 1e-10)
        
        # 加权求和，更重视均值和标准差
        return 0.3*d1 + 0.3*d2 + 0.2*d3 + 0.2*d4
    
    def fit_ssm(self, data, n_iterations=1000, n_burnin=100):
        """状态空间模型的ABC采样实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            # 使用先验对象
            prior = SSMPrior()
            
            # 计算观测数据的统计量
            obs_stats = self._compute_ssm_statistics(y)
            
            # 存储采样结果
            x_samples = np.zeros((n_iterations, T))
            Q_samples = np.zeros(n_iterations)
            R_samples = np.zeros(n_iterations)
            
            # 设置阈值
            epsilon = self._adaptive_epsilon(y)
            
            # ABC采样
            accepted = 0
            iteration = 0
            while accepted < n_iterations:
                # 从先验分布采样参数
                Q = prior.Q_prior.rvs()
                R = prior.R_prior.rvs()
                x0 = prior.x0_prior.rvs()
                
                # 生成模拟数据
                x_sim = np.zeros(T)
                y_sim = np.zeros(T)
                
                # 初始状态
                x_sim[0] = x0
                y_sim[0] = x_sim[0]**2/20 + np.random.normal(0, np.sqrt(R))
                
                # 生成序列
                for t in range(1, T):
                    x_sim[t] = 0.5*x_sim[t-1] + 25*x_sim[t-1]/(1+x_sim[t-1]**2) + 8*np.cos(1.2*t) + np.random.normal(0, np.sqrt(Q))
                    y_sim[t] = x_sim[t]**2/20 + np.random.normal(0, np.sqrt(R))
                
                # 计算模拟数据的统计量
                sim_stats = self._compute_ssm_statistics(y_sim)
                
                # 计算距离并决定是否接受
                distance = self._compute_ssm_distance(obs_stats, sim_stats)
                
                if distance < epsilon:
                    x_samples[accepted] = x_sim
                    Q_samples[accepted] = Q
                    R_samples[accepted] = R
                    accepted += 1
                    
                    if accepted % 100 == 0:
                        print(f"已接受 {accepted}/{n_iterations} 个样本")
                
                iteration += 1
                if iteration > n_iterations * 100:  
                    print("接受率过低，提前停止")
                    break
            
            # 计算接受率
            acceptance_rate = accepted / iteration
            
            return {
                'state_samples': x_samples[n_burnin:accepted],
                'Q_samples': Q_samples[n_burnin:accepted],
                'R_samples': R_samples[n_burnin:accepted],
                'n_iterations': accepted,
                'n_burnin': n_burnin,
                'acceptance_rate': acceptance_rate
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def _compute_sparse_statistics(self, X, y):
        """计算稀疏回归的统计量
        优化计算效率，减少不必要的统计量
        """
        stats = {}
        # 基本统计量
        stats['y_mean'] = np.mean(y)
        stats['y_std'] = np.std(y)
        
        # 使用岭回归代替普通最小二乘
        n_samples, n_features = X.shape
        lambda_reg = 0.1  # 正则化参数
        
        # 计算beta的稳定解
        XtX = X.T @ X
        Xty = X.T @ y
        beta_ridge = np.linalg.solve(XtX + lambda_reg * np.eye(n_features), Xty)
        
        # 计算预测值
        y_pred = X @ beta_ridge
        
        # 计算R2
        y_centered = y - stats['y_mean']
        stats['r2'] = 1 - np.sum((y - y_pred)**2) / np.sum(y_centered**2)
        
        # 计算稀疏度（使用阈值筛选）
        stats['sparsity'] = np.mean(np.abs(beta_ridge) < 1e-3)
        
        return stats
    
    def _compute_sparse_distance(self, stats1, stats2):
        """计算稀疏回归统计量之间的距离
        简化距离计算，更注重稀疏性和拟合优度
        """
        # 1. R2差异（已标准化）
        d1 = abs(stats1['r2'] - stats2['r2'])
        
        # 2. 稀疏度差异（已标准化）
        d2 = abs(stats1['sparsity'] - stats2['sparsity'])
        
        # 3. y的分布差异（标准化）
        d3 = abs(stats1['y_mean'] - stats2['y_mean']) / (abs(stats1['y_mean']) + 1e-10)
        d4 = abs(stats1['y_std'] - stats2['y_std']) / (stats1['y_std'] + 1e-10)
        
        # 主要关注R2和稀疏度
        return 0.4*d1 + 0.4*d2 + 0.1*d3 + 0.1*d4
    
    def fit_sparse(self, data, n_iterations=1000, n_burnin=100):
        """稀疏回归的ABC采样实现"""
        def _fit():
            X, y = data  # 解包元组
            n_samples, n_features = X.shape
            
            # 使用先验对象
            prior = SparsePrior(n_features)
            
            # 计算观测数据的统计量
            obs_stats = self._compute_sparse_statistics(X, y)
            
            # 存储采样结果
            beta_samples = np.zeros((n_iterations, n_features))
            sigma2_samples = np.zeros(n_iterations)
            
            # 设置阈值
            epsilon = self._adaptive_epsilon(y)
            
            # ABC采样
            accepted = 0
            iteration = 0
            while accepted < n_iterations:
                # 从先验分布采样参数
                beta = np.zeros(n_features)  # 从零开始
                for j in range(n_features):
                    # 使用Laplace先验
                    beta[j] = np.random.laplace(0, prior.lambda_prior)
                sigma2 = prior.sigma2_prior.rvs()
                
                # 生成模拟数据
                y_sim = X @ beta + np.random.normal(0, np.sqrt(sigma2), n_samples)
                
                # 计算模拟数据的统计量
                sim_stats = self._compute_sparse_statistics(X, y_sim)
                
                # 计算距离并决定是否接受
                distance = self._compute_sparse_distance(obs_stats, sim_stats)
                
                if distance < epsilon:
                    beta_samples[accepted] = beta
                    sigma2_samples[accepted] = sigma2
                    accepted += 1
                    
                    if accepted % 100 == 0:
                        print(f"已接受 {accepted}/{n_iterations} 个样本")
                
                iteration += 1
                if iteration > n_iterations * 100:  # 防止无限循环
                    print("警告：接受率过低，提前停止")
                    break
            
            # 计算接受率
            acceptance_rate = accepted / iteration
            
            return {
                'beta_samples': beta_samples[n_burnin:accepted],
                'sigma2_samples': sigma2_samples[n_burnin:accepted],
                'n_iterations': accepted,
                'n_burnin': n_burnin,
                'acceptance_rate': acceptance_rate
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples