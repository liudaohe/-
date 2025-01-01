import numpy as np
import time
from scipy.stats import multivariate_normal, invwishart
from ..base import BaseAlgorithm
from ..initialization import initialize_gmm_params, initialize_ssm_params, initialize_sparse_params

class GibbsSampler(BaseAlgorithm):
    """Gibbs采样算法"""
    
    def __init__(self, random_seed=42):
        super().__init__("Gibbs", random_seed)
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的Gibbs采样实现"""
        def _fit():
            X = data['observations']
            n_samples, dim = X.shape
            
            # 使用统一初始化和先验
            params = initialize_gmm_params(dim, self.random_seed)
            mu1, mu2 = params['mu1'], params['mu2']
            sigma1, sigma2 = params['sigma1'], params['sigma2']
            pi = params['pi']
            
            # 存储先验对象
            self.gmm_prior = params['prior']
            
            # 存储采样结果
            mu1_samples = np.zeros((n_iterations, dim))
            mu2_samples = np.zeros((n_iterations, dim))
            sigma1_samples = np.zeros((n_iterations, dim, dim))
            sigma2_samples = np.zeros((n_iterations, dim, dim))
            pi_samples = np.zeros((n_iterations, 2))
            
            # Gibbs采样
            for t in range(n_iterations):
                # 1. 采样隐变量z
                z = self._sample_z(X, mu1, mu2, sigma1, sigma2, pi)
                
                # 2. 采样均值
                mu1 = self._sample_mu(X[z==0], sigma1)
                mu2 = self._sample_mu(X[z==1], sigma2)
                
                # 3. 采样协方差
                sigma1 = self._sample_sigma(X[z==0], mu1)
                sigma2 = self._sample_sigma(X[z==1], mu2)
                
                # 4. 采样混合权重
                pi = self._sample_pi(z)
                
                # 存储样本
                mu1_samples[t] = mu1
                mu2_samples[t] = mu2
                sigma1_samples[t] = sigma1
                sigma2_samples[t] = sigma2
                pi_samples[t] = pi
            
            return {
                'mu1_samples': mu1_samples[n_burnin:],
                'mu2_samples': mu2_samples[n_burnin:],
                'sigma1_samples': sigma1_samples[n_burnin:],
                'sigma2_samples': sigma2_samples[n_burnin:],
                'pi_samples': pi_samples[n_burnin:],
                'n_iterations': n_iterations,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_ssm(self, data, n_iterations=1000, n_burnin=100):
        """状态空间模型的Gibbs采样实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            params = initialize_ssm_params(T, self.random_seed)
            x = params['x']
            Q = params['Q']
            R = params['R']
            
            self.ssm_prior = params['prior']
            
            x_samples = np.zeros((n_iterations, T))
            Q_samples = np.zeros(n_iterations)
            R_samples = np.zeros(n_iterations)
            
            # Gibbs采样
            for t in range(n_iterations):
                # 1. 采样状态序列
                x = self._sample_states(y, Q, R)
                
                # 2. 采样噪声参数
                Q = self._sample_Q(x)
                R = self._sample_R(y, x)
                
                # 存储样本
                x_samples[t] = x
                Q_samples[t] = Q
                R_samples[t] = R
            
            return {
                'state_samples': x_samples[n_burnin:],
                'Q_samples': Q_samples[n_burnin:],
                'R_samples': R_samples[n_burnin:],
                'n_iterations': n_iterations,
                'n_burnin': n_burnin
            }

        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_sparse(self, data, n_iterations=1000, n_burnin=100):
        """稀疏回归的Gibbs采样实现"""
        def _fit():
            X, y = data  
            n_samples, n_features = X.shape
            
            # 使用统一初始化和先验
            params = initialize_sparse_params(n_features, self.random_seed)
            beta = params['beta']
            sigma2 = params['sigma2']
            self.sparse_prior = params['prior']
            beta_samples = np.zeros((n_iterations, n_features))
            sigma2_samples = np.zeros(n_iterations)
            
            # Gibbs采样
            for t in range(n_iterations):
                for j in range(n_features):
                    Xj = X[:, j]
                    resid = y - X @ beta + beta[j] * Xj
                    var_j = 1 / (np.sum(Xj**2) / sigma2 + 2 * self.sparse_prior.lambda_prior)
                    mean_j = var_j * np.sum(Xj * resid) / sigma2
                    
                    # 直接从条件后验分布采样
                    beta[j] = np.random.normal(mean_j, np.sqrt(var_j))

                resid = y - X @ beta
                shape = self.sparse_prior.sigma2_a + n_samples / 2
                scale = self.sparse_prior.sigma2_scale + np.sum(resid**2) / 2
                sigma2 = 1 / np.random.gamma(shape, 1/scale)
                
                beta_samples[t] = beta
                sigma2_samples[t] = sigma2
            
            return {
                'beta_samples': beta_samples[n_burnin:],
                'sigma2_samples': sigma2_samples[n_burnin:],
                'n_iterations': n_iterations,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    # GMM的辅助函数
    def _sample_z(self, X, mu1, mu2, sigma1, sigma2, pi):
        """采样隐变量z"""
        n_samples = len(X)
        z = np.zeros(n_samples)
        
        # 计算每个样本属于第二个分量的概率
        p1 = multivariate_normal.pdf(X, mu1, sigma1)
        p2 = multivariate_normal.pdf(X, mu2, sigma2)
        
        # 计算归一化概率
        prob = (pi[1] * p2) / (pi[0] * p1 + pi[1] * p2)
        
        # 二项分布采样
        z = np.random.binomial(1, prob)
        return z
    
    def _sample_mu(self, X, sigma):
        """采样均值"""
        if len(X) == 0:
            return np.zeros_like(sigma[0])
        
        n = len(X)
        X_mean = np.mean(X, axis=0)
        
        #后验参数
        prior_mean = self.gmm_prior.mu_prior.mean
        prior_cov = self.gmm_prior.mu_prior.cov
        
        posterior_var = np.linalg.inv(np.linalg.inv(prior_cov) + n * np.linalg.inv(sigma))
        posterior_mean = posterior_var @ (np.linalg.inv(prior_cov) @ prior_mean + n * np.linalg.inv(sigma) @ X_mean)
        
        return np.random.multivariate_normal(posterior_mean, posterior_var)
    
    def _sample_sigma(self, X, mu):
        """采样协方差矩阵"""
        if len(X) == 0:
            return np.eye(len(mu))
        
        n = len(X)
        S = np.sum([(x - mu).reshape(-1,1) @ (x - mu).reshape(1,-1) for x in X], axis=0)
        
        #后验参数
        df = self.gmm_prior.sigma_prior.df
        scale = self.gmm_prior.sigma_prior.scale
        
        posterior_df = df + n
        posterior_scale = scale + S
        
        return invwishart.rvs(df=posterior_df, scale=posterior_scale)
    
    def _sample_pi(self, z):
        """采样混合权重"""
        #后验参数
        alpha = self.gmm_prior.pi_prior.alpha
        counts = np.array([np.sum(z == 0), np.sum(z == 1)])
        return np.random.dirichlet(alpha + counts)
    
    # SSM的辅助函数
    def _sample_states(self, y, Q, R):
        """采样状态序列"""
        T = len(y)
        x = np.zeros(T)
        
        # 使用先验对象的参数
        x[0] = np.random.normal(self.ssm_prior.x0_mean, np.sqrt(self.ssm_prior.x0_var))
        
        # 前向滤波
        filtered_means = np.zeros(T)
        filtered_vars = np.zeros(T)
        
        filtered_means[0] = self.ssm_prior.x0_mean
        filtered_vars[0] = max(self.ssm_prior.x0_var, 1e-6)  # 确保初始方差为正
        
        # 预计算sigma点权重
        n = 1  # 状态维度
        alpha, beta, kappa = 1e-3, 2.0, 0.0
        lambda_ = alpha**2 * (n + kappa) - n
        
        #权重向量化计算
        Wm = np.array([lambda_ / (n + lambda_)] + [1 / (2*(n + lambda_))] * 2)
        Wc = np.array([lambda_ / (n + lambda_) + (1 - alpha**2 + beta)] + [1 / (2*(n + lambda_))] * 2)
        
        for t in range(1, T):
            # 确保方差为正
            filtered_vars[t-1] = max(filtered_vars[t-1], 1e-6)
            
            #生成sigma点
            sigma_points = np.array([
                filtered_means[t-1],
                filtered_means[t-1] + np.sqrt((n + lambda_) * filtered_vars[t-1]),
                filtered_means[t-1] - np.sqrt((n + lambda_) * filtered_vars[t-1])
            ])
            
            #状态转移
            pred_sigma_points = (0.5 * sigma_points + 
                               25 * sigma_points / (1 + sigma_points**2) + 
                               8 * np.cos(1.2*t))
            
            #预测步
            pred_mean = np.sum(Wm * pred_sigma_points)
            pred_var = max(np.sum(Wc * (pred_sigma_points - pred_mean)**2) + Q, 1e-6)  # 确保预测方差为正
            
            #观测预测 
            obs_sigma_points = pred_sigma_points**2 / 20
            obs_pred = np.sum(Wm * obs_sigma_points)
            
            #更新步
            Pxy = np.sum(Wc * (pred_sigma_points - pred_mean) * (obs_sigma_points - obs_pred))
            Pyy = max(np.sum(Wc * (obs_sigma_points - obs_pred)**2) + R, 1e-6)  # 确保观测方差为正
            K = Pxy / Pyy
            
            filtered_means[t] = pred_mean + K * (y[t] - obs_pred)
            filtered_vars[t] = max(pred_var - K**2 * Pyy, 1e-6)  # 确保滤波方差为正
        
        # 后向采样
        filtered_vars[T-1] = max(filtered_vars[T-1], 1e-6)  # 确保最后一个方差为正
        x[T-1] = np.random.normal(filtered_means[T-1], np.sqrt(filtered_vars[T-1]))
        
        # 预计算状态转移函数
        for t in range(T-2, -1, -1):
            next_pred = 0.5 * x[t+1] + 25 * x[t+1] / (1 + x[t+1]**2) + 8 * np.cos(1.2*(t+1))
            back_var = max((filtered_vars[t] * Q) / (filtered_vars[t] + Q), 1e-6)  # 确保后向方差为正
            back_mean = (Q * filtered_means[t] + filtered_vars[t] * next_pred) / (filtered_vars[t] + Q)
            x[t] = np.random.normal(back_mean, np.sqrt(back_var))
        
        return x
    
    def _sample_Q(self, x):
        """采样状态噪声方差"""
        T = len(x)
        
        # 使用先验对象的参数
        shape = self.ssm_prior.Q_a + (T - 1) / 2
        
        # 向量化计算状态转移
        t_range = np.arange(1, T)
        pred = (0.5 * x[:-1] + 
                25 * x[:-1] / (1 + x[:-1]**2) + 
                8 * np.cos(1.2 * t_range))
        
        # 向量化计算残差
        residuals = x[1:] - pred
        
        # 计算scale
        scale = self.ssm_prior.Q_scale + np.sum(residuals**2) / 2
        scale = max(scale, 1e-6)
        
        # 直接采样
        gamma_sample = np.random.gamma(shape, 1/scale)
        return 1 / gamma_sample
    
    def _sample_R(self, y, x):
        """采样观测噪声方差"""
        T = len(y)
        
        # 使用先验对象的参数
        shape = self.ssm_prior.R_a + T / 2
        
        # 向量化计算残差
        residuals = y - x**2 / 20
        
        # 计算scale
        scale = self.ssm_prior.R_scale + np.sum(residuals**2) / 2
        scale = max(scale, 1e-6)
        
        # 直接采样
        gamma_sample = np.random.gamma(shape, 1/scale)
        return 1 / gamma_sample