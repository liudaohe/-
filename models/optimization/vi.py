import numpy as np
from scipy.stats import multivariate_normal, wishart
from scipy.special import digamma, gammaln
from ..base import BaseAlgorithm
from ..initialization import initialize_gmm_params, initialize_ssm_params, initialize_sparse_params

class VISampler(BaseAlgorithm):
    """变分推断采样器"""
    
    def __init__(self, random_seed=42, max_iter=100, tol=1e-6):
        super().__init__("VI", random_seed)
        self.max_iter = max_iter
        self.tol = tol
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的变分推断实现"""
        def _fit():
            X = data['observations']
            n_samples, dim = X.shape
            
            # 使用统一的初始化
            params = initialize_gmm_params(dim, self.random_seed)
            mu1 = params['mu1']
            mu2 = params['mu2']
            mu_prior_var = params['mu_prior_var']
            pi_alpha = params['pi_prior'].copy()
            nu1 = params['sigma_prior_df']
            nu2 = params['sigma_prior_df']
            W1 = params['sigma_prior_scale'].copy()
            W2 = params['sigma_prior_scale'].copy()
            
            # 初始化协方差矩阵
            S1 = np.eye(dim)
            S2 = np.eye(dim)
            
            # 预计算
            X_squared = np.sum(X**2, axis=1)
            
            elbo_old = -np.inf
            for iter in range(self.max_iter):
                # E步：计算责任
                # 预计算常数项
                log_det1 = np.log(np.linalg.det(S1) + 1e-10)
                log_det2 = np.log(np.linalg.det(S2) + 1e-10)
                S1_inv = np.linalg.inv(S1)
                S2_inv = np.linalg.inv(S2)
                
                # 向量化计算对数概率
                x_mu1 = X - mu1
                x_mu2 = X - mu2
                
                # 使用矩阵乘法代替循环
                log_resp1 = -0.5 * (dim * np.log(2*np.pi) + log_det1 + 
                                  np.sum(x_mu1 @ S1_inv * x_mu1, axis=1))
                log_resp2 = -0.5 * (dim * np.log(2*np.pi) + log_det2 + 
                                  np.sum(x_mu2 @ S2_inv * x_mu2, axis=1))
                
                log_resp = np.column_stack([log_resp1, log_resp2])
                
                # 添加混合权重的影响
                log_pi = digamma(pi_alpha) - digamma(pi_alpha.sum())
                log_resp += log_pi
                
                # 标准化责任
                log_resp_max = np.max(log_resp, axis=1, keepdims=True)
                resp = np.exp(log_resp - log_resp_max)
                resp /= resp.sum(axis=1, keepdims=True)
                
                # M步：更新参数
                N = resp.sum(axis=0)
                
                # 更新均值
                mu1 = (resp[:,0:1] * X).sum(axis=0) / (N[0] + 1e-10)
                mu2 = (resp[:,1:2] * X).sum(axis=0) / (N[1] + 1e-10)
                
                # 更新协方差矩阵
                x_mu1 = X - mu1
                x_mu2 = X - mu2
                S1 = (resp[:,0:1].reshape(-1,1,1) * x_mu1[:,:,np.newaxis] @ x_mu1[:,np.newaxis,:]).sum(axis=0)
                S2 = (resp[:,1:2].reshape(-1,1,1) * x_mu2[:,:,np.newaxis] @ x_mu2[:,np.newaxis,:]).sum(axis=0)
                
                S1 = (S1 + params['sigma_prior_scale']) / (N[0] + nu1 + dim + 1)
                S2 = (S2 + params['sigma_prior_scale']) / (N[1] + nu2 + dim + 1)
                
                # 确保协方差矩阵是正定的
                S1 = 0.5 * (S1 + S1.T)
                S2 = 0.5 * (S2 + S2.T)
                min_eig1 = np.min(np.real(np.linalg.eigvals(S1)))
                min_eig2 = np.min(np.real(np.linalg.eigvals(S2)))
                if min_eig1 < 1e-6:
                    S1 += (1e-6 - min_eig1) * np.eye(dim)
                if min_eig2 < 1e-6:
                    S2 += (1e-6 - min_eig2) * np.eye(dim)
                
                # 更新混合权重
                pi_alpha = params['pi_prior'].copy() + N
                
                # 计算ELBO (向量化)
                elbo = np.sum(resp * log_resp1[:,np.newaxis]) + np.sum(resp * log_resp2[:,np.newaxis])
                elbo -= np.sum(resp * np.log(resp + 1e-10))
                
                # 混合权重的KL散度
                elbo += np.sum(resp @ log_pi)
                elbo += gammaln(pi_alpha.sum()) - np.sum(gammaln(pi_alpha))
                elbo -= gammaln(params['pi_prior'].sum()) - np.sum(gammaln(params['pi_prior']))
                elbo -= np.sum((pi_alpha - params['pi_prior']) * (digamma(pi_alpha) - digamma(pi_alpha.sum())))
                
                # 均值和协方差的KL散度
                mu_prior_prec = np.linalg.inv(mu_prior_var)
                elbo -= 0.5 * (np.sum((mu1 - 0) * mu_prior_prec @ (mu1 - 0)) + 
                              np.sum((mu2 - 0) * mu_prior_prec @ (mu2 - 0)))
                
                for S, nu, W in [(S1, nu1, W1), (S2, nu2, W2)]:
                    W_inv = np.linalg.inv(W)
                    log_det_W = np.log(np.linalg.det(W) + 1e-10)
                    log_det_S = np.log(np.linalg.det(S) + 1e-10)
                    
                    elbo += 0.5 * ((nu + dim + 1) * log_det_S - (params['sigma_prior_df'] + dim + 1) * log_det_W)
                    elbo -= 0.5 * nu * np.trace(W_inv @ S)
                    elbo += 0.5 * params['sigma_prior_df'] * dim
                    elbo += 0.5 * nu * dim * np.log(2) + 0.5 * nu * log_det_S + \
                           0.5 * nu * dim * np.log(2/nu) + np.sum(gammaln(0.5 * (nu - np.arange(dim))))
                
                if abs(elbo - elbo_old) < self.tol:
                    break
                elbo_old = elbo
            
            # 生成样本 (批量处理)
            mu1_samples = np.random.multivariate_normal(mu1, mu_prior_var, size=n_iterations)
            mu2_samples = np.random.multivariate_normal(mu2, mu_prior_var, size=n_iterations)
            
            # 计算变分后验分布的参数
            nu1_post = nu1 + N[0]  # 后验自由度
            nu2_post = nu2 + N[1]  # 后验自由度
            
            # 计算变分后验分布的scale矩阵

            W1_post = (nu1_post) * np.linalg.inv(S1)  
            W2_post = (nu2_post) * np.linalg.inv(S2)  
            
            # 预分配内存
            sigma1_samples = np.zeros((n_iterations, dim, dim))
            sigma2_samples = np.zeros((n_iterations, dim, dim))
            
            # 批量生成Wishart样本
            batch_size = 100
            for i in range(0, n_iterations, batch_size):
                end_idx = min(i + batch_size, n_iterations)
                batch_len = end_idx - i
                
                # 直接从Wishart分布采样协方差矩阵
                sigma1_samples[i:i+batch_len] = wishart.rvs(df=nu1_post, scale=np.linalg.inv(W1_post), size=batch_len)
                sigma2_samples[i:i+batch_len] = wishart.rvs(df=nu2_post, scale=np.linalg.inv(W2_post), size=batch_len)
                
                # 确保数值稳定性
                for j in range(batch_len):
                    # 确保对称性
                    sigma1_samples[i+j] = 0.5 * (sigma1_samples[i+j] + sigma1_samples[i+j].T)
                    sigma2_samples[i+j] = 0.5 * (sigma2_samples[i+j] + sigma2_samples[i+j].T)
                    
                    # 确保正定性
                    min_eig1 = np.min(np.real(np.linalg.eigvals(sigma1_samples[i+j])))
                    min_eig2 = np.min(np.real(np.linalg.eigvals(sigma2_samples[i+j])))
                    if min_eig1 < 1e-6:
                        sigma1_samples[i+j] += (1e-6 - min_eig1) * np.eye(dim)
                    if min_eig2 < 1e-6:
                        sigma2_samples[i+j] += (1e-6 - min_eig2) * np.eye(dim)
            
            # 生成混合权重样本
            pi_samples = np.random.dirichlet(pi_alpha, size=n_iterations)
            
            return {
                'mu1_samples': mu1_samples[n_burnin:],
                'mu2_samples': mu2_samples[n_burnin:],
                'sigma1_samples': sigma1_samples[n_burnin:],
                'sigma2_samples': sigma2_samples[n_burnin:],
                'pi_samples': pi_samples[n_burnin:],
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_ssm(self, data, n_iterations=1000, n_burnin=100):
        """SSM的变分推断实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            # 使用统一初始化
            params = initialize_ssm_params(T, self.random_seed)
            
            # 变分参数初始化
            x_mean = params['x'].copy()
            x_var = np.ones(T)
            Q_alpha = params['Q_prior_a'] * np.ones(T-1)
            Q_beta = params['Q_prior_scale'] * np.ones(T-1)
            R_alpha = params['R_prior_a'] * np.ones(T)
            R_beta = params['R_prior_scale'] * np.ones(T)
            
            # 对观测数据进行标准化
            y_mean = np.mean(y)
            y_std = np.std(y)
            y = (y - y_mean) / (y_std + 1e-10)
            x_mean = (x_mean - y_mean) / (y_std + 1e-10)
            
            # 预计算常量
            t_range = np.arange(T)
            cos_t = 8.0 * np.cos(1.2 * t_range)
            
            elbo_old = -np.inf
            for iter in range(self.max_iter):
                # 计算状态转移函数
                f = 0.5 * x_mean + 25.0 * x_mean / (1.0 + x_mean**2) + cos_t
                f_prime = 0.5 + 25.0 * (1 - x_mean**2) / (1.0 + x_mean**2)**2
                f_var = f_prime**2 * x_var
                
                # 计算观测函数
                h = x_mean**2 / 20.0
                H = x_mean / 10.0
                h_var = H**2 * x_var
                
                # 更新状态分布 q(x)
                Q_expect = Q_beta / Q_alpha
                R_expect = R_beta / R_alpha
                
                # 前向传播
                alpha_mean = np.zeros(T)
                alpha_var = np.zeros(T)
                alpha_mean[0] = 0
                alpha_var[0] = 10.0
                
                for t in range(1, T):
                    alpha_mean[t] = f[t-1]
                    alpha_var[t] = f_var[t-1] + Q_expect[t-1]
                
                # 后向传播
                beta_mean = np.zeros(T)
                beta_var = np.zeros(T)
                beta_mean[-1] = 0
                beta_var[-1] = 10.0
                
                for t in range(T-2, -1, -1):
                    beta_mean[t] = x_mean[t+1]
                    beta_var[t] = f_var[t] + Q_expect[t]
                
                # 组合前向后向消息
                prec = 1/alpha_var + 1/beta_var + H**2/R_expect
                x_var = 1/prec
                x_mean = x_var * (alpha_mean/alpha_var + beta_mean/beta_var + 
                                H*(y - h + H*x_mean)/R_expect)
                
                # 更新噪声参数
                err_state = x_mean[1:] - f[:-1]
                err_obs = y - h
                
                Q_alpha = params['Q_prior_a'] + 0.5
                Q_beta = params['Q_prior_scale'] + 0.5 * (err_state**2 + x_var[1:] + f_var[:-1])
                
                R_alpha = params['R_prior_a'] + 0.5
                R_beta = params['R_prior_scale'] + 0.5 * (err_obs**2 + h_var)
                
                # 计算ELBO
                elbo = (-0.5 * np.sum(np.log(2*np.pi*Q_expect)) - 
                       0.5 * np.sum((err_state**2 + x_var[1:] + f_var[:-1]) / Q_expect))
                
                elbo += (-0.5 * np.sum(np.log(2*np.pi*R_expect)) - 
                        0.5 * np.sum((err_obs**2 + h_var) / R_expect))
                
                # KL散度项
                elbo += -0.5 * np.sum(np.log(2*np.pi*x_var)) - 0.5 * T
                elbo += np.sum(gammaln(Q_alpha) - Q_alpha * np.log(Q_beta))
                elbo -= np.sum((params['Q_prior_a'] - 1) * (digamma(Q_alpha) - np.log(Q_beta)))
                elbo -= params['Q_prior_scale'] * np.sum(Q_alpha / Q_beta)
                
                elbo += np.sum(gammaln(R_alpha) - R_alpha * np.log(R_beta))
                elbo -= np.sum((params['R_prior_a'] - 1) * (digamma(R_alpha) - np.log(R_beta)))
                elbo -= params['R_prior_scale'] * np.sum(R_alpha / R_beta)
                
                if abs(elbo - elbo_old) < self.tol:
                    break
                elbo_old = elbo
            
            # 批量生成样本
            state_samples = np.random.normal(x_mean, np.sqrt(x_var), size=(n_iterations, T))
            Q_samples = np.zeros((n_iterations, T-1))
            R_samples = np.zeros((n_iterations, T))
            
            # 批量生成噪声样本
            batch_size = 1000
            for i in range(0, n_iterations, batch_size):
                end_idx = min(i + batch_size, n_iterations)
                batch_len = end_idx - i
                
                Q_samples[i:end_idx] = 1 / np.random.gamma(Q_alpha, 1/Q_beta, size=(batch_len, T-1))
                R_samples[i:end_idx] = 1 / np.random.gamma(R_alpha, 1/R_beta, size=(batch_len, T))
            
            # 还原标准化
            state_samples = state_samples * (y_std + 1e-10) + y_mean
            Q_samples *= (y_std + 1e-10)**2
            R_samples *= (y_std + 1e-10)**2
            
            Q_mean = np.mean(Q_samples, axis=0)
            R_mean = np.mean(R_samples, axis=0)
            
            return {
                'state_samples': state_samples[n_burnin:],
                'Q_samples': Q_mean,
                'R_samples': R_mean,
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_sparse(self, data, n_iterations=1000, n_burnin=100):
        """稀疏回归的变分推断实现"""
        def _fit():
            X, y = data
            n_samples, n_features = X.shape
            
            # 使用统一初始化
            params = initialize_sparse_params(n_features, random_seed=self.random_seed)
            
            # 变分参数初始化
            beta_mean = params['beta'].copy()
            beta_var = np.ones(n_features)
            tau_alpha = np.ones(n_features)
            tau_beta = np.ones(n_features)
            sigma2_alpha = params['sigma2_prior_a']
            sigma2_beta = params['sigma2_prior_scale']
            
            # 预计算
            XtX = X.T @ X
            Xty = X.T @ y
            diag_XtX = np.diag(XtX)
            
            elbo_old = -np.inf
            for iter in range(self.max_iter):
                # 更新
                sigma2_expect = sigma2_beta / sigma2_alpha
                tau_expect = tau_beta / tau_alpha
                
                # 批量更新beta
                Xbeta = X @ beta_mean
                prec = 1/sigma2_expect * diag_XtX + tau_expect
                beta_var = 1/prec
                
                for j in range(n_features):
                    r = y - Xbeta + X[:,j] * beta_mean[j]
                    mu = 1/sigma2_expect * X[:,j] @ r
                    beta_mean[j] = beta_var[j] * mu
                    Xbeta = Xbeta - X[:,j] * beta_mean[j] + X[:,j] * beta_mean[j]
                
                # 更新tau
                tau_alpha.fill(1.5)
                tau_beta = params['lambda_prior']**2/2 + 0.5 * (beta_mean**2 + beta_var)
                
                # 更新sigma2
                err = y - X @ beta_mean
                err_var = np.sum(X**2 * beta_var.reshape(1,-1), axis=1)
                sigma2_alpha = params['sigma2_prior_a'] + 0.5 * n_samples
                sigma2_beta = params['sigma2_prior_scale'] + 0.5 * (np.sum(err**2) + np.sum(err_var))
                
                # 向量化计算ELBO
                elbo = -0.5 * n_samples * np.log(2*np.pi*sigma2_expect)
                elbo += -0.5/sigma2_expect * (np.sum(err**2) + np.sum(err_var))
                elbo += 0.5 * np.sum(np.log(2*np.pi*np.e*beta_var))
                
                # tau的KL散度
                elbo += np.sum(gammaln(tau_alpha) - tau_alpha * np.log(tau_beta))
                elbo -= np.sum((1.5 - 1) * (digamma(tau_alpha) - np.log(tau_beta)))
                elbo -= params['lambda_prior']**2/2 * np.sum(tau_alpha / tau_beta)
                
                # sigma2的KL散度
                elbo += gammaln(sigma2_alpha) - sigma2_alpha * np.log(sigma2_beta)
                elbo -= (params['sigma2_prior_a'] - 1) * (digamma(sigma2_alpha) - np.log(sigma2_beta))
                elbo -= params['sigma2_prior_scale'] * sigma2_alpha / sigma2_beta
                
                if abs(elbo - elbo_old) < self.tol:
                    break
                elbo_old = elbo
            
            # 批量生成样本
            beta_samples = np.random.normal(beta_mean, np.sqrt(beta_var), size=(n_iterations, n_features))
            sigma2_samples = np.zeros(n_iterations)
            
            # 批量生成噪声样本
            batch_size = 1000
            for i in range(0, n_iterations, batch_size):
                end_idx = min(i + batch_size, n_iterations)
                batch_len = end_idx - i
                
                sigma2_samples[i:end_idx] = 1 / np.random.gamma(sigma2_alpha, 1/sigma2_beta, size=batch_len)
            
            return {
                'beta_samples': beta_samples[n_burnin:],
                'sigma2_samples': sigma2_samples[n_burnin:],
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples 