import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self):
        pass
        
    def compute_ess(self, chain):
        """计算有效样本量 ESS 
            chain: 样本链
        """
        n = len(chain)
        if n < 2:
            return n
        
        # 计算均值和方差
        mean = np.mean(chain)
        chain_centered = chain - mean
        var = np.var(chain, ddof=1) 
        # 方差太小则说明样本几乎不变
        if var < 1e-10:  
            print(f"警告: 样本链方差过小 ({var:.2e})")
            return 1.0
        
        # 计算自相关系数（动态调整最大滞后阶数）
        max_lag = min(n - 1, int(10 * np.log10(n)))  
        auto_corr = np.zeros(max_lag + 1) 
        
        auto_corr[0] = 1.0  #lag0的自相关系数恒为1
        for k in range(1, max_lag + 1):
            # 使用无偏估计计算自协方差
            auto_cov_k = np.sum(chain_centered[:(n-k)] * chain_centered[k:]) / (n - k)
            auto_corr[k] = auto_cov_k / var
        print(f"前5个自相关系数: {auto_corr[1:6]}")
        
        # 使用Geyer的初始正序列法
        # 将自相关系数配对（从lag 1开始配对）
        paired_sums = []
        for k in range(1, len(auto_corr)-1, 2):
            sum_k = auto_corr[k] + auto_corr[k+1]
            if sum_k < 0: 
                break
            paired_sums.append(sum_k)
        
        if len(paired_sums) > 0:
            print(f"配对和: {paired_sums[:3]}...")  # 打印前三个配对和
        
        # 检查单调性（如果不单调，截断）
        for k in range(1, len(paired_sums)):
            if paired_sums[k] > paired_sums[k-1]:  
                paired_sums = paired_sums[:k]
                break
        
        # 计算ESS
        tau = 2.0 * sum(paired_sums) 
        print(f"tau值: {tau:.2f}")

        ess = min(float(n), n / max(1.0 + tau, 1.0))
        print(f"最终ESS: {ess:.2f} (总样本数: {n})")
        return ess
        
    def evaluate_gmm(self, samples, true_params, data):
        """评估GMM结果"""
        (train_data, test_data) = data
        #效率指标
        efficiency_metrics = {
            'wall_time': samples['runtime'],
            'iterations': samples['n_iterations'],
            'samples_per_second': samples['n_iterations'] / samples['runtime']
        }
        
        #参数估计误差
        mu1_mean = np.mean(samples['mu1_samples'], axis=0)
        mu2_mean = np.mean(samples['mu2_samples'], axis=0)
        sigma1_mean = np.mean(samples['sigma1_samples'], axis=0)
        sigma2_mean = np.mean(samples['sigma2_samples'], axis=0)
        pi_mean = np.mean(samples['pi_samples'], axis=0)
        
        #考虑标签切换
        error1 = (np.linalg.norm(mu1_mean - true_params['mu1']) + 
                 np.linalg.norm(mu2_mean - true_params['mu2']))
        error2 = (np.linalg.norm(mu1_mean - true_params['mu2']) + 
                 np.linalg.norm(mu2_mean - true_params['mu1']))
        
        #选择较小的误差并相应调整其他参数的评估
        if error1 < error2:
            mu_error = error1 / 2
            sigma_error = 0.5 * (np.linalg.norm(sigma1_mean - true_params['sigma1'], 'fro') + 
                                np.linalg.norm(sigma2_mean - true_params['sigma2'], 'fro'))
            pi_error = np.linalg.norm(pi_mean - true_params['weights'])
            order = [0, 1]
        else:
            mu_error = error2 / 2
            sigma_error = 0.5 * (np.linalg.norm(sigma1_mean - true_params['sigma2'], 'fro') + 
                                np.linalg.norm(sigma2_mean - true_params['sigma1'], 'fro'))
            pi_error = np.linalg.norm(pi_mean[::-1] - true_params['weights'])
            order = [1, 0]
        
        # 计算ESS
        ess_mu1 = min([self.compute_ess(samples['mu1_samples'][:, d]) for d in range(samples['mu1_samples'].shape[1])])
        ess_mu2 = min([self.compute_ess(samples['mu2_samples'][:, d]) for d in range(samples['mu2_samples'].shape[1])])
        
        # 确保协方差矩阵是正定的（添加小的对角项）
        def ensure_positive_definite(cov):
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            if min_eig < 1e-4:
                cov += (1e-4 - min_eig) * np.eye(cov.shape[0])
            #确保对称性
            cov = (cov + cov.T) / 2
            return cov
        
        # 计算训练集对数似然
        train_log_likelihood = 0
        X_train = train_data['observations']
        sigma1_mean = ensure_positive_definite(sigma1_mean)
        sigma2_mean = ensure_positive_definite(sigma2_mean)
        
        for x in X_train:
            log_ll1 = np.log(max(pi_mean[order[0]], 1e-10)) + multivariate_normal.logpdf(x, mu1_mean, sigma1_mean)
            log_ll2 = np.log(max(pi_mean[order[1]], 1e-10)) + multivariate_normal.logpdf(x, mu2_mean, sigma2_mean)
            max_log = max(log_ll1, log_ll2)
            train_log_likelihood += max_log + np.log(np.exp(log_ll1 - max_log) + np.exp(log_ll2 - max_log))
        train_log_likelihood /= len(X_train)
        
        # 计算测试集对数似然
        test_log_likelihood = 0
        X_test = test_data['observations']
        
        for x in X_test:
            log_ll1 = np.log(max(pi_mean[order[0]], 1e-10)) + multivariate_normal.logpdf(x, mu1_mean, sigma1_mean)
            log_ll2 = np.log(max(pi_mean[order[1]], 1e-10)) + multivariate_normal.logpdf(x, mu2_mean, sigma2_mean)
            max_log = max(log_ll1, log_ll2)
            test_log_likelihood += max_log + np.log(np.exp(log_ll1 - max_log) + np.exp(log_ll2 - max_log))
        test_log_likelihood /= len(X_test)
        
        return {
            **efficiency_metrics,
            'mu_error': mu_error,
            'sigma_error': sigma_error,
            'pi_error': pi_error,
            'train_log_likelihood': train_log_likelihood,
            'test_log_likelihood': test_log_likelihood,
            'ess_mu1': ess_mu1,
            'ess_mu2': ess_mu2
        }
        
    def evaluate_ssm(self, samples, true_params, data):
        """评估SSM结果"""
        (train_data, test_data) = data
        # 效率指标
        efficiency_metrics = {
            'wall_time': samples['runtime'],
            'iterations': samples['n_iterations'],
            'samples_per_second': samples['n_iterations'] / samples['runtime']
        }
        
        # 状态估计 - 训练集
        state_mean = np.mean(samples['state_samples'], axis=0)
        train_rmse_state = np.sqrt(np.mean((state_mean - train_data['states'])**2))
        
        # 一步预测误差 - 训练集
        train_pred_errors = []
        for t in range(len(train_data['states'])-1):
            pred = (0.5 * state_mean[t] + 
                    25 * state_mean[t] / (1 + state_mean[t]**2) + 
                    8 * np.cos(1.2*(t+1)))
            train_pred_errors.append((pred - train_data['states'][t+1])**2)
        train_rmse_pred = np.sqrt(np.mean(train_pred_errors))
        
        # 状态估计 - 测试集
        test_state_mean = np.mean(samples['state_samples'], axis=0)[-len(test_data['states']):]
        test_rmse_state = np.sqrt(np.mean((test_state_mean - test_data['states'])**2))
        
        # 一步预测误差 - 测试集
        test_pred_errors = []
        for t in range(len(test_data['states'])-1):
            pred = (0.5 * test_state_mean[t] + 
                    25 * test_state_mean[t] / (1 + test_state_mean[t]**2) + 
                    8 * np.cos(1.2*(t+1)))
            test_pred_errors.append((pred - test_data['states'][t+1])**2)
        test_rmse_pred = np.sqrt(np.mean(test_pred_errors))
        
        # 分别计算Q和R的相对误差
        Q_mean = np.mean(samples['Q_samples'])
        R_mean = np.mean(samples['R_samples'])
        Q_error = np.abs(Q_mean - true_params['Q']) / true_params['Q']
        R_error = np.abs(R_mean - true_params['R']) / true_params['R']
        
        return {
            **efficiency_metrics,
            'train_state_rmse': train_rmse_state,
            'train_prediction_rmse': train_rmse_pred,
            'test_state_rmse': test_rmse_state,
            'test_prediction_rmse': test_rmse_pred,
            'Q_error': Q_error,  # 状态噪声参数误差
            'R_error': R_error   # 观测噪声参数误差
        }
        
    def evaluate_sparse(self, samples, true_params, data):
        """评估稀疏回归结果"""
        (train_data, test_data) = data
        
        # 从训练数据中提取X和y
        if isinstance(train_data, np.ndarray):
            if train_data.shape[0] < train_data.shape[1]:
                train_data = train_data.T
            X_train = train_data[:, :-1]
            y_train = train_data[:, -1]
        else:
            X_train, y_train = train_data
            
        # 从测试数据中提取X和y
        if isinstance(test_data, np.ndarray):
            if test_data.shape[0] < test_data.shape[1]:
                test_data = test_data.T
            X_test = test_data[:, :-1]
            y_test = test_data[:, -1]
        else:
            X_test, y_test = test_data
            
        # 确保所有数据都是numpy数组
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # 效率指标
        efficiency_metrics = {
            'wall_time': samples['runtime'],
            'iterations': samples['n_iterations'],
            'samples_per_second': samples['n_iterations'] / samples['runtime']
        }
        
        # 变量选择评估
        beta_mean = np.mean(samples['beta_samples'], axis=0)
        
        # 确保维度匹配
        if len(true_params) != len(beta_mean):
            print(f"Warning: Dimension mismatch - true_params: {len(true_params)}, beta: {len(beta_mean)}")
            # 如果维度不匹配，我们只评估共同的维度
            min_dim = min(len(true_params), len(beta_mean))
            true_params_eval = true_params[:min_dim]
            beta_mean_eval = beta_mean[:min_dim]
        else:
            true_params_eval = true_params
            beta_mean_eval = beta_mean
            
        # 计算变量选择指标
        beta_nonzero = np.abs(beta_mean_eval) > 0.1  # 阈值
        true_nonzero = np.abs(true_params_eval) > 0
        
        precision = precision_score(true_nonzero, beta_nonzero, zero_division=1)
        recall = recall_score(true_nonzero, beta_nonzero)
        f1 = f1_score(true_nonzero, beta_nonzero)
        
        # 预测性能评估
        try:
            # 训练集评估
            y_train_pred = X_train @ beta_mean
            train_prediction_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
            
            # 测试集评估
            y_test_pred = X_test @ beta_mean
            test_prediction_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        except ValueError as e:
            print(f"Warning: Error in prediction - {str(e)}")
            print(f"X_train shape: {X_train.shape}, beta_mean shape: {beta_mean.shape}")
            train_prediction_rmse = np.nan
            test_prediction_rmse = np.nan
        
        # 稀疏度
        sparsity = 1 - np.mean(beta_nonzero)
        
        return {
            **efficiency_metrics,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_prediction_rmse': train_prediction_rmse,
            'test_prediction_rmse': test_prediction_rmse,
            'sparsity': sparsity
        }