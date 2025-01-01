"""
使用CmdStanPy实现的HMC采样器
"""

import numpy as np
from cmdstanpy import CmdStanModel
import os
from ..base import BaseAlgorithm
import time


class HMCStanSampler(BaseAlgorithm):
    """使用Stan的HMC采样器"""
    
    def __init__(self, random_seed=42):
        super().__init__("HMC_Stan", random_seed)
        np.random.seed(random_seed)
        
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix is None:
            raise ValueError("CONDA_PREFIX environment variable not found")
            
        cmdstan_path = os.path.join(conda_prefix, 'Library', 'bin', 'cmdstan')
        os.environ['CMDSTAN'] = cmdstan_path

        if not os.path.exists('stan_models'):
            os.makedirs('stan_models')
            
        # 创建模型
        self._create_model_files()
        
        try:
            # 编译所有模型
            self.gmm_model = CmdStanModel(stan_file='stan_models/gmm_model.stan')
            self.ssm_model = CmdStanModel(stan_file='stan_models/ssm_model.stan')
            self.sparse_model = CmdStanModel(stan_file='stan_models/sparse_model.stan')
        except Exception as e:
            print(f"模型编译错误: {str(e)}")
            print(f"当前CMDSTAN路径: {os.environ.get('CMDSTAN')}")
            raise
            
    def _create_model_files(self):
        """创建Stan模型文件"""
        # GMM模型的Stan代码
        self.gmm_code = """ 
        data {
          int<lower=1> N;  // 样本数
          int<lower=1> D;  // 维度
          matrix[N, D] X;  // 观测数据
        }

        parameters {
          simplex[2] pi;  // 混合权重
          vector[D] mu1;  // 第一个组分的均值
          vector[D] mu2;  // 第二个组分的均值
          vector<lower=0.01>[D] sigma1;  // 第一个组分的标准差，设置下限为0.01
          vector<lower=0.01>[D] sigma2;  // 第二个组分的标准差，设置下限为0.01
        }

        model {
          vector[2] log_prob;
          
          // 先验分布
          pi ~ dirichlet(rep_vector(1, 2));
          mu1 ~ normal(0, sqrt(10));
          mu2 ~ normal(0, sqrt(10));
          sigma1 ~ lognormal(0, 1);  // 使用对数正态分布确保标准差为正
          sigma2 ~ lognormal(0, 1);
          
          // 似然函数
          for (n in 1:N) {
            for (k in 1:2) {
              log_prob[k] = log(pi[k]);
              if (k == 1)
                log_prob[k] += normal_lpdf(X[n] | mu1, sigma1);
              else
                log_prob[k] += normal_lpdf(X[n] | mu2, sigma2);
            }
            target += log_sum_exp(log_prob);
          }
        }
        """
        
        # SSM模型的Stan代码
        self.ssm_code = """
        data {
          int<lower=1> T;  // 时间序列长度
          vector[T] y;     // 观测数据
        }

        parameters {
          vector[T] x;     // 隐状态
          real<lower=0> Q; // 状态噪声方差
          real<lower=0> R; // 观测噪声方差
        }

        model {
          // 先验分布
          Q ~ inv_gamma(2, 1);
          R ~ inv_gamma(2, 1);
          x[1] ~ normal(0, sqrt(10));
          
          // 状态转移
          for (t in 2:T) {
            real x_prev = x[t-1];
            real x_term = 1 / (1 + x_prev * x_prev);
            real pred = 0.5 * x_prev + 25 * x_prev * x_term + 8 * cos(1.2 * t);
            x[t] ~ normal(pred, sqrt(Q));
          }
          
          // 观测方程
          for (t in 1:T) {
            y[t] ~ normal(x[t] * x[t] / 20, sqrt(R));
          }
        }
        """
        
        # Sparse回归模型的Stan代码
        self.sparse_code = """
        data {
          int<lower=1> N;  // 样本数
          int<lower=1> D;  // 特征维度
          matrix[N, D] X;  // 特征矩阵
          vector[N] y;     // 目标变量
        }

        parameters {
          vector[D] beta;          // 回归系数
          real<lower=0> global_scale;    // 全局尺度参数
          vector<lower=0>[D] local_scale; // 局部尺度参数
          real<lower=0> sigma;     // 噪声标准差
        }

        transformed parameters {
          vector<lower=0>[D] scale;  // 总体尺度
          scale = local_scale * global_scale;
        }

        model {
          // 先验分布
          global_scale ~ student_t(3, 0, 1);
          local_scale ~ student_t(3, 0, 1);
          
          // 回归系数的分层先验
          for (d in 1:D) {
            target += log_mix(0.95,  // 混合比例：95% 概率为小值
                            normal_lpdf(beta[d] | 0, 0.1),  // 接近0的分量
                            normal_lpdf(beta[d] | 0, scale[d]));  // 可能的大值分量
          }
          
          // 噪声先验
          sigma ~ student_t(3, 0, 1);
          
          // 似然函数
          y ~ normal(X * beta, sigma);
        }

        generated quantities {
          vector[N] y_pred;    // 预测值
          vector[N] log_lik;   // 对数似然
          
          y_pred = X * beta;
          
          for (n in 1:N) {
            log_lik[n] = normal_lpdf(y[n] | y_pred[n], sigma);
          }
        }
        """
        
        # 写入模型文件
        with open('stan_models/gmm_model.stan', 'w') as f:
            f.write(self.gmm_code)
        with open('stan_models/ssm_model.stan', 'w') as f:
            f.write(self.ssm_code)
        with open('stan_models/sparse_model.stan', 'w') as f:
            f.write(self.sparse_code)
            
    def fit_gmm(self, data, n_iterations=10000, n_burnin=1000):
        """GMM模型的HMC采样
        
        参数:
        n_iterations: 总采样数，将平均分配到4条链上
        n_burnin: 每条链的预热期长度
        """
        X = data['observations']
        n_samples, dim = X.shape
        
        # 准备数据
        stan_data = {
            'N': n_samples,
            'D': dim,
            'X': X
        }
        
        # 运行采样，每条链采样 n_iterations/4 次
        start_time = time.time()
        fit = self.gmm_model.sample(
            data=stan_data,
            iter_sampling=n_iterations // 4,  # 每条链的采样数
            iter_warmup=n_burnin,
            chains=4,
            seed=self.random_seed
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # 提取样本
        mu1_samples = fit.stan_variable('mu1')
        mu2_samples = fit.stan_variable('mu2')
        sigma1_samples = fit.stan_variable('sigma1')
        sigma2_samples = fit.stan_variable('sigma2')
        pi_samples = fit.stan_variable('pi')
        
        # 转换为协方差矩阵形式
        sigma1_samples_full = np.array([np.diag(s**2) for s in sigma1_samples])
        sigma2_samples_full = np.array([np.diag(s**2) for s in sigma2_samples])
        
        return {
            'mu1_samples': mu1_samples,
            'mu2_samples': mu2_samples,
            'sigma1_samples': sigma1_samples_full,
            'sigma2_samples': sigma2_samples_full,
            'pi_samples': pi_samples,
            'n_iterations': n_iterations,
            'n_burnin': n_burnin,
            'runtime': runtime
        }
        
    def fit_ssm(self, data, n_iterations=10000, n_burnin=1000):
        """SSM模型的HMC采样
        
        参数:
        n_iterations: 总采样数，将平均分配到4条链上
        n_burnin: 每条链的预热期长度
        """
        y = data['observations']
        T = len(y)
        
        # 准备数据
        stan_data = {
            'T': T,
            'y': y
        }
        
        # 运行采样，每条链采样 n_iterations/4 次
        start_time = time.time()
        fit = self.ssm_model.sample(
            data=stan_data,
            iter_sampling=n_iterations // 4,  # 每条链的采样数
            iter_warmup=n_burnin,
            chains=4,
            seed=self.random_seed
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # 提取样本
        x_samples = fit.stan_variable('x')
        Q_samples = fit.stan_variable('Q')
        R_samples = fit.stan_variable('R')
        
        return {
            'state_samples': x_samples,
            'Q_samples': Q_samples,
            'R_samples': R_samples,
            'n_iterations': n_iterations,
            'n_burnin': n_burnin,
            'runtime': runtime
        }
        
    def fit_sparse(self, data, n_iterations=10000, n_burnin=1000):
        """稀疏回归的HMC采样
        
        参数:
        data: 训练数据，由 SparseRegressionScenario.generate_data() 生成
        n_iterations: 总采样数，将平均分配到4条链上
        n_burnin: 每条链的预热期长度
        """
        X_train, y_train = data  
        
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
            
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            
        n_samples, dim = X_train.shape
        
        # 准备数据
        stan_data = {
            'N': n_samples,
            'D': dim,
            'X': X_train,
            'y': y_train
        }
        
        # 运行采样
        start_time = time.time()
        fit = self.sparse_model.sample(
            data=stan_data,
            iter_sampling=n_iterations // 4,
            iter_warmup=n_burnin,
            chains=4,
            seed=self.random_seed,
            adapt_delta=0.95,
            max_treedepth=12
        )
        end_time = time.time()
        runtime = end_time - start_time
        
        # 提取样本
        beta_samples = fit.stan_variable('beta')
        sigma_samples = fit.stan_variable('sigma')
        global_scale_samples = fit.stan_variable('global_scale')
        local_scale_samples = fit.stan_variable('local_scale')
        scale_samples = fit.stan_variable('scale')
        
        # 计算后验均值
        beta_mean = np.mean(beta_samples, axis=0)
        
        return {
            'beta_samples': beta_samples,
            'beta_mean': beta_mean,
            'sigma_samples': sigma_samples,
            'scale_samples': scale_samples,
            'global_scale_samples': global_scale_samples,
            'local_scale_samples': local_scale_samples,
            'n_iterations': n_iterations,
            'n_burnin': n_burnin,
            'runtime': runtime
        } 