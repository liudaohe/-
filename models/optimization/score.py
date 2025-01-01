"""
Score优化算法实现 - 基于分布-分布-值范式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAlgorithm
from sklearn.mixture import GaussianMixture

from torch.optim.lr_scheduler import CosineAnnealingLR

class ScoreNet(nn.Module):
    """分数网络"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # 分别预测均值、协方差矩阵和混合权重
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2)  
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * input_dim * 2)  
        )
        
        self.pi_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 预测所有参数
        mu = self.mu_net(x) 
        sigma = self.sigma_net(x)  
        pi = self.pi_net(x)  
        
        return mu, sigma, pi

class ScoreOptimizer(BaseAlgorithm):
    """Score优化算法"""
    
    def __init__(self, random_seed=42):
        super().__init__("Score", random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 添加训练参数
        self.learning_rate = 1e-3
        self.n_epochs = 100
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.empty_cache()
    
    def _standardize_data(self, data):
        """标准化数据"""
        mean = data.mean(0)
        std = data.std(0) + 1e-5  # 添加小的常数避免除零
        standardized_data = (data - mean) / std
        return standardized_data, mean, std
    
    def _train_score(self, score_net, data_loader, task_type='gmm'):
        """训练Score网络"""
        optimizer = torch.optim.Adam(score_net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        
        for epoch in range(self.n_epochs):
            total_loss = 0
            n_batches = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # 添加噪声
                noise = torch.randn_like(batch) * np.sqrt(0.01)
                noisy_data = batch + noise
                
                # 计算score
                if task_type == 'gmm':
                    mu, sigma, pi = score_net(noisy_data)
                    # 计算GMM的负对数似然作为损失
                    mu = mu.view(-1, 2, batch.shape[1])  # [batch_size, 2, dim]
                    sigma = sigma.view(-1, 2, batch.shape[1], batch.shape[1])  # [batch_size, 2, dim, dim]
                    sigma = torch.matmul(sigma, sigma.transpose(-2, -1)) + \
                           torch.eye(batch.shape[1]).to(self.device) * 1e-4
                    
                    # 计算每个组分的负对数似然
                    nll = torch.zeros(batch.size(0), 2).to(self.device)
                    for k in range(2):
                        diff = (batch.unsqueeze(1) - mu[:, k:k+1]).transpose(1, 2)  # [batch_size, dim, 1]
                        inv_sigma = torch.inverse(sigma[:, k])  # [batch_size, dim, dim]
                        mahalanobis = torch.bmm(torch.bmm(diff.transpose(-2, -1), inv_sigma), diff)  # [batch_size, 1, 1]
                        log_det = torch.logdet(sigma[:, k])  # [batch_size]
                        nll[:, k] = 0.5 * (mahalanobis.squeeze(-1).squeeze(-1) + log_det + batch.shape[1] * np.log(2 * np.pi))
                    
                    # 使用混合权重计算最终损失
                    loss = torch.mean(-torch.logsumexp(-nll + torch.log(pi + 1e-10), dim=1))
                    
                elif task_type == 'ssm':
                    states, Q, R = score_net(noisy_data)
                    # 使用MSE损失
                    loss = F.mse_loss(states, batch)
                    
                elif task_type == 'sparse':
                    beta, sigma2 = score_net(noisy_data)
                    # 使用MSE损失
                    loss = F.mse_loss(beta, batch)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        return score_net
    
    def _langevin_dynamics(self, score_net, n_samples, input_dim, n_steps=1000, step_size=0.01):
        """使用Langevin动力学生成样本"""
        x = torch.randn(n_samples, input_dim).to(self.device)
        
        for t in range(n_steps):
            noise = torch.randn_like(x) * np.sqrt(2 * step_size)
            with torch.no_grad():  # 在计算score时不需要梯度
                score = score_net(x)
            x = x + step_size * score + noise
        
        return x.detach().cpu().numpy()  # 使用detach()移除梯度
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的Score实现"""
        def _fit():
            X = torch.tensor(data['observations'], dtype=torch.float32).to(self.device)
            dim = X.shape[1]
            
            batch_size = min(64, len(X))
            data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
            
            score_net = ScoreNet(dim).to(self.device)
            
            print("训练Score网络...")
            self._train_score(score_net, data_loader, task_type='gmm')
            
            print("生成样本...")
            with torch.no_grad():
                x = torch.randn(n_iterations, dim).to(self.device)
                
                # 使用Langevin动力学生成样本
                for t in range(1000):
                    noise = torch.randn_like(x) * np.sqrt(2 * 0.01)
                    mu, sigma, pi = score_net(x)
                    
                    # 将参数转换为正确的形状
                    mu = mu.view(-1, 2, dim)  
                    sigma = sigma.view(-1, 2, dim, dim) 
                    sigma = torch.matmul(sigma, sigma.transpose(-2, -1))  # 确保正定性
                    
                    # 计算梯度
                    score = torch.zeros_like(x)
                    for k in range(2):
                        diff = x.unsqueeze(1) - mu[:, k:k+1]  
                        inv_sigma = torch.inverse(sigma[:, k])  
                        score_k = -torch.bmm(diff, inv_sigma).squeeze(1) 
                        score += pi[:, k:k+1] * score_k
                    
                    x = x + 0.01 * score + noise
                
                # 获取最终参数
                mu, sigma, pi = score_net(x)
                mu = mu.view(-1, 2, dim).cpu().numpy()
                sigma = sigma.view(-1, 2, dim, dim)
                sigma = torch.matmul(sigma, sigma.transpose(-2, -1)).cpu().numpy()
                pi = pi.cpu().numpy()
            
            return {
                'mu1_samples': mu[n_burnin:, 0],  
                'mu2_samples': mu[n_burnin:, 1], 
                'sigma1_samples': sigma[n_burnin:, 0],  
                'sigma2_samples': sigma[n_burnin:, 1],  
                'pi_samples': pi[n_burnin:],  
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_ssm(self, data, n_iterations=1000, n_burnin=100):
        """SSM的Score实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)  
                y = y.repeat(2, 1)  # 复制数据以确保有足够的样本进行标准化
            y, mean, std = self._standardize_data(y)
            
            batch_size = min(32, len(y))
            data_loader = torch.utils.data.DataLoader(y, batch_size=batch_size, shuffle=True)
            
            score_net = SSMScoreNet(T).to(self.device)
            
            print("训练Score网络...")
            self._train_score(score_net, data_loader, task_type='ssm')
            
            print("生成样本...")
            with torch.no_grad():
                x = torch.randn(n_iterations, T).to(self.device)
                
                # 使用Langevin动力学生成样本
                for t in range(1000):
                    noise = torch.randn_like(x) * np.sqrt(2 * 0.01)
                    states, Q, R = score_net(x)
                    
                    # 计算梯度
                    score = -states  
                    x = x + 0.01 * score + noise
                
                # 获取最终参数
                states, Q, R = score_net(x)
                
                # 先移到CPU
                states = states.cpu()
                Q = Q.cpu()
                R = R.cpu()
                
                # 反标准化
                states = states.numpy() * std.cpu().numpy() + mean.cpu().numpy()
                Q = Q.numpy()
                R = R.numpy()
            
            return {
                'state_samples': states[n_burnin:],
                'Q_samples': Q[n_burnin:],
                'R_samples': R[n_burnin:],
                'n_iterations': n_iterations,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_sparse(self, data, n_iterations=1000, n_burnin=100):
        """Sparse回归的Score实现"""
        def _fit():
            X, y = data
            n_samples, n_features = X.shape
            
            # 将数据转换为张量
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            X, X_mean, X_std = self._standardize_data(X)
            y, y_mean, y_std = self._standardize_data(y)
            
            batch_size = min(64, n_samples)
            data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
            
            score_net = SparseScoreNet(n_features).to(self.device)
            
            print("训练Score网络...")
            self._train_score(score_net, data_loader, task_type='sparse')
            
            print("生成样本...")
            with torch.no_grad():
                x = torch.randn(n_iterations, n_features).to(self.device)
                
                # 使用Langevin动力学生成样本
                for t in range(1000):
                    noise = torch.randn_like(x) * np.sqrt(2 * 0.01)
                    beta, sigma2 = score_net(x)
                    
                    # 计算梯度
                    score = -beta  # 简单的高斯先验
                    x = x + 0.01 * score + noise
                
                # 获取最终参数
                beta, sigma2 = score_net(x)
                
                # 先移到CPU
                beta = beta.cpu()
                sigma2 = sigma2.cpu()
                
                # 反标准化
                beta = beta.numpy() * X_std.cpu().numpy() + X_mean.cpu().numpy()
                sigma2 = sigma2.numpy()
            
            return {
                'beta_samples': beta[n_burnin:],
                'sigma2_samples': sigma2[n_burnin:],
                'n_iterations': n_iterations,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples

class SSMScoreNet(nn.Module):
    """SSM分数网络"""
    def __init__(self, seq_len, hidden_dim=128):
        super().__init__()
        
        # 预测状态序列
        self.state_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        
        # 预测Q和R
        self.qr_net = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), 
            nn.Softplus()  # 确保Q和R为正
        )
        
    def forward(self, x):
        # 预测所有参数
        states = self.state_net(x)  
        qr = self.qr_net(x)  
        Q, R = qr[:, 0], qr[:, 1]  
        
        return states, Q, R

class SparseScoreNet(nn.Module):
    """Sparse回归分数网络"""
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        
        # 预测beta
        self.beta_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features)
        )
        
        # 预测sigma2
        self.sigma2_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保sigma2为正
        )
        
    def forward(self, x):
        # 预测所有参数
        beta = self.beta_net(x)  
        sigma2 = self.sigma2_net(x)  
        
        return beta, sigma2