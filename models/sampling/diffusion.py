"""
优化的Diffusion采样算法实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import stats
from ..base import BaseAlgorithm
from torch.optim.lr_scheduler import CosineAnnealingLR

class TimeEmbedding(nn.Module):
    """时间编码层"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        t = t.float().view(-1, 1) / 1000.0
        return self.mlp(t)

class GMMPredictor(nn.Module):
    """GMM预测器"""
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 128
        
        self.time_embed = TimeEmbedding(hidden_dim)
        
        # 分别预测均值、协方差矩阵和混合权重
        self.mu_net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2)  # 预测两个组分的均值
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * dim * 2)  # 预测两个组分的协方差矩阵
        )
        
        self.pi_net = nn.Sequential(
            nn.Linear(dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # 预测混合权重
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=1)
        
        # 预测所有参数
        mu = self.mu_net(h)  
        sigma = self.sigma_net(h)  
        pi = self.pi_net(h)  
        
        return mu, sigma, pi

class SSMPredictor(nn.Module):
    """SSM预测器"""
    def __init__(self, seq_len):
        super().__init__()
        hidden_dim = 128
        
        self.time_embed = TimeEmbedding(hidden_dim)
        
        # 预测状态序列
        self.state_net = nn.Sequential(
            nn.Linear(seq_len + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        
        # 预测Q和R
        self.qr_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # 输出Q和R
            nn.Softplus()  # 确保Q和R为正
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)  
        
        # 预测状态序列
        h_state = torch.cat([x, t_emb], dim=1)  
        states = self.state_net(h_state)  
        
        # 预测Q和R
        qr = self.qr_net(t_emb)  
        Q, R = qr[:, 0], qr[:, 1]  #
        
        return states, Q, R

class SparsePredictor(nn.Module):
    """稀疏回归预测器"""
    def __init__(self, n_features):
        super().__init__()
        hidden_dim = 128
        
        self.time_embed = TimeEmbedding(hidden_dim)
        
        # 预测beta
        self.beta_net = nn.Sequential(
            nn.Linear(n_features + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_features)
        )
        
        # 预测sigma2，使用与beta相同的输入维度
        self.sigma2_net = nn.Sequential(
            nn.Linear(n_features + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保sigma2为正
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t) 
        
        # 预测beta
        h_beta = torch.cat([x, t_emb], dim=1)  
        beta = self.beta_net(h_beta)  
        
        # 预测sigma2
        h_sigma2 = torch.cat([x, t_emb], dim=1)  
        sigma2 = self.sigma2_net(h_sigma2)  
        
        return beta, sigma2

class DiffusionSampler(BaseAlgorithm):
    """Diffusion采样算法"""
    
    def __init__(self, random_seed=42):
        super().__init__("Diffusion", random_seed)
        
        self.n_steps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.batch_size = 32
        self.n_epochs = 100
        self.learning_rate = 1e-3
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.prepare_noise_schedule()
    
    def prepare_noise_schedule(self):
        """准备噪声调度"""
        steps = torch.linspace(0, self.n_steps, self.n_steps + 1, device=self.device)
        alpha_bar = torch.cos(((steps / self.n_steps + 0.008) / 1.008) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clip(betas, 0.0001, 0.02)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的Diffusion采样实现"""
        def _fit():
            X = data['observations']
            n_samples, dim = X.shape
            
            # 将数据转换为PyTorch张量
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            X, mean, std = self._standardize_data(X)
            
            # 初始化模型
            model = GMMPredictor(dim).to(self.device)
            
            # 训练模型
            model = self._train_model(model, X)
            
            # 生成采样
            with torch.no_grad():
                z = torch.randn(n_iterations, dim).to(self.device)
                t = torch.zeros(n_iterations).to(self.device)
                mu, sigma, pi = model(z, t)
                
                # 分离参数
                mu = mu.view(-1, 2, dim)  # [n_iterations, 2, dim]
                sigma = sigma.view(-1, 2, dim, dim)  # [n_iterations, 2, dim, dim]
                sigma = torch.matmul(sigma, sigma.transpose(-2, -1))  # 确保正定性
                
                # 先移到CPU
                mu = mu.cpu()
                sigma = sigma.cpu()
                pi = pi.cpu()
                
                # 反标准化
                mu = mu.numpy() * std.cpu().numpy() + mean.cpu().numpy()
                sigma = sigma.numpy() * (std.cpu().numpy() ** 2)
                pi = pi.numpy()
            
            return {
                'mu1_samples': mu[n_burnin:, 0],  # 第一个组分的均值
                'mu2_samples': mu[n_burnin:, 1],  # 第二个组分的均值
                'sigma1_samples': sigma[n_burnin:, 0],  # 第一个组分的协方差
                'sigma2_samples': sigma[n_burnin:, 1],  # 第二个组分的协方差
                'pi_samples': pi[n_burnin:],  # 混合权重
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_ssm(self, data, n_iterations=1000, n_burnin=100):
        """SSM的Diffusion采样实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            # 将观测数据转换为PyTorch张量
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            if len(y.shape) == 1:
                y = y.unsqueeze(0) 
                y = y.repeat(2, 1)  # 复制数据以确保有足够的样本进行标准化
            y, mean, std = self._standardize_data(y)
            
            # 初始化模型
            model = SSMPredictor(T).to(self.device)
            
            # 训练模型
            model = self._train_model(model, y)
            
            # 生成采样
            with torch.no_grad():
                z = torch.randn(n_iterations, T).to(self.device)
                t = torch.zeros(n_iterations).to(self.device)
                states, Q, R = model(z, t)
                
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
        """稀疏回归的Diffusion采样实现"""
        def _fit():
            X, y = data
            n_samples, n_features = X.shape
            
            # 将数据转换为PyTorch张量
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            X, X_mean, X_std = self._standardize_data(X)
            y, y_mean, y_std = self._standardize_data(y)
            
            # 初始化模型
            model = SparsePredictor(n_features).to(self.device)
            
            # 训练模型
            model = self._train_model(model, X)
            
            # 生成采样
            with torch.no_grad():
                z = torch.randn(n_iterations, n_features).to(self.device)
                t = torch.zeros(n_iterations).to(self.device)
                beta, sigma2 = model(z, t)
                
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
    
    def _train_model(self, model, data):
        """训练模型"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        
        for epoch in range(self.n_epochs):
            losses = []
            
            # 计算噪声
            t = torch.randint(0, self.n_steps, (data.size(0),)).to(self.device)
            noise = torch.randn_like(data)
            noisy_data = self.sqrt_alphas_cumprod[t].view(-1, 1) * data + \
                        self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
            
            # 预测噪声
            if isinstance(model, GMMPredictor):
                mu, sigma, pi = model(noisy_data, t)
                # 计算GMM的负对数似然作为损失
                mu = mu.view(-1, 2, data.shape[1])  # [batch_size, 2, dim]
                sigma = sigma.view(-1, 2, data.shape[1], data.shape[1])  # [batch_size, 2, dim, dim]
                sigma = torch.matmul(sigma, sigma.transpose(-2, -1)) + \
                       torch.eye(data.shape[1]).to(self.device) * 1e-4
                
                # 计算每个组分的负对数似然
                nll = torch.zeros(data.size(0), 2).to(self.device)
                for k in range(2):
                    diff = (data.unsqueeze(1) - mu[:, k:k+1]).transpose(1, 2)  # [batch_size, dim, 1]
                    inv_sigma = torch.inverse(sigma[:, k])  # [batch_size, dim, dim]
                    mahalanobis = torch.bmm(torch.bmm(diff.transpose(-2, -1), inv_sigma), diff)  # [batch_size, 1, 1]
                    log_det = torch.logdet(sigma[:, k])  # [batch_size]
                    nll[:, k] = 0.5 * (mahalanobis.squeeze(-1).squeeze(-1) + log_det + data.shape[1] * np.log(2 * np.pi))
                
                # 使用混合权重计算最终损失
                loss = torch.mean(-torch.logsumexp(-nll + torch.log(pi + 1e-10), dim=1))
                
            elif isinstance(model, SSMPredictor):
                states, Q, R = model(noisy_data, t)
                # 使用MSE损失，确保维度匹配
                if len(states.shape) != len(data.shape):
                    states = states.view(data.shape)
                loss = F.mse_loss(states, data)
                
            elif isinstance(model, SparsePredictor):
                beta, sigma2 = model(noisy_data, t)
                # 使用MSE损失
                loss = F.mse_loss(beta, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {np.mean(losses):.4f}")
        
        return model
    
    def _reverse_diffusion(self, x_t, t, model):
        """反向扩散过程"""
        predicted_noise = model(x_t, t)
        
        if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
            predicted_noise = torch.randn_like(x_t) * 0.01
        
        alpha = self.alphas[t].view(-1, 1)
        alpha_cumprod = self.alphas_cumprod[t].view(-1, 1)
        beta = self.betas[t].view(-1, 1)
        
        coef1 = torch.sqrt(1. / alpha)
        coef2 = beta / torch.sqrt(1. - alpha_cumprod)
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        if t[0] > 0:
            noise_scale = torch.sqrt(beta) * (1. - t.float() / self.n_steps).view(-1, 1)
            noise = torch.randn_like(x_t) * noise_scale
            x_next = mean + noise
        else:
            x_next = mean
        
        return torch.clamp(x_next, -5., 5.)
    
    def _sample(self, model, n_samples, dim):
        """采样过程"""
        with torch.no_grad():
            x = torch.randn(n_samples, dim).to(self.device) * 0.1
            
            for t in reversed(range(self.n_steps)):
                batch_size = min(500, n_samples)
                num_batches = (n_samples + batch_size - 1) // batch_size
                
                new_x = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    x_batch = x[start_idx:end_idx]
                    
                    t_batch = torch.full((end_idx - start_idx,), t, device=self.device, dtype=torch.long)
                    x_batch = self._reverse_diffusion(x_batch, t_batch, model)
                    new_x.append(x_batch)
                
                x = torch.cat(new_x, dim=0)
                
                if (t + 1) % 100 == 0:
                    print(f"采样进度: {self.n_steps - t}/{self.n_steps}")
            
            return x.cpu().numpy()
    
    def _standardize_data(self, data):
        """标准化数据"""
        mean = data.mean(0)
        std = data.std(0) + 1e-5
        return (data - mean) / std, mean, std
