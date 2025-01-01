"""
GAN优化算法实现 - 基于分布-分布-值范式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAlgorithm


class Generator(nn.Module):
    """生成器网络"""
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super().__init__()
        
        # 分别生成均值、协方差矩阵和混合权重
        self.mu_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim * 2)  # 生成两个组分的均值
        )
        
        self.sigma_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim * output_dim * 2)  # 生成两个组分的协方差矩阵
        )
        
        self.pi_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),  # 生成混合权重
            nn.Softmax(dim=1)
        )
        
    def forward(self, z):
        # 生成所有参数
        mu = self.mu_net(z)  
        sigma = self.sigma_net(z)  
        pi = self.pi_net(z)  
        
        return mu, sigma, pi

class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class SSMGenerator(nn.Module):
    """SSM生成器网络"""
    def __init__(self, latent_dim, seq_len, hidden_dim=128):
        super().__init__()
        
        # 生成状态序列
        self.state_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, seq_len)
        )
        
        # 生成Q和R
        self.qr_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),  # 生成Q和R
            nn.Softplus()  # 确保Q和R为正
        )
        
    def forward(self, z):
        # 生成所有参数
        states = self.state_net(z)  
        qr = self.qr_net(z)  
        Q, R = qr[:, 0], qr[:, 1]  
        
        return states, Q, R

class SparseGenerator(nn.Module):
    """Sparse回归生成器网络"""
    def __init__(self, latent_dim, n_features, hidden_dim=128):
        super().__init__()
        
        # 生成beta
        self.beta_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_features)
        )
        
        # 生成sigma2
        self.sigma2_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 确保sigma2为正
        )
        
    def forward(self, z):
        # 生成所有参数
        beta = self.beta_net(z)  
        sigma2 = self.sigma2_net(z)  
        
        return beta, sigma2

class GANOptimizer(BaseAlgorithm):
    def __init__(self, random_seed=42):
        super().__init__("GAN", random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.empty_cache()
    
    def _standardize_data(self, data):
        """标准化数据"""
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True) + 1e-6  # 添加小的常数避免除零
        standardized_data = (data - mean) / std
        return standardized_data, mean, std
    
    def _train_gan(self, generator, discriminator, data_loader, n_epochs=200, task_type='gmm'):
        """训练GAN"""
        if task_type == 'gmm':
            lr = 1e-4
            n_critic = 1
        elif task_type == 'ssm':
            lr = 2e-5
            n_critic = 5
        else:  # sparse
            lr = 5e-5
            n_critic = 3
            
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        for epoch in range(n_epochs):
            g_losses = []
            d_losses = []
            
            for batch_data in data_loader:
                # 确保batch是正确的形状
                if task_type == 'ssm':
                    batch = batch_data[0]  
                else:
                    batch = batch_data
                
                batch_size = batch.size(0)
                
                # 训练判别器
                for _ in range(n_critic):
                    d_optimizer.zero_grad()
                    
                    # 根据任务类型确定潜变量维度
                    if task_type == 'gmm':
                        latent_dim = generator.mu_net[0].in_features
                    elif task_type == 'ssm':
                        latent_dim = generator.state_net[0].in_features
                    else:  # sparse
                        latent_dim = generator.beta_net[0].in_features
                    
                    z = torch.randn(batch_size, latent_dim).to(self.device)
                    
                    # 根据任务类型生成样本
                    if task_type == 'gmm':
                        mu, sigma, pi = generator(z)
                        
                        # 从生成的分布中采样
                        mu = mu.view(-1, 2, batch.shape[1])  
                        sigma = sigma.view(-1, 2, batch.shape[1], batch.shape[1])  
                        
                        # 确保协方差矩阵正定（添加小的对角线噪声）
                        diag_noise = torch.eye(batch.shape[1]).unsqueeze(0).unsqueeze(0).to(self.device) * 1e-4
                        sigma = torch.matmul(sigma, sigma.transpose(-2, -1)) + diag_noise  
                        
                        # 从每个高斯分量中采样
                        fake_samples = []
                        for k in range(2):
                            eps = torch.randn(batch_size, batch.shape[1]).to(self.device)
                            try:
                                L = torch.linalg.cholesky(sigma[:, k])  # Cholesky分解
                                component_samples = mu[:, k] + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
                                fake_samples.append(component_samples)
                            except:
                                # 如果Cholesky分解失败，使用更简单的采样方法
                                component_samples = mu[:, k] + eps * torch.sqrt(torch.diagonal(sigma[:, k], dim1=-2, dim2=-1))
                                fake_samples.append(component_samples)
                        
                        # 根据混合权重选择样本
                        probs = torch.multinomial(pi, batch_size, replacement=True)
                        fake_data = torch.where(
                            probs.unsqueeze(-1) == 0,
                            fake_samples[0],
                            fake_samples[1]
                        )
                        
                    elif task_type == 'ssm':
                        states, Q, R = generator(z)
                        fake_data = states  # 直接使用生成的状态序列
                        
                    else:  # sparse
                        beta, sigma2 = generator(z)
                        fake_data = beta  # 直接使用生成的beta
                    
                    # 添加梯度裁剪以防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    
                    real_scores = discriminator(batch)
                    fake_scores = discriminator(fake_data.detach())
                    
                    # Wasserstein损失
                    d_loss = -torch.mean(real_scores) + torch.mean(fake_scores)
                    
                    # 梯度惩罚
                    gp = self._gradient_penalty(discriminator, batch, fake_data.detach())
                    d_loss = d_loss + 10.0 * gp
                    
                    if torch.isnan(d_loss):
                        print("Warning: D loss is nan, skipping batch")
                        continue
                    
                    d_loss.backward()
                    d_optimizer.step()
                
                # 训练生成器
                g_optimizer.zero_grad()
                fake_scores = discriminator(fake_data)
                g_loss = -torch.mean(fake_scores)
                
                if torch.isnan(g_loss):
                    print("Warning: G loss is nan, skipping batch")
                    continue
                
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, G Loss: {np.mean(g_losses):.4f}, D Loss: {np.mean(d_losses):.4f}")
    
    def _gradient_penalty(self, discriminator, real_data, fake_data):
        """梯度惩罚项"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        d_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的GAN实现"""
        def _fit():
            X = torch.tensor(data['observations'], dtype=torch.float32).to(self.device)
            dim = X.shape[1]
            
            batch_size = min(64, len(X))
            data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True)
            
            latent_dim = dim * 2
            generator = Generator(latent_dim, dim).to(self.device)
            discriminator = Discriminator(dim).to(self.device)
            
            print("训练GAN...")
            self._train_gan(generator, discriminator, data_loader, task_type='gmm')
            
            print("生成样本...")
            generator.eval()
            with torch.no_grad():
                z = torch.randn(n_iterations, latent_dim).to(self.device)
                mu, sigma, pi = generator(z)
                
                # 将参数转换为正确的形状
                mu = mu.view(-1, 2, dim)  
                sigma = sigma.view(-1, 2, dim, dim)  
                
                # 确保协方差矩阵正定
                diag_noise = torch.eye(dim).unsqueeze(0).unsqueeze(0).to(self.device) * 1e-4
                sigma = torch.matmul(sigma, sigma.transpose(-2, -1)) + diag_noise
                
                # 转换为numpy数组
                mu = mu.cpu().numpy()
                sigma = sigma.cpu().numpy()
                pi = pi.cpu().numpy()
            
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
        """SSM的GAN实现"""
        def _fit():
            y = data['observations']
            T = len(y)
            
            # 将观测数据转换为张量并确保维度正确
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)  
                y = y.repeat(2, 1)  # 复制数据以确保有足够的样本进行标准化
            y, mean, std = self._standardize_data(y)
            
            dataset = torch.utils.data.TensorDataset(y)
            batch_size = min(32, max(2, len(y)))  # 确保batch_size至少为2
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            latent_dim = T * 2
            generator = SSMGenerator(latent_dim, T).to(self.device)
            discriminator = Discriminator(T).to(self.device)
            
            print("训练GAN...")
            self._train_gan(generator, discriminator, data_loader, task_type='ssm')
            
            print("生成样本...")
            generator.eval()
            with torch.no_grad():
                z = torch.randn(n_iterations, latent_dim).to(self.device)
                states, Q, R = generator(z)
                
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
        """Sparse回归的GAN实现"""
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
            
            latent_dim = n_features * 2
            generator = SparseGenerator(latent_dim, n_features).to(self.device)
            discriminator = Discriminator(n_features).to(self.device)
            
            print("训练GAN...")
            self._train_gan(generator, discriminator, data_loader, task_type='sparse')
            
            print("生成样本...")
            generator.eval()
            with torch.no_grad():
                z = torch.randn(n_iterations, latent_dim).to(self.device)
                beta, sigma2 = generator(z)
                
                # 先移到CPU再转换为numpy
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