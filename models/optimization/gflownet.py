"""
使用GFlowNet的贝叶斯采样实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseAlgorithm
from ..initialization import initialize_gmm_params, initialize_ssm_params, initialize_sparse_params
import traceback

class FlowNetwork(nn.Module):
    """GFlowNet的基础网络结构"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 前向策略网络 -用于选择动作
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 后向策略网络 -用于估计状态值
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_policy(self, state):
        """计算前向策略"""
        logits = self.policy_net(state)
        return F.log_softmax(logits, dim=-1)
    
    def backward_policy(self, state):
        """计算后向策略"""
        return self.value_net(state)

class GFlowNetOptimizer(BaseAlgorithm):
    """GFlowNet优化器"""
    
    def __init__(self, random_seed=42):
        super().__init__("GFlowNet", random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        
        # 超参数设置
        self.learning_rate = 0.001
        self.batch_size = 32
        self.n_epochs = 50
        self.step_size = 0.1
        self.max_steps = 20
        
    def _create_model(self, state_dim, action_dim):
        """创建并初始化模型"""
        model = FlowNetwork(state_dim, action_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer
    
    def _compute_loss(self, model, states, actions, next_states, rewards):
        """计算流匹配损失"""
        batch_size = states.shape[0]
        
        # 计算前向概率
        forward_logits = model.forward_policy(states)  # [B, A]
        log_pf = torch.gather(forward_logits, 1, actions.unsqueeze(1)).squeeze(1)  # [B]
        
        # 计算后向概率
        log_pb = model.backward_policy(next_states).squeeze(1)  # [B]
        
        # 计算初始状态的值
        log_Z = model.backward_policy(torch.zeros_like(states[0]).unsqueeze(0)).squeeze()  # [1]
        log_Z = log_Z.expand(batch_size)  # [B]
        
        # 计算流匹配损失
        loss = (log_Z + log_pf - torch.log(rewards + 1e-10) - log_pb).pow(2).mean()
        return loss
    
    def _step(self, state, action):
        """执行动作获得下一状态"""
        next_state = state.clone()
        param_idx = action // 2
        direction = action % 2
        step = self.step_size if direction == 0 else -self.step_size
        
        if param_idx < len(state) - 2:  # 普通参数
            next_state[param_idx] += step
        else:  # 特殊参数(如混合权重)
            pi_idx = param_idx - (len(state) - 2)
            next_state[-2:] = F.softmax(next_state[-2:] + step * F.one_hot(torch.tensor(pi_idx), 2).to(self.device), dim=-1)
        
        return next_state
    
    def _collect_trajectory(self, model, initial_states):
        """批量收集轨迹"""
        batch_size = initial_states.shape[0]
        states = [initial_states]
        actions = []
        rewards = []
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_steps):
            if torch.all(is_done):
                break
                
            # 获取动作
            with torch.no_grad():
                logits = model.forward_policy(states[-1])
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            
            # 执行动作
            next_state = self._step_batch(states[-1], action)
            reward = self._compute_reward_batch(next_state)
            
            # 保存轨迹
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            
            # 更新终止状态
            is_done = is_done | torch.any(torch.abs(next_state) > 100, dim=1)
        
        # 将列表转换为张量，并调整维度顺序为 [T, B, ...]
        states = torch.stack(states)  
        actions = torch.stack(actions)  
        rewards = torch.stack(rewards)  
        
        return states, actions, rewards

    def _step_batch(self, states, actions):
        """批量执行动作"""
        next_states = states.clone()
        batch_size = states.shape[0]
        
        # 计算参数索引和方向
        param_indices = actions // 2
        directions = actions % 2
        steps = torch.where(
            directions == 0,
            torch.ones(batch_size, device=self.device) * self.step_size,
            torch.ones(batch_size, device=self.device) * -self.step_size
        )
        
        # 更新普通参数
        regular_mask = param_indices < (states.shape[1] - 2)
        if regular_mask.any():
            batch_indices = torch.arange(batch_size, device=self.device)[regular_mask]
            next_states[batch_indices, param_indices[regular_mask]] += steps[regular_mask]
        
        # 更新特殊参数(如混合权重)
        special_mask = ~regular_mask
        if special_mask.any():
            batch_indices = torch.arange(batch_size, device=self.device)[special_mask]
            pi_indices = param_indices[special_mask] - (states.shape[1] - 2)
            logits = next_states[batch_indices, -2:]
            one_hot = F.one_hot(pi_indices, 2).to(self.device)
            next_states[batch_indices, -2:] = F.softmax(logits + steps[special_mask].unsqueeze(1) * one_hot, dim=-1)
        
        return next_states

    def _compute_reward_batch(self, states):
        """批量计算奖励"""
        if hasattr(self, 'data'):
            return self._compute_gmm_reward_batch(states)
        elif hasattr(self, 'ssm_data'):
            return self._compute_ssm_reward_batch(states)
        elif hasattr(self, 'sparse_X'):
            return self._compute_sparse_reward_batch(states)
        else:
            raise ValueError("未知的场景类型")

    def _compute_gmm_reward_batch(self, states):
        """批量计算GMM场景的奖励"""
        try:
            batch_size = states.shape[0]
            dim = self.data.shape[1]
            
            # 批量解析参数
            mu1 = states[:, :dim]  # [B, D]
            mu2 = states[:, dim:2*dim]  # [B, D]
            pi = F.softmax(states[:, 2*dim:2*dim+2], dim=1)  # [B, 2]
            
            # 直接重塑协方差矩阵
            sigma1_start = 2*dim+2
            sigma1_end = sigma1_start + dim*dim
            sigma2_end = sigma1_end + dim*dim
            
            sigma1 = states[:, sigma1_start:sigma1_end].reshape(batch_size, dim, dim)
            sigma2 = states[:, sigma1_end:sigma2_end].reshape(batch_size, dim, dim)
            
            # 批量确保协方差矩阵正定
            eye = torch.eye(dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
            sigma1 = sigma1 @ sigma1.transpose(1, 2) + eye * 1e-6
            sigma2 = sigma2 @ sigma2.transpose(1, 2) + eye * 1e-6
            
            # 批量计算似然
            ll = torch.zeros(batch_size, device=self.device)
            
            # 计算第一个组分的概率
            diff1 = self.data.unsqueeze(0) - mu1.unsqueeze(1)  # [B, N, D]
            
            try:
                # 计算马氏距离
                sigma1_inv = torch.inverse(sigma1)  # [B, D, D]
                
                # 计算马氏距离
                mahalanobis1 = torch.sum(
                    diff1 * (diff1 @ sigma1_inv),
                    dim=-1
                )  # [B, N]
                
                log_det1 = torch.logdet(sigma1)  # [B]
                log_p1 = -0.5 * (dim * np.log(2 * np.pi) + log_det1.unsqueeze(1) + mahalanobis1)  # [B, N]
                p1 = pi[:, 0:1] * torch.exp(log_p1)  # [B, N]
                
            except RuntimeError as e:
                print(f"Error in first component: {str(e)}")
                return torch.ones(batch_size, device=self.device) * 1e-10
            
            # 计算第二个组分的概率
            diff2 = self.data.unsqueeze(0) - mu2.unsqueeze(1)  # [B, N, D]
            try:
                # 计算马氏距离
                sigma2_inv = torch.inverse(sigma2)  # [B, D, D]
                mahalanobis2 = torch.sum(
                    diff2 * (diff2 @ sigma2_inv),
                    dim=-1
                )  # [B, N]
                
                log_det2 = torch.logdet(sigma2)  # [B]
                log_p2 = -0.5 * (dim * np.log(2 * np.pi) + log_det2.unsqueeze(1) + mahalanobis2)  # [B, N]
                p2 = pi[:, 1:2] * torch.exp(log_p2)  # [B, N]
                
            except RuntimeError as e:
                print(f"Error in second component: {str(e)}")
                return torch.ones(batch_size, device=self.device) * 1e-10
            
            # 数值稳定的对数计算
            max_val = torch.maximum(p1, p2)
            ll = torch.sum(
                torch.log(max_val + 1e-10) +
                torch.log1p(torch.exp(-torch.abs(p1 - p2)) + 1e-10),
                dim=1
            )
            
            # 批量计算先验
            prior = (
                -0.5 * torch.sum(mu1**2, dim=1)
                -0.5 * torch.sum(mu2**2, dim=1)
                -0.5 * torch.sum(log_det1)
                -0.5 * torch.sum(log_det2)
                +torch.sum(torch.log(pi + 1e-10), dim=1)
            )
            
            reward = torch.exp((ll + prior) / len(self.data))
            return torch.where(
                torch.isfinite(reward),
                reward,
                torch.tensor(1e-10, device=self.device)
            )
            
        except Exception as e:
            print(f"Warning: Batch GMM reward computation failed: {str(e)}")
            return torch.ones(batch_size, device=self.device) * 1e-10

    def _compute_ssm_reward_batch(self, states):
        """批量计算SSM场景的奖励"""
        try:
            batch_size = states.shape[0]
            T = len(self.ssm_data)
            
            # 批量解析参数
            x = states[:, :T]  # [B, T]
            Q = torch.abs(states[:, -2])  # [B]
            R = torch.abs(states[:, -1])  # [B]
            
            # 批量计算状态转移似然
            ll = torch.zeros(batch_size, device=self.device)
            t_range = torch.arange(1, T, device=self.device)
            
            # 状态转移
            x_prev = x[:, :-1]  # [B, T-1]
            pred = (0.5 * x_prev + 
                   25 * x_prev / (1 + x_prev**2) + 
                   8 * torch.cos(1.2 * t_range))  # [B, T-1]
            
            ll += -0.5 * (T-1) * torch.log(Q + 1e-10)
            ll += -0.5 * torch.sum((x[:, 1:] - pred)**2 / (Q.unsqueeze(1) + 1e-10), dim=1)
            
            # 观测似然
            ll += -0.5 * T * torch.log(R + 1e-10)
            ll += -0.5 * torch.sum((self.ssm_data - x**2/20)**2 / (R.unsqueeze(1) + 1e-10), dim=1)
            
            # 批量计算先验
            prior = (
                -0.5 * x[:, 0]**2
                -torch.log(Q + 1e-10) - 1/(Q + 1e-10)
                -torch.log(R + 1e-10) - 1/(R + 1e-10)
            )
            
            reward = torch.exp((ll + prior) / T)
            return torch.where(
                torch.isfinite(reward),
                reward,
                torch.tensor(1e-10, device=self.device)
            )
            
        except Exception as e:
            print(f"Warning: Batch SSM reward computation failed: {str(e)}")
            return torch.ones(batch_size, device=self.device) * 1e-10

    def _compute_sparse_reward_batch(self, states):
        """批量计算稀疏回归场景的奖励"""
        try:
            batch_size = states.shape[0]
            
            # 批量解析参数
            beta = states[:, :-1]  # [B, D]
            sigma2 = torch.exp(states[:, -1])  # [B]
            
            # 批量计算似然
            pred = self.sparse_X @ beta.t()  # [N, B]
            resid = (self.sparse_y.unsqueeze(1) - pred)  # [N, B]
            ll = -0.5 * len(self.sparse_y) * torch.log(sigma2 + 1e-10)
            ll += -0.5 * torch.sum(resid**2, dim=0) / (sigma2 + 1e-10)
            
            # 批量计算先验
            prior = -0.5 * torch.sum(beta**2, dim=1)
            prior += -torch.log(sigma2 + 1e-10) - 1/(sigma2 + 1e-10)
            
            reward = torch.exp((ll + prior) / len(self.sparse_y))
            return torch.where(
                torch.isfinite(reward),
                reward,
                torch.tensor(1e-10, device=self.device)
            )
            
        except Exception as e:
            print(f"Warning: Batch sparse reward computation failed: {str(e)}")
            return torch.ones(batch_size, device=self.device) * 1e-10

    def _train_model(self, model, optimizer, initial_state):
        """训练模型"""
        for epoch in range(self.n_epochs):
            # 创建批量初始状态
            initial_states = initial_state.unsqueeze(0).expand(self.batch_size, -1)
            
            # 收集轨迹
            states, actions, rewards = self._collect_trajectory(model, initial_states)
            
            # 确保维度正确
            n_steps = states.shape[0] - 1  # 轨迹长度减1
            batch_size = states.shape[1]  # 批量大小
            state_dim = states.shape[2]  # 状态维度
            
            states_flat = states[:-1].reshape(-1, state_dim)  # [B*T, D]
            next_states_flat = states[1:].reshape(-1, state_dim)  # [B*T, D]
            actions_flat = actions.reshape(-1)  # [B*T]
            rewards_flat = rewards.reshape(-1)  # [B*T]
            
            # 计算损失并更新
            optimizer.zero_grad()
            loss = self._compute_loss(model, states_flat, actions_flat, next_states_flat, rewards_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    def _generate_samples(self, model, n_samples, initial_state):
        """批量生成样本"""
        samples = []
        batch_size = min(256, n_samples)
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                if (i + 1) % 1000 == 0:
                    print(f"已生成 {i+1}/{n_samples} 个样本")
                
                current_batch_size = min(batch_size, n_samples - i)
                current_states = initial_state.unsqueeze(0).expand(current_batch_size, -1)
                
                for _ in range(self.max_steps):
                    logits = model.forward_policy(current_states)
                    actions = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                    current_states = self._step_batch(current_states, actions)
                    
                    if torch.any(torch.abs(current_states) > 100):
                        break
                
                samples.append(current_states.cpu().numpy())
        
        return np.concatenate(samples)
    
    def fit_gmm(self, data, n_iterations=1000, n_burnin=100):
        """GMM的GFlowNet实现"""
        def _fit():
            # 准备数据
            X = torch.tensor(data['observations'], dtype=torch.float32)
            self.data = X.to(self.device)
            dim = X.shape[1]
            
            # 设置状态和动作维度
            state_dim = 2*dim + 2 + 2*dim*dim  # [μ1, μ2, π, Σ1, Σ2]
            action_dim = 4*dim + 4 + 4*dim*dim  # 每个参数可增可减
            
            # 创建模型
            model, optimizer = self._create_model(state_dim, action_dim)
            
            # 初始化状态
            initial_state = torch.zeros(state_dim, device=self.device)
            initial_state[2*dim:2*dim+2] = torch.tensor([0.5, 0.5], device=self.device)
            sigma_idx = 2*dim + 2
            for i in range(dim):
                initial_state[sigma_idx + i*dim + i] = 1.0
                initial_state[sigma_idx + dim*dim + i*dim + i] = 1.0
            
            # 训练模型
            print("训练GFlowNet...")
            self._train_model(model, optimizer, initial_state)
            
            # 生成样本
            print("生成样本...")
            samples = self._generate_samples(model, n_iterations, initial_state)
            
            # 解包结果
            mu1_samples = samples[:, :dim]
            mu2_samples = samples[:, dim:2*dim]
            pi_samples = samples[:, 2*dim:2*dim+2]
            sigma1_samples = samples[:, 2*dim+2:2*dim+2+dim*dim].reshape(-1, dim, dim)
            sigma2_samples = samples[:, 2*dim+2+dim*dim:].reshape(-1, dim, dim)
            
            # 后处理样本
            for i in range(len(samples)):
                sigma1_samples[i] = sigma1_samples[i] @ sigma1_samples[i].T + np.eye(dim) * 1e-6
                sigma2_samples[i] = sigma2_samples[i] @ sigma2_samples[i].T + np.eye(dim) * 1e-6
            
            pi_samples = np.abs(pi_samples)
            pi_samples = pi_samples / pi_samples.sum(axis=1, keepdims=True)
            
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
        """SSM的GFlowNet实现"""
        def _fit():
            # 准备数据
            y = data['observations']
            T = len(y)
            self.ssm_data = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # 设置状态和动作维度
            state_dim = T + 2  # [x(T), Q, R]
            action_dim = 2*(T + 2)  # 可变
            
            # 创建模型
            model, optimizer = self._create_model(state_dim, action_dim)
            
            # 初始化状态
            params = initialize_ssm_params(T, self.random_seed)
            initial_state = torch.zeros(state_dim, device=self.device)
            initial_state[:T] = torch.tensor(params['x'], device=self.device)
            initial_state[-2:] = torch.tensor([params['Q'], params['R']], device=self.device)
            
            # 训练模型
            print("训练GFlowNet...")
            self._train_model(model, optimizer, initial_state)
            
            # 生成样本
            print("生成样本...")
            samples = self._generate_samples(model, n_iterations, initial_state)
            
            # 解包结果
            x_samples = samples[:, :T]
            Q_samples = np.abs(samples[:, -2])
            R_samples = np.abs(samples[:, -1])
            
            return {
                'state_samples': x_samples[n_burnin:],
                'Q_samples': Q_samples[n_burnin:],
                'R_samples': R_samples[n_burnin:],
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples
    
    def fit_sparse(self, data, n_iterations=1000, n_burnin=100):
        """稀疏回归的GFlowNet实现"""
        def _fit():
            # 准备数据
            X, y = data
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            
            self.sparse_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            self.sparse_y = torch.tensor(y, dtype=torch.float32).to(self.device)
            n_features = X.shape[1]
            
            # 设置状态和动作维度
            state_dim = n_features + 1  # [β, σ²]
            action_dim = 2*(n_features + 1)  # 每个参数可增可减
            
            # 创建模型
            model, optimizer = self._create_model(state_dim, action_dim)
            
            # 初始化状态
            params = initialize_sparse_params(n_features, self.random_seed)
            initial_state = torch.zeros(state_dim, device=self.device)
            initial_state[:-1] = torch.tensor(params['beta'], device=self.device)
            initial_state[-1] = torch.tensor(np.log(params['sigma2']), device=self.device)
            
            # 训练模型
            print("训练GFlowNet...")
            self._train_model(model, optimizer, initial_state)
            
            # 生成样本
            print("生成样本...")
            samples = self._generate_samples(model, n_iterations, initial_state)
            
            # 解包结果
            beta_samples = samples[:, :-1]
            sigma2_samples = np.exp(samples[:, -1])
            
            # 计算后验均值
            beta_mean = np.mean(beta_samples[n_burnin:], axis=0)
            
            return {
                'beta_samples': beta_samples[n_burnin:],
                'beta_mean': beta_mean,
                'sigma2_samples': sigma2_samples[n_burnin:],
                'n_iterations': n_iterations - n_burnin,
                'n_burnin': n_burnin
            }
        
        samples, runtime = self._time_execution(_fit)
        samples['runtime'] = runtime
        return samples 