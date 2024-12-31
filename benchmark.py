import numpy as np
try:
    from scenarios.gmm import GMMScenario
    from scenarios.ssm import SSMScenario
    from scenarios.sparse import SparseRegressionScenario
    from models.sampling.gibbs import GibbsSampler
    from models.sampling.abc import ABCSampler
    from models.sampling.diffusion import DiffusionSampler
    from models.optimization.vi import VISampler
    from models.optimization.gan import GANOptimizer
    from models.optimization.gflownet import GFlowNetOptimizer
    from models.optimization.score import ScoreOptimizer
    from models.numerical.inla import INLAOptimizer
    from evaluation.evaluator import Evaluator
    from models.sampling.hmc_stan import HMCStanSampler
    from models.sampling.abc import ABCSampler
    from models.sampling.diffusion import DiffusionSampler
except ImportError as e:
    print(f"导入错误: {e}")
    raise

class Benchmark:
    """基准测试类"""
    
    def __init__(self, random_seed=42):
        try:
            # 初始化捏
            self.scenarios = {
                'gmm': GMMScenario(random_seed),
                'ssm': SSMScenario(random_seed),
                'sparse': SparseRegressionScenario(random_seed)
            }
            self.algorithms = {
                'gibbs': GibbsSampler(random_seed),
                'abc': ABCSampler(random_seed),
                'diffusion': DiffusionSampler(random_seed),
                'vi': VISampler(random_seed),
                'gan': GANOptimizer(random_seed),
                'gflownet': GFlowNetOptimizer(random_seed),
                'score': ScoreOptimizer(random_seed),
                'inla': INLAOptimizer(random_seed),
                'hmc_stan': HMCStanSampler(random_seed),
            }
            self.evaluator = Evaluator()
            
            print("成功初始化所有场景和算法")
        except Exception as e:
            print(f"初始化错误: {e}")
            raise
    
    def run_single_test(self, scenario_name, algorithm, n_iterations=1000, n_burnin=100, data=None):
        """运行单个测试"""
        print(f"\n测试启动！！！！！！: {scenario_name} - {algorithm.name}")
        
        # 生成数据捏
        if data is None:
            print("生成数据...阿巴阿巴")
            # 为VI算法在sparse场景下特别设置样本量
            if scenario_name == 'sparse' and algorithm.name == 'VI':
                data = self.scenarios[scenario_name].generate_data(n_samples=10000)
            else:
                data = self.scenarios[scenario_name].generate_data()
            true_params = self.scenarios[scenario_name].get_true_params()
        test_data = self.scenarios[scenario_name].generate_test_data()
        
        # 运行算法
        print(f"\nrun！ {algorithm.name} 算法...")
        if scenario_name == 'gmm':
            samples = algorithm.fit_gmm(data, n_iterations, n_burnin)
        elif scenario_name == 'ssm':
            samples = algorithm.fit_ssm(data, n_iterations, n_burnin)
        else:  
            samples = algorithm.fit_sparse(data, n_iterations, n_burnin)
        
        # 评估结果
        print("\n评估指标捏:")
        if scenario_name == 'gmm':
            metrics = self.evaluator.evaluate_gmm(samples, true_params, (data, test_data))
        elif scenario_name == 'ssm':
            metrics = self.evaluator.evaluate_ssm(samples, true_params, (data, test_data))
        else:  # sparse
            metrics = self.evaluator.evaluate_sparse(samples, true_params, (data, test_data))
        
        # 打印效率指标
        print("A. 效率指标:")
        print(f"  总运行时间: {metrics['wall_time']:.2f}秒")
        print(f"  迭代次数: {metrics['iterations']}")
        print(f"  单位时间样本数: {metrics['samples_per_second']:.2f}")
        
        # 打印模型特定指标
        print("\nB. 模型特定指标:")
        if scenario_name == 'gmm':
            print("  训练集评估:")
            print(f"    均值误差: {metrics['mu_error']:.4f}")
            print(f"    协方差误差: {metrics['sigma_error']:.4f}")
            print(f"    混合权重误差: {metrics['pi_error']:.4f}")
            print(f"    对数似然值: {metrics['train_log_likelihood']:.4f}")
            print("  测试集评估:")
            print(f"    测试集对数似然值: {metrics['test_log_likelihood']:.4f}")
            if algorithm.name != 'vi':
                print("  有效样本量:")
                print(f"    mu1: {metrics['ess_mu1']:.2f}")
                print(f"    mu2: {metrics['ess_mu2']:.2f}")
            if algorithm.name == 'vi':
                print(f"  ELBO: {metrics.get('elbo', 'N/A')}")
        
        elif scenario_name == 'ssm':
            print("  训练集评估:")
            print(f"    状态估计RMSE: {metrics['train_state_rmse']:.4f}")
            print(f"    预测RMSE: {metrics['train_prediction_rmse']:.4f}")
            print(f"    Q参数误差: {metrics['Q_error']:.4f}")
            print(f"    R参数误差: {metrics['R_error']:.4f}")
            print("  测试集评估:")
            print(f"    测试集状态估计RMSE: {metrics['test_state_rmse']:.4f}")
            print(f"    测试集预测RMSE: {metrics['test_prediction_rmse']:.4f}")
            if algorithm.name == 'vi':
                print(f"  ELBO: {metrics.get('elbo', 'N/A')}")
        
        else:  # sparse
            print("  变量选择性能:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print("  预测性能:")
            print(f"    训练集RMSE: {metrics['train_prediction_rmse']:.4f}")
            print(f"    测试集RMSE: {metrics['test_prediction_rmse']:.4f}")
            print(f"  稀疏度: {metrics['sparsity']:.4f}")
            if algorithm.name == 'vi':
                print(f"  ELBO: {metrics.get('elbo', 'N/A')}")
        
        return metrics

def init_algorithms():
    """初始化所有算法"""
    algorithms = {
        'hmc_stan': HMCStanSampler(),  
        'abc': ABCSampler(),
        'diffusion': DiffusionSampler(),
    }
    return algorithms