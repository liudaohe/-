from benchmark import Benchmark
import numpy as np
from config import get_config

def run_benchmark(seed, args):
    """运行单次基准测试"""
    print(f"\n=== 使用随机种子 {seed} 运行测试 ===")
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 创建基准测试实例
    benchmark = Benchmark()
    
    # 测试选定的算法
    results = {}
    
    for algorithm in args.algorithms:
        print(f"\n=== 测试算法: {algorithm} ===")
        algorithm_results = {}
        
        for scenario in args.scenarios:
            print(f"\n运行测试: {scenario} - {algorithm}")
            algorithm_results[scenario] = benchmark.run_single_test(
                scenario_name=scenario,
                algorithm=benchmark.algorithms[algorithm],
                n_iterations=args.n_iterations,
                n_burnin=args.n_burnin
            )
        
        results[algorithm] = algorithm_results
    
    return results

def print_results(results, seed):
    """打印单次运行的结果"""
    print(f"\n=== 随机种子 {seed} 的结果汇总 ===\n")
    
    for algorithm_name, results in results.items():
        print(f"\n{algorithm_name} 算法结果:\n")
        
        for scenario, metrics in results.items():
            print(f"{scenario} 场景:")
            print(f"运行时间: {metrics['wall_time']:.2f}秒")
            print(f"每秒样本数: {metrics['samples_per_second']:.2f}")
            
            if scenario == 'gmm':
                print(f"均值误差: {metrics['mu_error']:.4f}")
                print(f"协方差误差: {metrics['sigma_error']:.4f}")
                print(f"混合权重误差: {metrics['pi_error']:.4f}")
                print("有效样本量:")
                print(f"  mu1: {metrics['ess_mu1']:.2f}")
                print(f"  mu2: {metrics['ess_mu2']:.2f}")
                print(f"混合权重误差: {metrics['pi_error']:.4f}")
                print(f"    测试集对数似然值: {metrics['test_log_likelihood']:.4f}")
                print(f"    对数似然值: {metrics['train_log_likelihood']:.4f}")
                
            elif scenario == 'ssm':
                print(f"训练集状态RMSE: {metrics['train_state_rmse']:.4f}")
                print(f"训练集预测RMSE: {metrics['train_prediction_rmse']:.4f}")
                print(f"测试集状态RMSE: {metrics['test_state_rmse']:.4f}")
                print(f"测试集预测RMSE: {metrics['test_prediction_rmse']:.4f}")
                print(f"Q相对误差: {metrics['Q_error']:.4f}")
                print(f"R相对误差: {metrics['R_error']:.4f}")
                
            elif scenario == 'sparse':
                print(f"精确率: {metrics['precision']:.4f}")
                print(f"召回率: {metrics['recall']:.4f}")
                print(f"F1分数: {metrics['f1_score']:.4f}")
                print(f"训练集RMSE: {metrics['train_prediction_rmse']:.4f}")
                print(f"测试集RMSE: {metrics['test_prediction_rmse']:.4f}")
                print(f"稀疏度: {metrics['sparsity']:.4f}")
            
            print()

def calculate_statistics(all_results):
    """计算多次运行的统计结果"""
    print("\n=== 多次运行的统计结果 ===\n")
    
    # 用于存储平均值结果
    average_results = {}
    
    for algorithm in all_results[0].keys():
        print(f"\n{algorithm} 算法统计结果:")
        average_results[algorithm] = {}
        
        for scenario in all_results[0][algorithm].keys():
            print(f"\n{scenario} 场景:")
            average_results[algorithm][scenario] = {}
            
            # 获取所有指标名称
            metrics = all_results[0][algorithm][scenario].keys()
            
            # 计算每个指标的均值和标准差
            for metric in metrics:
                values = [run[algorithm][scenario][metric] for run in all_results]
                mean = np.mean(values)
                std = np.std(values)
                print(f"{metric}:")
                print(f"  均值: {mean:.4f}")
                print(f"  标准差: {std:.4f}")
                
                # 存储均值结果
                average_results[algorithm][scenario][metric] = mean
    
    return average_results

def main():
    """主函数"""
    try:
        # 获取配置参数
        args = get_config()
        print("开始算法基准测试...")
        
        if args.n_runs == 1:
            # 单次运行，使用指定种子
            results = run_benchmark(args.seed, args)
            print_results(results, args.seed)
        else:
            # 多次运行，使用随机种子
            random_seeds = np.random.randint(0, 10000, size=args.n_runs)
            all_results = []
            
            for seed in random_seeds:
                results = run_benchmark(seed, args)
                all_results.append(results)
                print_results(results, seed)
            
            # 计算统计结果
            calculate_statistics(all_results)
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"主程序错误: {e}")
        raise

if __name__ == "__main__":
    main()