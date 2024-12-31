import argparse

def get_config():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="贝叶斯采样算法基准测试")
    
    # 运行参数
    parser.add_argument("--n-runs", type=int, default=1,
                       help="运行次数，若为1则使用指定种子")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子，仅在运行次数为1时使用")
    parser.add_argument("--n-iterations", type=int, default=10000,
                       help="每个算法的迭代次数")
    parser.add_argument("--n-burnin", type=int, default=1000,
                       help="burn-in阶段的迭代次数")
    
    # 算法和场景选择
    parser.add_argument("--algorithms", nargs="+",
                       default=['gibbs', 'hmc_stan', 'abc', 'diffusion', 
                               'vi', 'gflownet', 'gan', 'score', 'inla'],
                       help="要测试的算法列表")
    parser.add_argument("--scenarios", nargs="+",
                       default=['gmm', 'ssm', 'sparse'],
                       help="要测试的场景列表")
    
    return parser.parse_args() 