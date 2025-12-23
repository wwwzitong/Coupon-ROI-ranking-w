import subprocess
import os
import sys

# 1. 配置基础参数
base_command = [
    "python", "train_copy.py",
    "--model_class_name", "DCN_base_DFCL",
    "--alpha", "1.2"
]

# 原始保存路径
base_model_path = "./model/DCN_base_DFCL_2pll_2pos_gradient_lr4_alpha=1.2"

# 运行次数
num_runs = 5

def run_training():
    for i in range(num_runs):
        print(f"\n{'='*20} 开始第 {i+1}/{num_runs} 次训练 {'='*20}")
        
        # 构造带有后缀的新路径，例如: ...alpha=1.2_run0
        current_save_path = f"{base_model_path}_run{i}"
        
        # 组合完整的命令
        # 注意：这里我们动态替换 model_path 参数
        cmd = base_command + ["--model_path", current_save_path]
        
        # 打印命令以便检查
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 调用命令行执行训练
            # check=True 会在脚本报错时停止运行
            subprocess.run(cmd, check=True)
            print(f"第 {i+1} 次训练完成。模型已保存至: {current_save_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"第 {i+1} 次训练出错，停止运行。错误信息: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_training()
    print("\n所有5次训练已完成！")