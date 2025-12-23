"""
残差分布分析与可视化工具
功能：
1. 训练模型
2. 提取预测值和真实标签
3. 计算残差
4. 拟合多种分布并选择最优
5. 可视化对比
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：支持的分布类型
# ============================================

class DistributionFitter:
    """
    支持多种概率分布的拟合与诊断
    """
    
    DISTRIBUTIONS = {
        'normal': {
            'scipy': stats.norm,
            'params': ['loc', 'scale'],
            'description': 'Normal (Gaussian) Distribution'
        },
        'beta': {
            'scipy': stats.beta,
            'params':  ['a', 'b', 'loc', 'scale'],
            'description': 'Beta Distribution (适合 [0,1] 区间)'
        },
        'gamma':  {
            'scipy': stats. gamma,
            'params': ['a', 'loc', 'scale'],
            'description': 'Gamma Distribution (适合正偏分布)'
        },
        'lognorm': {
            'scipy': stats.lognorm,
            'params': ['s', 'loc', 'scale'],
            'description': 'Log-Normal Distribution'
        },
        'weibull': {
            'scipy': stats.weibull_min,
            'params': ['c', 'loc', 'scale'],
            'description': 'Weibull Distribution'
        },
        'laplace': {
            'scipy': stats.laplace,
            'params': ['loc', 'scale'],
            'description': 'Laplace Distribution (双指数分布，适合有尖峰)'
        },
        't': {
            'scipy': stats.t,
            'params':  ['df', 'loc', 'scale'],
            'description':  'Student-t Distribution (重尾分布)'
        },
        'cauchy': {
            'scipy':  stats.cauchy,
            'params': ['loc', 'scale'],
            'description': 'Cauchy Distribution (极重尾)'
        },
        'bernou·········lli': {
            'scipy': stats.bernoulli,
            'params': ['p', 'loc'],
            'description': 'Bernoulli Distribution (0/1 二元分布，离散)'
        }
    }
    
    @staticmethod
    def fit_distribution(data: np.ndarray, dist_name: str) -> Tuple[tuple, float, float]:
        """
        拟合指定分布
        
        Returns:
            params: 拟合的参数
            aic: 赤池信息准则（越小越好）
            bic: 贝叶斯信息准则（越小越好）
        """
        dist = DistributionFitter. DISTRIBUTIONS[dist_name]['scipy']
        
        # 拟合参数
        try:
            params = dist.fit(data)
        except Exception as e:
            print(f"警告:  {dist_name} 拟合失败 - {e}")
            return None, np.inf, np.inf
        
        # 计算对数似然
        log_likelihood = np.sum(dist.logpdf(data, *params))
        
        # 计算 AIC 和 BIC
        k = len(params)  # 参数个数
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        return params, aic, bic
    
    @staticmethod
    def goodness_of_fit_tests(data: np.ndarray, dist_name: str, params: tuple) -> Dict:
        """
        执行拟合优度检验
        """
        dist = DistributionFitter. DISTRIBUTIONS[dist_name]['scipy']
        
        # 1.  Kolmogorov-Smirnov 检验
        ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))
        
        # 2. Anderson-Darling 检验（仅支持部分分布）
        ad_stat, ad_crit, ad_sig = None, None, None
        if dist_name in ['normal', 'lognorm']: 
            try:
                ad_result = stats.anderson(data, dist=dist_name)
                ad_stat = ad_result.statistic
                ad_crit = ad_result.critical_values
                ad_sig = ad_result.significance_level
            except:
                pass
        
        return {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'ad_statistic': ad_stat,
            'ad_critical_values': ad_crit,
            'ad_significance_level': ad_sig
        }

# ============================================
# 第二部分：残差提取器
# ============================================

class ResidualExtractor:
    """
    从训练好的模型中提取残差
    """
    
    def __init__(self, model:  tf.keras.Model, dataset: tf.data.Dataset):
        self.model = model
        self.dataset = dataset
        self.residuals = {}
        self.predictions = {}
        self.labels = {}
    
    def extract(self, num_batches: int = None) -> Dict[str, np.ndarray]: 
        """
        提取残差：residual = label - prediction
        
        对于二分类问题：
        - label ∈ {0, 1}
        - prediction ∈ [0, 1] (sigmoid 输出)
        - residual ∈ [-1, 1]
        """
        all_predictions = {key: [] for key in ['paid_treatment_0', 'paid_treatment_1',
                                                 'cost_treatment_0', 'cost_treatment_1']}
        all_labels = {'paid':  [], 'cost': [], 'treatment': []}
        
        for batch_idx, (features, labels) in enumerate(self.dataset):
            if num_batches and batch_idx >= num_batches:
                break
            
            # 前向传播
            predictions = self.model(features, training=False)
            
            # 收集预测值（logit -> probability）
            for key in all_predictions. keys():
                if key in predictions:
                    logits = predictions[key]. numpy()
                    probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))  # sigmoid
                    all_predictions[key].append(probs)
            
            # 收集标签
            all_labels['paid'].append(labels['paid'].numpy())
            all_labels['cost'].append(labels['cost'].numpy())
            all_labels['treatment'].append(labels['treatment'].numpy())
        
        # 合并所有 batch
        for key in all_predictions: 
            all_predictions[key] = np.concatenate(all_predictions[key])
        
        for key in all_labels:
            all_labels[key] = np.concatenate(all_labels[key])
        
        # 计算残差（按 treatment 分组）
        residuals = {}
        treatment_idx = all_labels['treatment'].astype(int)
        
        for target in ['paid', 'cost']: 
            for treatment in [0, 1]: 
                mask = (treatment_idx == treatment)
                pred_key = f"{target}_treatment_{treatment}"
                
                y_true = all_labels[target][mask]
                y_pred = all_predictions[pred_key][mask]
                
                # 残差 = 真实值 - 预测值
                residual = y_true - y_pred
                
                residuals[pred_key] = residual
                self.predictions[pred_key] = y_pred
                self.labels[f"{target}_{treatment}"] = y_true
        
        self.residuals = residuals
        return residuals

# ============================================
# 第三部分：可视化工具
# ============================================

class ResidualVisualizer:
    """
    残差分布的可视化工具
    """
    
    @staticmethod
    def plot_residual_analysis(residuals: Dict[str, np.ndarray], 
                                fitted_distributions: Dict,
                                output_dir: str = './residual_analysis'):
        """
        生成完整的残差分析报告
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for res_key, res_data in residuals.items():
            # 每个残差类型生成一个独立的图
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. 原始残差直方图 + KDE
            ax1 = fig.add_subplot(gs[0, : 2])
            sns.histplot(res_data, bins=50, kde=True, stat='density', ax=ax1, alpha=0.6)
            ax1.set_title(f'{res_key} - Residual Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Residual (True - Predicted)')
            ax1.set_ylabel('Density')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
            ax1.legend()
            
            # 2. Q-Q 图（与正态分布对比）
            ax2 = fig.add_subplot(gs[0, 2:])
            stats.probplot(res_data, dist="norm", plot=ax2)
            ax2.set_title(f'{res_key} - Q-Q Plot (vs Normal)', fontsize=14)
            
            # 3-6. 拟合的分布对比
            best_dists = fitted_distributions[res_key]['ranked_distributions'][: 4]
            
            for idx, (dist_name, dist_info) in enumerate(best_dists):
                row = (idx // 2) + 1
                col = (idx % 2) * 2
                ax = fig.add_subplot(gs[row, col: col+2])
                
                # 绘制直方图
                ax.hist(res_data, bins=50, density=True, alpha=0.5, 
                       edgecolor='black', label='Observed')
                
                # 绘制拟合的分布
                dist = DistributionFitter. DISTRIBUTIONS[dist_name]['scipy']
                params = dist_info['params']
                x = np.linspace(res_data.min(), res_data.max(), 200)
                pdf = dist.pdf(x, *params)
                ax.plot(x, pdf, 'r-', linewidth=2, 
                       label=f'{dist_name. capitalize()} Fit')
                
                # 标注统计信息
                textstr = (f"AIC: {dist_info['aic']:.2f}\n"
                          f"BIC: {dist_info['bic']:.2f}\n"
                          f"KS p-value: {dist_info['ks_p_value']:.4f}")
                ax.text(0.65, 0.95, textstr, transform=ax. transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5), fontsize=10)
                
                ax.set_title(f'{dist_name.capitalize()} Distribution', fontsize=12)
                ax.set_xlabel('Residual')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.savefig(f'{output_dir}/{res_key}_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 已生成 {res_key} 的分析图表")
    
    @staticmethod
    def generate_summary_report(fitted_distributions: Dict, output_path: str):
        """
        生成文本摘要报告
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("残差分布拟合报告\n")
            f.write("=" * 80 + "\n\n")
            
            for res_key, dist_results in fitted_distributions.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"目标:  {res_key}\n")
                f.write(f"{'='*60}\n\n")
                
                ranked = dist_results['ranked_distributions']
                
                f.write("分布拟合排名 (按 AIC 从小到大):\n\n")
                
                for rank, (dist_name, info) in enumerate(ranked, 1):
                    f.write(f"{rank}. {dist_name. upper()}\n")
                    f.write(f"   描述: {DistributionFitter.DISTRIBUTIONS[dist_name]['description']}\n")
                    f.write(f"   AIC: {info['aic']:.4f}\n")
                    f.write(f"   BIC: {info['bic']:.4f}\n")
                    f.write(f"   KS统计量: {info['ks_statistic']:.4f}\n")
                    f.write(f"   KS p值: {info['ks_p_value']:.4f}\n")
                    f.write(f"   参数:  {info['params']}\n")
                    
                    # 判断是否接受该分布（p > 0.05 表示不能拒绝原假设）
                    if info['ks_p_value'] > 0.05:
                        f.write(f"   ✅ 拟合优度检验通过 (p > 0.05)\n")
                    else: 
                        f.write(f"   ❌ 拟合优度检验未通过 (p < 0.05)\n")
                    
                    f.write("\n")
                
                # 推荐
                best_dist_name, best_dist_info = ranked[0]
                f.write(f"\n🎯 推荐分布: {best_dist_name.upper()}\n")
                f.write(f"   原因: AIC 最小 ({best_dist_info['aic']:.4f})\n\n")
        
        print(f"✅ 摘要报告已保存到 {output_path}")

# ============================================
# 第四部分：自定义损失函数生成器
# ============================================

class CustomLossGenerator:
    """
    根据拟合的分布自动生成对应的负对数似然损失函数
    """
    
    @staticmethod
    def generate_nll_loss(dist_name: str, fitted_params: tuple) -> Callable:
        """
        生成基于特定分布的负对数似然损失（TensorFlow版本）
        
        注意：对于回归问题，残差 = y_true - y_pred
        我们要建模残差的分布，因此损失是 -log P(residual | params)
        """
        
        if dist_name == 'normal': 
            # 正态分布:  N(0, σ²)
            # 参数: loc=0, scale=σ
            loc, scale = fitted_params
            
            def normal_nll(y_true, y_pred):
                """
                y_true: 真实标签 [batch_size]
                y_pred: 预测值 [batch_size]
                """
                residual = y_true - y_pred
                # -log p(residual) = 0.5 * log(2πσ²) + (residual - μ)² / (2σ²)
                nll = 0.5 * tf.math.log(2 * np.pi * scale**2) + \
                      0.5 * tf.square(residual - loc) / (scale**2)
                return tf.reduce_mean(nll)
            
            return normal_nll
        
        elif dist_name == 'laplace':
            # 拉普拉斯分布 (L1损失的概率解释)
            loc, scale = fitted_params
            
            def laplace_nll(y_true, y_pred):
                residual = y_true - y_pred
                nll = tf.math.log(2 * scale) + tf.abs(residual - loc) / scale
                return tf.reduce_mean(nll)
            
            return laplace_nll
        
        elif dist_name == 't':
            # Student-t 分布（对离群值更鲁棒）
            df, loc, scale = fitted_params
            
            def t_nll(y_true, y_pred):
                residual = y_true - y_pred
                # t分布的负对数似然（近似）
                standardized = (residual - loc) / scale
                nll = -tf.math.lgamma((df + 1) / 2) + tf.math.lgamma(df / 2) + \
                      0.5 * tf.math. log(df * np.pi * scale**2) + \
                      ((df + 1) / 2) * tf.math.log(1 + tf.square(standardized) / df)
                return tf. reduce_mean(nll)
            
            return t_nll
        
        elif dist_name == 'beta':
            # Beta 分布（适合 [0, 1] 范围的残差）
            a, b, loc, scale = fitted_params
            
            def beta_nll(y_true, y_pred):
                residual = y_true - y_pred
                # 将残差映射到 [0, 1]
                residual_scaled = (residual - loc) / scale
                residual_clipped = tf.clip_by_value(residual_scaled, 1e-6, 1 - 1e-6)
                
                # Beta 分布的负对数似然
                nll = -((a - 1) * tf.math.log(residual_clipped) + \
                       (b - 1) * tf.math.log(1 - residual_clipped)) + \
                      tf.math. lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
                return tf.reduce_mean(nll)
            
            return beta_nll
        
        elif dist_name == 'gamma':
            # Gamma 分布
            a, loc, scale = fitted_params
            
            def gamma_nll(y_true, y_pred):
                residual = y_true - y_pred
                # Gamma 要求 x > 0，需要平移
                residual_shifted = residual - loc
                residual_clipped = tf.maximum(residual_shifted, 1e-6)
                
                nll = a * tf.math.log(scale) + tf.math.lgamma(a) - \
                      (a - 1) * tf.math.log(residual_clipped) + \
                      residual_clipped / scale
                return tf.reduce_mean(nll)
            
            return gamma_nll

        elif dist_name == 'bernoulli':
            # 伯努利分布: 对应二分类问题的 Binary Cross Entropy
            # 注意：伯努利分布不用于建模“残差”(y - y_pred)，而是直接建模 y_true (0或1)
            # fitted_params 在这里通常是全局的平均概率 p，但在 NLL 计算中我们通常使用模型输出的 y_pred 作为动态概率
            
            # scipy.stats.bernoulli.fit 通常返回 (p, loc)，这里主要为了解包参数占位
            if len(fitted_params) >= 1:
                p_prior = fitted_params[0] 
            
            def bernoulli_nll(y_true, y_pred):
                """
                y_true: 真实标签，必须是 0 或 1
                y_pred: 模型预测的概率值 (通常经过 Sigmoid)，范围 (0, 1)
                """
                # 数值稳定性处理：防止 log(0)
                # 将概率限制在 [epsilon, 1-epsilon] 之间
                epsilon = 1e-7
                y_pred_clipped = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
                
                # 伯努利分布的负对数似然公式 (即二元交叉熵):
                # NLL = - [y * log(p) + (1-y) * log(1-p)]
                nll = -(y_true * tf.math.log(y_pred_clipped) + \
                        (1 - y_true) * tf.math.log(1 - y_pred_clipped))
                
                return tf.reduce_mean(nll)
            
            return bernoulli_nll
        
        else:
            raise ValueError(f"不支持的分布:  {dist_name}")
    
    @staticmethod
    def save_loss_functions_as_code(fitted_distributions: Dict, output_path: str):
        """
        将最优损失函数保存为可直接使用的 Python 代码
        """
        code_lines = [
            "# 自动生成的自定义损失函数\n",
            "import tensorflow as tf\n",
            "import numpy as np\n\n"
        ]
        
        for res_key, dist_results in fitted_distributions.items():
            best_dist_name, best_dist_info = dist_results['ranked_distributions'][0]
            params = best_dist_info['params']
            
            code_lines.append(f"# {res_key} 的最优损失函数 (基于 {best_dist_name} 分布)\n")
            code_lines.append(f"# AIC: {best_dist_info['aic']:.4f}, BIC: {best_dist_info['bic']:.4f}\n\n")
            
            # 生成函数代码
            func_name = f"loss_{res_key. replace('_', '__')}"
            
            if best_dist_name == 'normal':
                loc, scale = params
                code_lines.append(f"def {func_name}(y_true, y_pred):\n")
                code_lines.append(f"    \"\"\"正态分布负对数似然损失 (μ={loc:.4f}, σ={scale:.4f})\"\"\"\n")
                code_lines.append(f"    residual = y_true - y_pred\n")
                code_lines.append(f"    nll = 0.5 * tf.math.log({2 * np.pi * scale**2:. 6f}) + \\\n")
                code_lines. append(f"          0.5 * tf.square(residual - {loc:.6f}) / {scale**2:.6f}\n")
                code_lines.append(f"    return tf.reduce_mean(nll)\n\n")
            
            elif best_dist_name == 'laplace':
                loc, scale = params
                code_lines.append(f"def {func_name}(y_true, y_pred):\n")
                code_lines.append(f"    \"\"\"拉普拉斯分布负对数似然损失 (μ={loc:.4f}, b={scale:.4f})\"\"\"\n")
                code_lines.append(f"    residual = y_true - y_pred\n")
                code_lines.append(f"    nll = tf.math.log({2 * scale:.6f}) + tf.abs(residual - {loc:.6f}) / {scale:.6f}\n")
                code_lines.append(f"    return tf.reduce_mean(nll)\n\n")
            
            elif best_dist_name == 't':
                df, loc, scale = params
                code_lines.append(f"def {func_name}(y_true, y_pred):\n")
                code_lines.append(f"    \"\"\"Student-t 分布负对数似然损失 (df={df:.2f}, μ={loc:.4f}, σ={scale:.4f})\"\"\"\n")
                code_lines.append(f"    residual = y_true - y_pred\n")
                code_lines.append(f"    standardized = (residual - {loc:.6f}) / {scale:.6f}\n")
                code_lines.append(f"    nll = -tf.math.lgamma({(df + 1) / 2:.6f}) + tf.math.lgamma({df / 2:.6f}) + \\\n")
                code_lines.append(f"          0.5 * tf.math.log({df * np.pi * scale**2:.6f}) + \\\n")
                code_lines.append(f"          {(df + 1) / 2:.6f} * tf. math.log(1 + tf.square(standardized) / {df:.6f})\n")
                code_lines.append(f"    return tf.reduce_mean(nll)\n\n")

            elif best_dist_name == 'bernoulli':
                p_mean = params[0] if params else 0.5 
                
                code_lines.append(f"def {func_name}(y_true, y_pred):\n")
                code_lines.append(f"    \"\"\"伯努利分布负对数似然损失 (即 Binary Cross Entropy, Avg p={p_mean:.4f})\"\"\"\n")
                code_lines.append(f"    # 注意：伯努利分布不计算残差，而是直接对概率建模\n")
                code_lines.append(f"    # 数值稳定性处理，防止 log(0)\n")
                code_lines.append(f"    epsilon = 1e-7\n")
                code_lines.append(f"    y_pred_c = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)\n")
                code_lines.append(f"    nll = -(y_true * tf.math.log(y_pred_c) + (1 - y_true) * tf.math.log(1 - y_pred_c))\n")
                code_lines.append(f"    return tf.reduce_mean(nll)\n\n")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(code_lines)
        
        print(f"✅ 损失函数代码已保存到 {output_path}")

# ============================================
# 第五部分：主流程
# ============================================

def analyze_residual_distribution(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_dir: str = './residual_analysis',
    num_batches: int = 100
):
    """
    完整的残差分布分析流程
    
    Args:
        model: 训练好的模型
        dataset: 验证数据集
        output_dir: 输出目录
        num_batches: 使用多少个batch进行分析
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("残差分布分析流程")
    print("="*80)
    
    # 步骤 1: 提取残差
    print("\n📊 步骤 1/4: 提取残差...")
    extractor = ResidualExtractor(model, dataset)
    residuals = extractor.extract(num_batches=num_batches)
    
    for key, res in residuals.items():
        print(f"  {key}: {len(res)} 个样本, 均值={res.mean():.4f}, 标准差={res.std():.4f}")
    
    # 步骤 2: 拟合多种分布
    print("\n🔬 步骤 2/4: 拟合分布...")
    fitted_distributions = {}
    
    for res_key, res_data in residuals.items():
        print(f"\n  分析 {res_key}...")
        results = {}
        
        for dist_name in DistributionFitter.DISTRIBUTIONS. keys():
            params, aic, bic = DistributionFitter.fit_distribution(res_data, dist_name)
            
            if params is None:
                continue
            
            # 拟合优度检验
            gof_tests = DistributionFitter. goodness_of_fit_tests(res_data, dist_name, params)
            
            results[dist_name] = {
                'params': params,
                'aic': aic,
                'bic': bic,
                **gof_tests
            }
            
            print(f"    {dist_name:<15s}:  AIC={aic: 10.2f}, BIC={bic:10.2f}, KS_p={gof_tests['ks_p_value']:.4f}")
        
        # 按 AIC 排序
        ranked = sorted(results.items(), key=lambda x: x[1]['aic'])
        fitted_distributions[res_key] = {
            'ranked_distributions': ranked,
            'best_distribution': ranked[0][0] if ranked else None
        }
    
    # 步骤 3: 可视化
    print("\n📈 步骤 3/4: 生成可视化...")
    ResidualVisualizer.plot_residual_analysis(residuals, fitted_distributions, output_dir)
    
    # 步骤 4: 生成报告和代码
    print("\n📝 步骤 4/4: 生成报告...")
    ResidualVisualizer. generate_summary_report(
        fitted_distributions, 
        f'{output_dir}/residual_analysis_report.txt'
    )
    
    CustomLossGenerator.save_loss_functions_as_code(
        fitted_distributions,
        f'{output_dir}/custom_loss_functions.py'
    )
    
    # 保存拟合结果为 JSON
    summary = {}
    for res_key, dist_results in fitted_distributions.items():
        summary[res_key] = {
            'best_distribution': dist_results['best_distribution'],
            'top_3_distributions': [
                {
                    'name': name,
                    'aic': info['aic'],
                    'bic': info['bic'],
                    'ks_p_value': info['ks_p_value']
                }
                for name, info in dist_results['ranked_distributions'][: 3]
            ]
        }
    
    with open(f'{output_dir}/fitted_distributions.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ 分析完成！")
    print(f"📁 结果保存在:  {output_dir}/")
    print(f"   - 可视化图表: {output_dir}/*_analysis.png")
    print(f"   - 文本报告: {output_dir}/residual_analysis_report.txt")
    print(f"   - 自定义损失函数:  {output_dir}/custom_loss_functions.py")
    print(f"   - JSON摘要: {output_dir}/fitted_distributions.json")
    
    return fitted_distributions, residuals

# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 示例：加载模型和数据
    # model = tf.keras.models.load_model('path/to/your/model')
    # dataset = ...   # 你的 tf.data.Dataset
    
    # analyze_residual_distribution(model, dataset, num_batches=200)
    
    print("请在主脚本中导入此模块并调用 analyze_residual_distribution()")