import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd

# 基础设置
matplotlib.use('pdf')
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 读取JSON文件
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def extract_data(data):
    sorted_data = sorted(data, key=lambda x: x[1])
    epsilon_values = [item[1] for item in sorted_data]
    metric_values = [item[2] for item in sorted_data]
    return epsilon_values, metric_values

def smooth_values(values, weight=0.8):
    """
    使用指数移动平均进行平滑处理
    weight: 平滑权重 (0,1), 越大平滑度越高
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# 定义所有指标及其文件名
metrics = {
    'BERTScore': ('./F1_score/run-blackbox-tag-F1_Score.json', './F1_score/run-local-tag-F1_Score.json'),
    'Coherence': ('./coherence/run-blackbox-tag-Coherence.json', './coherence/run-local-tag-Coherence.json'),
    'Diversity': ('./diversitry/blackbox-tag-Diversity.json', './diversitry/local-tag-Diversity.json'),
    'ROUGE-1': ('./ROUCE-1/run-blackbox-tag-ROUGE-1.json', './ROUCE-1/run-local-tag-ROUGE-1.json'),
    'ROUGE-2': ('./ROUCE-2/run-blackbox-tag-ROUGE-2.json', './ROUCE-2/run-local-tag-ROUGE-2.json'),
    'ROUGE-L': ('./ROUCE-L/run-blackbox-tag-ROUGE-L.json', './ROUCE-L/run-local-tag-ROUGE-L.json')
}


# 创建2x3的子图布局，每个位置包含两个子图
fig, axes = plt.subplots(4, 3, figsize=(15, 12), 
                        gridspec_kw={'height_ratios': [1.5, 0.75, 1.5, 0.75]},
                        sharex='col')

# 为每个指标创建子图
for idx, (metric_name, (blackbox_file, local_file)) in enumerate(metrics.items()):
    # 计算子图位置
    col = idx % 3
    row = (idx // 3) * 2  # 每个指标占用两行
    
    # 主图和差异图
    ax_main = axes[row, col]
    ax_diff = axes[row + 1, col]
    
    # 加载数据
    random_data = load_json(blackbox_file)
    recovered_data = load_json(local_file)
    
    # 提取数据
    epsilon_values, random_metric = extract_data(random_data)
    _, recovered_metric = extract_data(recovered_data)
    
    # 应用平滑处理
    random_smooth = smooth_values(random_metric, weight=0.6)
    recovered_smooth = smooth_values(recovered_metric, weight=0.6)
    
    # 计算差异值
    diff_values = [r - p for r, p in zip(random_metric, recovered_metric)]
    diff_smooth = smooth_values(diff_values, weight=0.6)
    
    # 绘制主图
    ax_main.plot(epsilon_values, random_smooth,
                label='$U(P)$',
                linewidth=1.5, linestyle='-',
                color=sns.color_palette("deep")[0])
    ax_main.scatter(epsilon_values, random_metric,
                   marker='o', s=30, color=sns.color_palette("deep")[0],
                   alpha=0.6, edgecolor='white')
    
    ax_main.plot(epsilon_values, recovered_smooth,
                label='$ U(\\widetilde{P})$',
                linewidth=1.5, linestyle='-',
                color=sns.color_palette("deep")[2])
    ax_main.scatter(epsilon_values, recovered_metric,
                   marker='s', s=30, color=sns.color_palette("deep")[2],
                   alpha=0.6, edgecolor='white')
    
    # 绘制差异图
    ax_diff.plot(epsilon_values, diff_smooth,
                label='$\\epsilon_u$',
                linewidth=1.5, linestyle='-',
                color=sns.color_palette("deep")[3])
    ax_diff.scatter(epsilon_values, diff_values,
                   marker='^', s=30, color=sns.color_palette("deep")[3],
                   alpha=0.6, edgecolor='white')
    
    # 设置主图属性
    ax_main.set_xscale('log')
    ax_main.grid(True, linestyle='--', alpha=0.5)
    ax_main.set_title(metric_name, fontsize=12, fontweight='bold', pad=10)
    
    if col == 0:  # 只在左侧子图添加y轴标签
        ax_main.set_ylabel('Utility Score', fontsize=10)
        ax_diff.set_ylabel('Utility Loss', fontsize=10)
    
    # 设置差异图属性
    ax_diff.set_xscale('log')
    ax_diff.grid(True, linestyle='--', alpha=0.5)
    
    if row == 2:  # 只在底部子图添加x轴标签
        ax_diff.set_xlabel('Privacy Budget (ε)', fontsize=10)
    
    # 只在第一个主图显示图例
    if idx == 0:
        ax_main.legend(loc='upper right', fontsize=8, frameon=True,
                      facecolor='white', edgecolor='gray', framealpha=0.9)
        ax_diff.legend(loc='upper right', fontsize=8, frameon=True,
                      facecolor='white', edgecolor='gray', framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('all_metrics_analysis.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('all_metrics_analysis.eps', format='eps', bbox_inches='tight', dpi=300)

# 显示图表
plt.show()

# 打印所有指标的详细数值
for metric_name, (blackbox_file, local_file) in metrics.items():
    print(f"\n{metric_name} Values:")
    print("ε\tGPT-3.5\tInferDPT\tDifference")
    
    random_data = load_json(blackbox_file)
    recovered_data = load_json(local_file)
    
    epsilon_values, random_metric = extract_data(random_data)
    _, recovered_metric = extract_data(recovered_data)
    
    for i in range(len(epsilon_values)):
        diff = random_metric[i] - recovered_metric[i]
        print(f"{epsilon_values[i]:.1f}\t{random_metric[i]:.4f}\t{recovered_metric[i]:.4f}\t{diff:.4f}")