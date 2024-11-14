import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

# 设置后端为矢量格式
matplotlib.use('pdf')  # 或 'ps' 用于 EPS 格式

# 设置 seaborn 样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)  # 调整为更适合印刷的字体大小

# 读取JSON文件
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 加载数据
random_data = load_json('./Random_Prompt(R1)-tag-Recovery_Metrics.json')
recovered_data = load_json('./Perturbed_Prompt(R2)-tag-Recovery_Metrics.json')

def extract_data(data):
    sorted_data = sorted(data, key=lambda x: x[1])
    epsilon_values = [item[1] for item in sorted_data]
    cosine_values = [item[2] for item in sorted_data]
    return epsilon_values, cosine_values

# 提取并处理数据
epsilon_values, random_cosine = extract_data(random_data)
_, recovered_cosine = extract_data(recovered_data)

# 计算指标
R_random = [1 - cos for cos in random_cosine]
R_protected = [1 - cos for cos in recovered_cosine]
privacy_leakage = [rand - prot for rand, prot in zip(R_random, R_protected)]

# 创建图表 (单栏宽度 88mm ≈ 3.46 inches)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 3.5), height_ratios=[2, 1], gridspec_kw={'hspace': 0.3})

# 绘制恢复率的子图
ax1.plot(epsilon_values, R_random, label='$R(\\breve{P})$', marker='o', markersize=5,
         linewidth=1.5, linestyle='-', color=sns.color_palette("deep")[0], markerfacecolor='white')
ax1.plot(epsilon_values, R_protected, label='$R(\\widetilde{P})$', marker='s', markersize=5,
         linewidth=1.5, linestyle='-', color=sns.color_palette("deep")[2], markerfacecolor='white')

ax1.set_xscale('log')
ax1.set_ylim(0.24, 0.258)
ax1.set_ylabel('Recovery Extent', fontsize=10)
ax1.set_title('Effect of Privacy Budget (ε) on Recovery Extent', fontsize=11, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# 绘制隐私泄露的子图
ax2.plot(epsilon_values, privacy_leakage, label='$\\epsilon_p$', marker='^', markersize=5,
         linewidth=1.5, linestyle='-', color=sns.color_palette("deep")[3], markerfacecolor='white')
ax2.set_xscale('log')
ax2.set_ylim(-0.011, 0.01)
ax2.set_xlabel('Privacy Budget (ε)', fontsize=10)
ax2.set_ylabel('Privacy Leakage', fontsize=10)
ax2.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)

# 调整布局
plt.tight_layout()

# 保存为矢量格式
plt.savefig('./privacy/privacy_analysis_v2.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('./privacy/privacy_analysis_v2.eps', format='eps', bbox_inches='tight', dpi=300)

plt.show()

# 打印数值
print("\nDetailed Values:")
print("ε\tR(ep)\tR(p~)\tPrivacy Leakage")
for i in range(len(epsilon_values)):
    print(f"{epsilon_values[i]:.1f}\t{R_random[i]:.4f}\t{R_protected[i]:.4f}\t{privacy_leakage[i]:.4f}")
