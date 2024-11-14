import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

# 设置后端为矢量格式
matplotlib.use('pdf')
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 读取JSON文件
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 加载数据
local_data = load_json('./local-tag-Diversity.json')
blackbox_data = load_json('./blackbox-tag-Diversity.json')

def extract_data(data):
    # 数据格式为 [timestamp, epsilon, distinct_ratio]
    sorted_data = sorted(data, key=lambda x: x[1])  # 按epsilon排序
    epsilon_values = [item[1] for item in sorted_data]
    distinct_ratio = [item[2] for item in sorted_data]
    return epsilon_values, distinct_ratio

# 提取数据
epsilon_values, blackbox_distinct = extract_data(blackbox_data)
_, local_distinct = extract_data(local_data)

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5), height_ratios=[1, 1], 
                              gridspec_kw={'hspace': 0.3})

# 第一个子图：显示原始的多样性比较
ax1.plot(epsilon_values, blackbox_distinct,
         label='Original+BlackBox', 
         marker='o', markersize=4,
         linewidth=1.5, linestyle='-',
         color=sns.color_palette("deep")[0],
         markerfacecolor='white')

ax1.plot(epsilon_values, local_distinct,
         label='Protected+LocalLLM', 
         marker='s', markersize=4,
         linewidth=1.5, linestyle='-',
         color=sns.color_palette("deep")[2],
         markerfacecolor='white')

# 第二个子图：显示多样性损失
diversity_loss = np.array(blackbox_distinct) - np.array(local_distinct)
ax2.plot(epsilon_values, diversity_loss,
         label='Diversity Loss', 
         marker='o', markersize=4,
         linewidth=1.5, linestyle='-',
         color=sns.color_palette("deep")[1],
         markerfacecolor='white')

# 设置两个子图的共同属性
for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Privacy Budget (ε)', fontsize=10)

# 设置第一个子图的特定属性
ax1.set_title('Diversity Comparison', fontsize=11, fontweight='bold', pad=15)
ax1.set_ylabel('Distinct Ratio', fontsize=10)
ax1.legend(loc='upper right',
          fontsize=7,
          frameon=True,
          facecolor='white',
          edgecolor='gray',
          framealpha=0.9)

# 设置第二个子图的特定属性
ax2.set_title('Diversity Loss', fontsize=11, fontweight='bold', pad=15)
ax2.set_ylabel('Loss', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('./diversity/diversity_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('./diversity/diversity_comparison.eps', format='eps', bbox_inches='tight', dpi=300)

plt.show()

# 打印详细数值
print("\nDetailed Values:")
print("ε\tBlackBox(D)\tLocalLLM(D)\tDiversity Loss")
for i in range(len(epsilon_values)):
    print(f"{epsilon_values[i]:.1f}\t{blackbox_distinct[i]:.4f}\t{local_distinct[i]:.4f}\t{diversity_loss[i]:.4f}")