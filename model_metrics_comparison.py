import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['SN', 'SP', 'PRE', 'ACC', 'F1-score', 'MCC']
# TSVM = [0.886, 0.951, 0.934, 0.900, 0.898, 0.800]
# send_kmeans = [0.370, 0.952, 0.881, 0.657, 0.519, 0.384]
# our_method = [0.9433, 0.9449, 0.9448, 0.9447, 0.945, 0.889]
#cpp2
# TSVM        = [0.834, 0.975, 0.956, 0.900, 0.863, 0.759]
# send_kmeans = [0.355, 0.997, 0.992, 0.670, 0.519, 0.455]
# our_method  = [0.9070, 0.9723, 0.9714, 0.9389, 0.938, 0.880]
#dbp1
# TSVM        = [0.829, 0.871, 0.859, 0.833, 0.830, 0.667]
# send_kmeans = [0.730, 0.430, 0.558, 0.575, 0.629, 0.156]
# our_method  = [0.8931, 0.9396, 0.9361, 0.9132, 0.911, 0.828]
#DBP2
TSVM        = [0.944, 0.815, 0.833, 0.879, 0.885, 0.765]
send_kmeans = [0.789, 0.663, 0.696, 0.725, 0.740, 0.455]
our_method  = [0.9101, 0.8587, 0.8602, 0.8784, 0.879, 0.758]
# Set the width of the bars and spacing
bar_width = 0.1
group_spacing = 0.4  # 调整指标组之间的间距

# Create positions for the bars
index = np.arange(len(metrics)) * group_spacing  # 每组指标的起始位置

# Create the figure
plt.figure(figsize=(6, 6))  # 加宽图形以避免标签重叠

# Plot the bars (三个柱子并列)
bar1 = plt.bar(index - bar_width, TSVM, bar_width,
               label='TSVM', color='#C4D8F2', edgecolor='black')
bar2 = plt.bar(index, send_kmeans, bar_width,
               label='Seed K-means', color='#F2E8E3', edgecolor='black')
bar3 = plt.bar(index + bar_width, our_method, bar_width,
               label='Our Method', color='#8E2D30', edgecolor='black')

# 添加标签和标题
plt.ylabel('Score', fontsize=12)
plt.xticks(index, metrics, fontsize=11)

# 设置坐标轴范围
plt.xlim(index[0] - 2*bar_width, index[-1] + 2*bar_width)
plt.ylim(0.3, 1.1)

# 添加图例和网格
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 调整布局并保存
plt.tight_layout()
plt.savefig('NEW.png', dpi=300, bbox_inches='tight')
plt.show()