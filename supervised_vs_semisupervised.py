import matplotlib.pyplot as plt
import numpy as np

# 数据
initial_training_sets = [50,80, 100, 150, 200]
# initial_training_sets = [50, 100, 200, 400, 600]
#DBP1
# supervised_acc = [48.3574, 59.6583, 63.7319, 70.6965, 77.5296]
# semi_supervised_acc = [90.6702, 90.0131, 90.6702, 91.3272, 90.5388]
#DBP2
# supervised_acc = [45.3039,40.3315, 41.4365, 62.6965, 71.9296]
# semi_supervised_acc = [83.9779, 84.5304, 84.6878, 84.5028, 83.4254]
#cpp1
# supervised_acc = [91.9763,92.5589, 92.6812, 93.8249, 93.9471]
# semi_supervised_acc = [94.4255, 94.4506, 94.2384, 94.2996, 94.4938]
#cpp2
supervised_acc = [91.2603,93.1059, 92.7487, 92.3975, 93.3679]
semi_supervised_acc = [93.8322, 93.7191, 93.868, 93.8858, 93.7906]
# supervised_err = [20 ,20, 15, 15,13]  # 示例误差值
# semi_supervised_err = [2.5,2.5,3.1, 3, 3.1]  # 示例误差值
supervised_err = [0.4 ,0.3, 0.4, 0.3,0.4]  # 示例误差值
semi_supervised_err = [0.2,0.2, 0.2, 0.2, 0.2]  # 示例误差值
# supervised_err = [20 ,20, 15, 15,13]  # 示例误差值
# semi_supervised_err = [5.5,5.0,4.8, 4, 4]  # 示例误差值
# 设置参数
bar_width = 0.4
group_spacing = 0.5
index = np.arange(len(initial_training_sets)) * (bar_width*2 + group_spacing)

# 创建图形
plt.figure(figsize=(15, 9))

# 绘制柱状图（带误差线）
bar1 = plt.bar(index - bar_width/2, supervised_acc, bar_width,
               yerr=supervised_err, capsize=5,
               label='Supervised', color='#FFE009', hatch='/',
               edgecolor='black', error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

bar2 = plt.bar(index + bar_width/2, semi_supervised_acc, bar_width,
               yerr=semi_supervised_err, capsize=5,
               label='Semi-supervised', color='#26A7E1', hatch='.',
               edgecolor='black', error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

# 美化标签和标题
plt.xlabel('Initial Training Set Size', fontsize=14, labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
plt.title('Comparison of Supervised vs Semi-supervised Learning Accuracy\n',
          fontsize=16, pad=20)
plt.xticks(index, initial_training_sets, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(65, 105)

# 坐标轴调整

plt.xlim(index[0] - bar_width*1.5, index[-1] + bar_width*1.5)

# 图例和网格
plt.legend(loc='upper left', fontsize=12, framealpha=1)
plt.grid(axis='y', linestyle=':', alpha=0.5)

# 保存高质量图片
plt.tight_layout()
plt.savefig('new1', dpi=300, bbox_inches='tight')
plt.close()