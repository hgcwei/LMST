import numpy as np
from matplotlib import pyplot as plt
#DBP1
# val_acc_history = [42.0, 43.0, 63.0, 62.0, 64.0, 68.0, 71.0, 74.0, 76.0, 78.0, 85.0, 85.0, 87.0, 87.0, 87.0, 88.0, 87.0, 87.0, 87.0, 87.0, 87.0]
# inter_acc_history = [49.3, 50.0, 53.0222, 53.5, 61.5111, 67.8333, 71.6889, 76.9333, 83.1222, 87.2778, 91.1, 94.8556, 97.9556, 98.6222, 98.7556, 98.8333, 98.7778, 98.8889, 98.7889, 98.9111, 98.8333]
# test_acc_history = [47.7004, 49.9343, 54.4021, 52.1682, 62.2865, 70.4336, 73.9816, 79.1064, 81.6032, 84.0999, 86.3338, 88.8305, 90.8016, 91.59, 91.0644, 91.3272, 91.1958, 91.4586, 91.3272, 91.4586, 91.3272]
# train_acc_history=[ 46.0, 57.0, 44.0, 46.0, 52.0, 62.0, 67.0, 70.0, 75.0, 80.0, 83.0, 95.0, 96.0, 98.0, 97.0, 98.0, 97.0, 99.0, 99.0, 99.0, 99.0]
# DBP2
# val_acc_history = [62.0, 61.0, 64.0, 67.0, 71.0, 74.0, 75.0, 76.0, 76.0, 75.0, 75.0, 75.0, 75.0, 75.0 ]
# inter_acc_history = [65.9755, 65.9755, 75.4006, 85.3911, 91.8002, 94.1565, 95.2875, 95.476, 95.5702, 96.3242, 96.23, 95.9472, 96.0415, 96.0415 ]
# test_acc_history = [67.4033, 69.0608, 74.0331, 82.3204, 84.5304, 85.0829, 87.2928, 87.2928, 87.8453, 87.8453, 87.8453, 86.7403, 87.2928, 87.2928]
# train_acc_history=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
#CPP1
# val_acc_history = [47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 47.0, 68.5, 82.0, 88.5, 91.0, 91.0, 91.5, 91.5, 91.5 ]
# inter_acc_history = [65.556, 65.556, 65.561, 65.561, 65.5672, 65.5547, 65.5934, 78.621, 91.9481, 94.8109, 95.4599, 95.5547, 95.6209, 95.6271, 95.6421 ]
# test_acc_history = [92.6812, 49.9982, 49.9982, 49.9982, 49.9982, 49.9694, 50.0378, 58.2881, 84.5459, 91.1814, 93.2099, 94.0119, 94.2061, 94.2384, 94.2384]
# train_acc_history=[53.0, 53.0, 53.0, 53.0, 53.0, 53.0, 53.0, 67.5, 86.5, 93.0, 93.5, 93.0, 93.0, 93.5, 93.5]
#CPP2
val_acc_history = [48.0, 48.0, 95.0, 97.0, 93.0, 93.0, 96.0, 98.0, 98.0, 98.0, 98.0, 98.0]
inter_acc_history = [57.9942, 57.9959, 94.8907, 92.9976, 89.3903, 89.818, 91.313, 93.8703, 95.6887, 95.8677, 95.8782, 95.8677]
test_acc_history = [93.1059, 50.9377, 93.5346, 90.4745, 86.4857, 86.9739, 88.5218, 91.2246, 93.4631, 93.731, 93.7072, 93.7191]
train_acc_history=[52.0, 52.0, 96.0, 92.0, 89.0, 89.0, 89.0, 93.0, 95.0, 95.0, 95.0, 95.0]


plt.figure(figsize=(10, 6))
plt.title(f'Initial Training Set: {100}')

morandi_colors = ['#26A7E1',  '#E95412','#13AF68', '#FFE009','#E274A9']
line_styles = ['-', '-', '-', '--','-']  # 不同的线型

    #绘制曲线
plt.plot(val_acc_history, color=morandi_colors[0], linestyle=line_styles[0], label='Validation Set')
plt.plot(inter_acc_history, color=morandi_colors[1], linestyle=line_styles[1], label='Inter Set')
plt.plot(test_acc_history, color=morandi_colors[2], linestyle=line_styles[2], label='Test Set')
plt.axhline(y=test_acc_history[0], color=morandi_colors[3], linestyle=line_styles[3],  label="Test Set(SVM0)")
plt.plot(train_acc_history, color=morandi_colors[4], linestyle=line_styles[4], label='Train Set')

plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig('lncRNA_2(600。1).png')
plt.savefig('4.png')
plt.close()
