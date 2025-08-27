import os
import sys
import time
import random
from io import StringIO

import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.core import shape

import featureUniform
import svModel

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#label_state_ls: 标记状态列表，表示每个样本是否被标记（-1表示未标记，0/1表示已标记的类别）
#initial_indices: 初始样本的索引列表（包含初始选择的训练集和验证集样本）

def check_convergence(acc_history, window=5, tolerance=1):
    """
    检查验证集准确率是否收敛
    :param acc_history: 历史准确率列表
    :param window: 检测窗口大小（连续几轮）
    :param tolerance: 允许的波动阈值（百分比）
    :return: 是否收敛
    """
    if len(acc_history) < window:
        return False
    recent_acc = acc_history[-window:]
    max_diff = max(recent_acc) - min(recent_acc)
    return max_diff < tolerance

def get_tr_mat_by_label_state_ls(X, y, label_state_ls, initial_indices):
    """获取训练集和固定验证集"""
    # 获取所有已标记样本（包括初始标记和后续添加的）
    all_labeled_indices = [i for i, state in enumerate(label_state_ls) if state != -1]

    # 初始验证集是初始样本中未被选为训练集的部分（保持不变）
    val_indices = initial_indices[len(initial_indices) // 2:]

    # 训练集包括初始训练集和后续添加的高置信度样本
    train_indices = [i for i in all_labeled_indices if i not in val_indices]

    # 获取数据
    X_tr = X[train_indices, :]
    y_tr = [y[i] for i in train_indices]
    X_val = X[val_indices, :]
    y_val = [y[i] for i in val_indices]

    return X_tr, y_tr, X_val, y_val


def update_label_state_ls(label_state_ls, pred_train, num, low_thres, high_thres, initial_indices):
    """更新标记状态列表（不再修改初始验证集的标签）"""
    assert 0 < low_thres < high_thres < 1
    val_indices = initial_indices[len(initial_indices) // 2:]  # 固定验证集索引

    # 检查并移除预测不一致的样本（跳过验证集）
    for i in range(len(label_state_ls)):
        if label_state_ls[i] != -1 and i not in val_indices:  # 不修改验证集
            pred_label = 1 if pred_train[i] > 0.5 else 0
            if pred_label != label_state_ls[i]:
                label_state_ls[i] = -1

    # 添加高置信度的负样本
    pred_sort_train = sorted(enumerate(pred_train), key=lambda x: x[1])
    new_neg = 0
    for i in range(len(label_state_ls)):
        if new_neg >= num:
            break
        index, value = pred_sort_train[i]
        if value < low_thres and label_state_ls[index] == -1:
            label_state_ls[index] = 0
            new_neg += 1

    # 添加高置信度的正样本
    new_pos = 0
    for i in range(len(label_state_ls) - 1, -1, -1):
        if new_pos >= num:
            break
        index, value = pred_sort_train[i]
        if value > high_thres and label_state_ls[index] == -1:
            label_state_ls[index] = 1
            new_pos += 1

    print(f"更新了 {new_neg + new_pos} 个样本 (负:{new_neg}, 正:{new_pos})")
    return label_state_ls


def initialize_label_state(X,y, initial_pos=100, initial_neg=100, pos_num=4500, neg_num=4500):
    X_initial = []
    y_initial = []

    """初始化标记状态和固定验证集"""
    label_state_ls = [-1 for _ in range(pos_num + neg_num)]

    # 获取所有正样本和负样本的索引
    pos_indices = [i for i in range(pos_num) if y[i] == 1]
    neg_indices = [i for i in range(pos_num, pos_num + neg_num) if y[i] == 0]

    # 随机选择初始正负样本
    selected_pos = random.sample(pos_indices, initial_pos)
    selected_neg = random.sample(neg_indices, initial_neg)
    initial_indices = selected_pos + selected_neg
    random.shuffle(initial_indices)

    # 将初始样本分为训练集和验证集（验证集标签保持不变）
    split_point = len(initial_indices) // 2
    for i in initial_indices[:split_point]:  # 训练集部分
        label_state_ls[i] = y[i]
        X_initial.append(X[i, :])
        y_initial.append(y[i])
    for i in initial_indices[split_point:]:  # 验证集部分
        label_state_ls[i] = y[i]

    return label_state_ls, initial_indices,np.array(X_initial), y_initial


def main(pos_num, neg_num, initial_pos, initial_neg, max_iter, update_num, low_thres,
         high_thres, c, g, kb):
    """主函数"""
    start_time = time.time()
    val_acc_history = []
    inter_acc_history = []
    test_acc_history = []
    train_acc_history=[]
    best_val_acc = 0.0
    best_test_pred = None  # 保存最佳测试集预测结果
    best_test_acc = 0.0


    TNAME = 'swi'
    svm_dir = 'svm/' + TNAME
    os.makedirs(svm_dir, exist_ok=True)

    # 加载数据
    # X_train = np.loadtxt('data/DBP_1(415)/train_data.csv', delimiter=',', dtype=np.float64)
    # X_test = np.loadtxt('data/DBP_1(415)/test_data.csv', delimiter=',', dtype=np.float64)
    # y_train = np.loadtxt('data/DBP_1(415)/train_label.csv', delimiter=',').astype(np.int64)
    # y_test = np.loadtxt('data/DBP_1(415)/test_label.csv', delimiter=',').astype(np.int64)
    # X_train = np.loadtxt('data/DBP_2(338)/train_data.csv', delimiter=',', dtype=np.float64)
    # X_test = np.loadtxt('data/DBP_2(338)/test_data.csv', delimiter=',', dtype=np.float64)
    # y_train = np.loadtxt('data/DBP_2(338)/train_label.csv', delimiter=',').astype(np.int64)
    # y_test = np.loadtxt('data/DBP_2(338)/test_label.csv', delimiter=',').astype(np.int64)
    X_train = np.loadtxt('data/DBP_3(413)/train_data.csv', delimiter=',', dtype=np.float64)
    X_test = np.loadtxt('data/DBP_3(413)/test_data.csv', delimiter=',', dtype=np.float64)
    y_train = np.loadtxt('data/DBP_3(413)/train_label.csv', delimiter=',', dtype=np.int64)
    y_test = np.loadtxt('data/DBP_3(413)/test_label.csv', delimiter=',', dtype=np.int64)
    # X_train = np.loadtxt('data/CPP_1/train_data.csv', delimiter=',', dtype=np.float64)
    # X_test  = np.loadtxt('data/CPP_1/test_data.csv', delimiter=',', dtype=np.float64)
    # y_train = np.loadtxt('data/CPP_1/train_label.csv', delimiter=',').astype(np.int64)
    # y_test  = np.loadtxt('data/CPP_1/test_label.csv', delimiter=',').astype(np.int64)
    # X_train = np.loadtxt('data/CPP_2/train_data.csv', delimiter=',', dtype=np.float64)
    # X_test = np.loadtxt('data/CPP_2/test_data.csv', delimiter=',', dtype=np.float64)
    # y_train = np.loadtxt('data/CPP_2/train_label.csv', delimiter=',').astype(np.int64)
    # y_test = np.loadtxt('data/CPP_2/test_label.csv', delimiter=',').astype(np.int64)
    # X_train = np.loadtxt('data/CPP_3/train_data.csv', delimiter=',', dtype=np.float64)
    # X_test = np.loadtxt('data/CPP_3/test_data.csv', delimiter=',', dtype=np.float64)
    # y_train = np.loadtxt('data/CPP_3/train_label.csv', delimiter=',').astype(np.int64)
    # y_test = np.loadtxt('data/CPP_3/test_label.csv', delimiter=',').astype(np.int64)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train,y_test))
    # 初始化标记状态和固定验证集
    label_state_ls, initial_indices,X_initial, y_initial = initialize_label_state(X,y, initial_pos, initial_neg, pos_num, neg_num)
    for i in range(max_iter):
        print(f"Epoch {i + 1}")

        # 获取训练集和固定验证集
        X_tr, y_tr, X_val, y_val = get_tr_mat_by_label_state_ls(X, y, label_state_ls, initial_indices)

        # 格式化数据
        fu = featureUniform.FeatureUniform()
        fu.libsvm_form(X_tr, np.array(y_tr), svm_dir + '/train')
        fu.libsvm_form(X_val, np.array(y_val), svm_dir + '/val')
        fu.libsvm_form(X[0:pos_num + neg_num, :], np.array(y[0:pos_num + neg_num]), svm_dir + '/inter')
        fu.libsvm_form(X[pos_num + neg_num + 1:], np.array(y[pos_num + neg_num + 1:]), svm_dir + '/test')
        fu.libsvm_form(X_initial, np.array(y_initial), svm_dir + '/initial_train')

        # 训练和预测
        sm = svModel.SVModel(svm_dir, c=c, g=g, kb=kb)

        old_stdout = sys.stdout
        try:
            sys.stdout = captured_output = StringIO()
            sm.svm_scale()
            sm.svm_train()
            accuracy_val = sm.svm_predict_val()
            accuracy_inter = sm.svm_predict_inter()
            accuracy_test = sm.svm_predict_test()
            accuracy_train = sm.svm_predict_initial_train()
            sys.stdout = old_stdout
            print(captured_output.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Epoch {i + 1} 训练/预测出错: {str(e)}")
            continue

        # 记录准确率
        val_acc_history.append(accuracy_val)
        inter_acc_history.append(accuracy_inter)
        test_acc_history.append(accuracy_test)
        train_acc_history.append(accuracy_train)

        if i >= 10 and check_convergence(val_acc_history):
            converged_epoch = i + 1
            print(f"验证集准确率在{converged_epoch}轮收敛，提前停止训练")
            break

        pred_train, current_test_pred = sm.svm_evaluate()

        if accuracy_val > best_val_acc:
            best_val_acc = accuracy_val
            best_test_pred = current_test_pred  # 保存概率值
            best_test_acc = accuracy_test
            print(f"更新最佳模型: 验证集准确率 = {best_val_acc:.2f}%, 测试集准确率 = {best_test_acc:.2f}%")

        label_state_ls = update_label_state_ls(label_state_ls, pred_train, update_num, low_thres, high_thres,
                                              initial_indices)

    if best_test_pred is not None:
        np.savetxt('cpp-best_test_predictions.txt', best_test_pred, fmt='%.6f')  # 保存为6位小数的文本
        print("最佳测试集预测结果已保存到 dbp3_best_test_predictions.txt")
    print("\n准确率历史记录:")
    print(f"验证集准确率 (val_acc_history): {val_acc_history}")
    print(f"中间集准确率 (inter_acc_history): {inter_acc_history}")
    print(f"测试集准确率 (test_acc_history): {test_acc_history}")
    print(f"训练集准确率 (train_acc_history): {train_acc_history}")
    # # 绘图和输出结果（保持不变）
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f}秒")

    plt.figure(figsize=(10, 6))
    plt.title(f'Initial Training Set: {initial_pos // 2+initial_neg//2}')

    morandi_colors = ['#26A7E1',  '#E95412', '#13AF68','#FFE009','#E274A9']
    line_styles = ['-', '-', '-', '--','-']  # 不同的线型

    #绘制曲线
    plt.plot(val_acc_history, color=morandi_colors[0], linestyle=line_styles[0], label='Validation Set')
    plt.plot(inter_acc_history, color=morandi_colors[1], linestyle=line_styles[1], label='Inter Set')
    plt.plot(test_acc_history, color=morandi_colors[2], linestyle=line_styles[2], label='Test Set')
    plt.axhline(y=test_acc_history[0], color=morandi_colors[3], linestyle=line_styles[3],  label="SVM0 Test Set")
    plt.plot(train_acc_history, color=morandi_colors[4], linestyle=line_styles[4], label='Train Set')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig('lncRNA_2(600。1).png')
    plt.savefig('DBP11.1.png')
    plt.close()


if __name__ == "__main__":
    # random.seed(90)
    main(
        pos_num=4500,
        neg_num=4500,
        # test_pos_num=381,
        # test_neg_num=381,
        # pos_num=520,
        # neg_num=541,
        # pos_num=52530,
        # neg_num=27600,
        # pos_num=33360,
        # neg_num=24163,
        # pos_num=52530,
        # neg_num=27600,
        initial_pos=1000,
        initial_neg=1000,
        max_iter=60,
        update_num=1000,
        low_thres=0.4,
        high_thres=0.6,
        c=2,
        g=0.5,
        kb=413,
    )