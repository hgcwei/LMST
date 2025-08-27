import os
from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import f1_score,matthews_corrcoef,precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np


class SVModel:

    def __init__(self, save_dir, c, g, kb):
        os.environ["PATH"] += os.pathsep + os.getcwd() + '/libsvm/windows/'

        self.range = save_dir + '/range.file'
        self.train_file = save_dir + '/train.data.nonscaled'
        self.inter_file = save_dir + '/inter.data.nonscaled'
        self.scaled_train_file = save_dir + '/train.data.scaled'
        self.scaled_inter_file = save_dir + '/inter.data.scaled'
        self.test_file = save_dir + '/test.data.nonscaled'
        self.scaled_test_file = save_dir + '/test.data.scaled'
        self.model_save_file = save_dir + '/model.c'+ str(c) + '.g' + str(g) + '.kb' + str(kb)
        self.predict_save_file = save_dir + '/predict.file'
        self.predict_inter_save_file = save_dir + '/predict.inter.file'
        self.predict_train_save_file = save_dir + '/predict.train.file'
        self.results_save_file = save_dir + '/results.file'

        self.initial_train_file = save_dir + '/initial_train.data.nonscaled'
        self.scaled_initial_train_file = save_dir + '/initial_train.data.scaled'
        self.predict_initial_train_file = save_dir + '/predict.initial_train.file'

        self.val_file = save_dir + '/val.data.nonscaled'
        self.scaled_val_file = save_dir + '/val.data.scaled'
        self.predict_val_save_file = save_dir + '/predict.val.file'
        self.c = c
        self.g = g
        self.kb = kb

    def get_test_y(self):
        test_y = []
        f = open(self.test_file,'r')
        for line in f.readlines():
            test_y.append(int(line[0]))
        return test_y

    def svm_scale(self):
        cmd0 = 'svm-scale -l 0 -u 1 -s '+ self.range + ' '+ self.train_file + ' > ' + self.scaled_train_file
        cmd1 = 'svm-scale -r '+ self.range + ' '+ self.test_file + ' > ' + self.scaled_test_file
        cmd2 = 'svm-scale -l 0 -u 1 -s ' + self.range + ' ' + self.inter_file + ' > ' + self.scaled_inter_file
        cmd3 = 'svm-scale -r ' + self.range + ' ' + self.val_file + ' > ' + self.scaled_val_file
        cmd4 = 'svm-scale -r  ' + self.range + ' ' + self.initial_train_file + ' > ' + self.scaled_initial_train_file

        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)

    # def svm_train(self):
    #     cmd = 'svm-train -c '+ str(self.c) +' -g '+ str(self.g) +' -b 1 '+ self.scaled_train_file + ' ' + self.model_save_file
    #     os.system(cmd)
    def svm_train(self):
        cmd = f'svm-train -h 0 -c {self.c} -g {self.g} -b 1 {self.scaled_train_file} {self.model_save_file}'
        os.system(cmd)

    def svm_predict_test(self):
        cmd = 'svm-predict -b 1 ' + self.scaled_test_file + ' ' + self.model_save_file + ' ' + self.predict_save_file
        # 捕获命令行输出
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout
        # 提取准确率
        accuracy = None
        for line in output.split('\n'):
            if 'Accuracy' in line:
                try:
                    accuracy = float(line.split('=')[1].split('%')[0].strip())
                    print(f"测试集准确率: {accuracy}%")  # 打印准确率（可选）
                    break
                except (IndexError, ValueError):
                    pass

        # 原有的文件清理逻辑（保持不变）
        with open(self.predict_save_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            with open(self.predict_save_file, 'w') as f:
                f.writelines(lines[1:])
        return accuracy  # 返回准确率

    def svm_predict_initial_train(self):
        cmd = 'svm-predict -b 1 ' + self.scaled_initial_train_file + ' ' + self.model_save_file + ' ' + self.predict_initial_train_file
        # 捕获命令行输出
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout
        # 提取准确率
        accuracy = None
        for line in output.split('\n'):
            if 'Accuracy' in line:
                try:
                    accuracy = float(line.split('=')[1].split('%')[0].strip())
                    print(f"训练集准确率: {accuracy}%")  # 打印准确率（可选）
                    break
                except (IndexError, ValueError):
                    pass

        # 原有的文件清理逻辑（保持不变）
        with open(self.predict_initial_train_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            with open(self.predict_initial_train_file, 'w') as f:
                f.writelines(lines[1:])
        return accuracy  # 返回准确率

    def svm_predict_inter(self):
        cmd = 'svm-predict -b 1 ' + self.scaled_inter_file + ' ' + self.model_save_file + ' ' + self.predict_inter_save_file
        # 捕获命令行输出
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout
        # 提取准确率
        accuracy = None
        for line in output.split('\n'):
            if 'Accuracy' in line:
                try:
                    accuracy = float(line.split('=')[1].split('%')[0].strip())
                    print(f"中间集准确率: {accuracy}%")  # 打印准确率（可选）
                    break
                except (IndexError, ValueError):
                    pass

        # 原有的文件清理逻辑（保持不变）
        with open(self.predict_inter_save_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            with open(self.predict_inter_save_file, 'w') as f:
                f.writelines(lines[1:])

        return accuracy  # 返回

    def svm_predict_val(self):
        cmd = 'svm-predict -b 1 ' + self.scaled_val_file + ' ' + self.model_save_file + ' ' + self.predict_val_save_file
        # 捕获命令行输出
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout
        # 提取准确率
        accuracy = None
        for line in output.split('\n'):
            if 'Accuracy' in line:
                try:
                    accuracy = float(line.split('=')[1].split('%')[0].strip())
                    print(f"验证集集准确率: {accuracy}%")  # 打印准确率（可选）
                    break
                except (IndexError, ValueError):
                    pass

        # 原有的文件清理逻辑（保持不变）
        with open(self.predict_val_save_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            with open(self.predict_val_save_file, 'w') as f:
                f.writelines(lines[1:])

        return accuracy  # 返回

    def eval_perf(self, label, scores):
        if len(np.unique(label)) < 2:  # 如果标签只有一类，返回默认值
            return 0.5, 0.5, 0.5, 0.5

        fpr, tpr, thresholds = roc_curve(label, scores)
        pr, re, _ = precision_recall_curve(label, scores)

        # 计算 sn (tpr) 和 sp (1-fpr) 在阈值最接近 0.5 时的值
        idx = np.argmin(np.abs(thresholds - 0.5))
        sn = tpr[idx]
        sp = 1 - fpr[idx]

        auROC = auc(fpr, tpr)
        auPRC = auc(re, pr)

        return sn, sp, auROC, auPRC

    # def eval_perf(self, label, scores):
    #     fpr, tpr, thresholds = roc_curve(label, scores)
    #     pr, re, thresholds2 = precision_recall_curve(label, scores)
    #     for i in range(len(thresholds)):
    #         if thresholds[i] < 0.5:
    #             return tpr[i], 1 - fpr[i], auc(fpr, tpr), auc(re, pr)

    # def svm_evaluate(self):
    #     test_y_predict = np.loadtxt(self.predict_save_file, delimiter=' ')
    #     train_y_predict = np.loadtxt(self.predict_inter_save_file, delimiter=' ')
    #     # print(train_y_predict)
    #     print(test_y_predict)
    #     test_y = self.get_test_y()
    #     sn, sp, auROC, auPRC = self.eval_perf(test_y, test_y_predict[:, 1])
    #
    #     pre = precision_score(test_y, test_y_predict[:, 0])
    #     acc = accuracy_score(test_y, test_y_predict[:, 0])
    #     f1 = f1_score(test_y, test_y_predict[:, 0])
    #     mcc = matthews_corrcoef(test_y, test_y_predict[:, 0])
    #     auc_score = roc_auc_score(test_y, test_y_predict[:, 1])
    #
    #     print(sp, sn, pre, acc, f1, auROC, auc_score, mcc)
    #     result = (f"{sp} {sn} {pre} {acc} {f1} {auROC} {auc_score} {mcc} "
    #               f"-c {self.c} -g {self.g} -kb {self.kb}")
    #
    #     with open(self.results_save_file, 'a') as f:
    #         f.write(result)
    #         f.write('\n')
    #
    #     return train_y_predict[:, 1], test_y_predict[:, 1]
    def svm_evaluate(self):
        test_y_predict = np.loadtxt(self.predict_save_file, delimiter=' ')
        train_y_predict = np.loadtxt(self.predict_inter_save_file, delimiter=' ')

        test_y = self.get_test_y()
        unique_classes = np.unique(test_y)  # 获取真实标签中的类别

        # 初始化所有指标为 None（如果无法计算则保持 None）
        sp, sn, auROC, auPRC = None, None, None, None
        pre, acc, f1, mcc, auc_score = None, None, None, None, None

        # 只有数据包含至少两个类别时才计算相关指标
        if len(unique_classes) > 1:
            sn, sp, auROC, auPRC = self.eval_perf(test_y, test_y_predict[:, 1])

            # 计算分类指标，显式指定 labels 避免警告
            pre = precision_score(test_y, test_y_predict[:, 0], labels=unique_classes, zero_division=0)
            acc = accuracy_score(test_y, test_y_predict[:, 0])
            f1 = f1_score(test_y, test_y_predict[:, 0], labels=unique_classes, zero_division=0)
            mcc = matthews_corrcoef(test_y, test_y_predict[:, 0])

            # 计算 AUC，如果失败则设为 None
            try:
                auc_score = roc_auc_score(test_y, test_y_predict[:, 1])
            except ValueError:
                auc_score = None
        else:
            # 如果只有一种类别，至少计算准确率（其他指标无意义）
            acc = accuracy_score(test_y, test_y_predict[:, 0])

        print(sp, sn, pre, acc, f1, auROC, auc_score, mcc)

        # 格式化输出，None 值替换为 "NA"（或 0，视需求而定）
        result = (f"{sp if sp is not None else 'NA'} "
                  f"{sn if sn is not None else 'NA'} "
                  f"{pre if pre is not None else 'NA'} "
                  f"{acc if acc is not None else 'NA'} "
                  f"{f1 if f1 is not None else 'NA'} "
                  f"{auROC if auROC is not None else 'NA'} "
                  f"{auc_score if auc_score is not None else 'NA'} "
                  f"{mcc if mcc is not None else 'NA'} "
                  f"-c {self.c} -g {self.g} -kb {self.kb}")

        with open(self.results_save_file, 'a') as f:
            f.write(result)
            f.write('\n')

        return train_y_predict[:, 1], test_y_predict
