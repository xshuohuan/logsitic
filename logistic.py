import os
import numpy as np
from sklearn.linear_model import LogisticRegression

# =====================
# 1. 数据读取函数
# =====================
def load_dataset(file_path):
    """
    读取数据：最后一列是标签，其余是特征
    horseColic 数据一般是空格/Tab 分隔，可能会有缺失值
    """
    # 用 genfromtxt 更稳：遇到缺失会变成 nan
    data = np.genfromtxt(file_path, delimiter=None)  # delimiter=None 让它自动识别空格/Tab
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# =====================
# 2. 缺失值处理函数
#    （把 0 / nan 当缺失，替换为该列均值）
# =====================
def replace_nan_with_mean(X):
    X = X.copy()
    for i in range(X.shape[1]):
        col = X[:, i]

        # 先把 nan 处理掉
        valid_not_nan = col[~np.isnan(col)]

        # 再把 0 当缺失（horse colic 常用 0 表示缺失）
        valid = valid_not_nan[valid_not_nan != 0]

        if len(valid) > 0:
            mean_val = np.mean(valid)

            # 替换 nan
            col[np.isnan(col)] = mean_val
            # 替换 0
            col[col == 0] = mean_val

            X[:, i] = col
    return X

# =====================
# 3. 主流程：读取训练/测试
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "horseColicTraining.txt")
test_path  = os.path.join(BASE_DIR, "horseColicTest.txt")

X_train, y_train = load_dataset(train_path)
X_test,  y_test  = load_dataset(test_path)

X_train = replace_nan_with_mean(X_train)
X_test  = replace_nan_with_mean(X_test)

# 标签最好转成 int（有的文件可能是 0.0/1.0）
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("训练集：", X_train.shape, "测试集：", X_test.shape)

# =====================
# 4. 构建并训练逻辑回归模型
# =====================
# 数据量不大，liblinear 稳
clf = LogisticRegression(
    solver="liblinear",
    max_iter=2000
)

clf.fit(X_train, y_train)

# =====================
# 5. 测试集预测
# =====================
y_pred = clf.predict(X_test)

# =====================
# 6. 计算准确率
# =====================
acc = (y_pred == y_test).mean()
print("测试集准确率：", acc)

