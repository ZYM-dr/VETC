import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_excel("172lasso.xlsx",sheet_name='Sheet4')
#####获取预测目标列和特征集
y = df['Y'].values
X = df.drop("Y",axis=1)
from sklearn.model_selection import train_test_split
####按照比例将数据集分为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=0)

###初始化CAT树模型
dtc = DecisionTreeClassifier(criterion='gini')

###训练模型
dtc.fit(X_train,y_train)

###预测结果
y_pred = dtc.predict(X_test)
y_tpred = dtc.predict(X_train)


####计算指标
def calculate_metrics(gt, pred,label):
    confusion = confusion_matrix(gt, pred)
    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print(f'{label}Sensitivity:', TP / float(TP + FN))
    print(f'{label}Specificity:', TN / float(TN + FP))
    print(f'{label}PPV:',TP / float(TP + FP))
    print(f"{label}NPV",TN/float(TN+FN))
calculate_metrics(y_test,y_pred,'测试集')
calculate_metrics(y_train,y_tpred,'训练集')
