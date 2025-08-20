#importing packages
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

filename = os.path.join("datasets", "urldata.csv")
data0 = pd.read_csv(filename)
data = data0.drop(['Domain'], axis = 1).copy()

fn = ['Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
            'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
            'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
cn =['Good','Bad']

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)

# Separating & assigning features and target columns to X & y
y = data['Label']
X = data.drop('Label',axis=1)

# Splitting the dataset into train and test sets: 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 12)

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []
dt = False
mlp = False
svm = True

'''----------------------- Start of DT ---------------------'''
if dt: 
    # Decision Tree model 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    # instantiate the model 
    dt = DecisionTreeClassifier(max_depth = 5)
    # fit the model 
    dt.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_tree = dt.predict(X_test)
    y_train_tree = dt.predict(X_train)

    # checking the feature importance in the model
    plt.figure(figsize=(9,7))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), dt.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig('decisiontree-feature-importance.png')
    plt.show()

    #tree visualization
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)

    tree.plot_tree(dt, feature_names = fn, class_names=cn, filled = True)
    fig.savefig('decision-tree.png')

    # print metrics
    cm = confusion_matrix(y_test, y_test_tree)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    instances = len(y_test)
    accuracy = (tp+tn)/instances
    fpr = fp/(tn+fp)
    fnr = fn/(tp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    y_prob_tree = dt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob_tree)
    print("\nMetrics of Decision Tree Model:")
    print(f"Accuracy:\t\t{accuracy:.3%}")
    print(f"False Positive Rate:\t{fpr:.3f}")
    print(f"False Negative Rate:\t{fnr:.3f}")
    print(f"Precision:\t\t{precision:.3f}")
    print(f"Recall:\t\t\t{recall:.3f}")
    print(f"AUC:\t\t\t{auc:.3f}")

'''------------------------ End of DT ----------------------'''

'''---------------------- Start of MLP ---------------------'''
if mlp:
    from sklearn.neural_network import MLPClassifier

    # instantiate the model 
    mlp = MLPClassifier(hidden_layer_sizes=(50,), solver='lbfgs', alpha=0.01, max_iter=500)
    # fit the model 
    mlp.fit(X_train, y_train)
    # predicting the target value from the model for the samples
    y_test_mlp = mlp.predict(X_test)
    y_train_mlp = mlp.predict(X_train)

    # print metrics
    cm = confusion_matrix(y_test, y_test_mlp)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    instances = len(y_test)
    accuracy = (tp+tn)/instances
    fpr = fp/(tn+fp)
    fnr = fn/(tp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    y_prob_mlp = dt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob_mlp)
    print("\nMetrics of MLP Model:")
    print(f"Accuracy:\t\t{accuracy:.3%}")
    print(f"False Positive Rate:\t{fpr:.3f}")
    print(f"False Negative Rate:\t{fnr:.3f}")
    print(f"Precision:\t\t{precision:.3f}")
    print(f"Recall:\t\t\t{recall:.3f}")
    print(f"AUC:\t\t\t{auc:.3f}")

'''------------------------ End of MLP ---------------------'''
if svm:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    C_values = [0.01, 0.1, 1, 10, 100]
    kernels = ['linear', 'rbf']

    results = []

    for kernel in kernels:
        for C in C_values:
            svm = SVC(C=C, kernel=kernel, random_state=12)
            svm.fit(X_train, y_train)
            
            # Predictions
            y_pred = svm.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results.append((kernel, C, acc, prec, rec, f1))

    # Print results as table
    print("{:<8} {:<6} {:<10} {:<10} {:<10} {:<10}".format("Kernel", "C", "Accuracy", "Precision", "Recall", "F1"))
    for kernel, C, acc, prec, rec, f1 in results:
        print("{:<8} {:<6} {:.3f}     {:.3f}     {:.3f}     {:.3f}".format(kernel, C, acc, prec, rec, f1))

