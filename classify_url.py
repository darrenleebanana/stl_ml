#importing packages
#importing packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

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
svm = False
rf = True

'''----------------------- Start of DT ---------------------'''
if dt: 
    # Decision Tree model
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
    '''-----------Selecting hyperparameters----------'''
    if False:
        C_values = [0.01, 0.1, 1, 10, 100]
        kernels = ['linear', 'rbf']
        results = []

        for kernel in kernels:
            for C in C_values:
                svm = SVC(C=C, kernel=kernel, random_state=16)
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

            '''
            Kernel   C      Accuracy   Precision  Recall     F1        
            linear   0.01   0.940       0.971     0.904     0.936
            linear   0.1    0.957       1.000     0.911     0.953
            linear   1      0.970       1.000     0.938     0.968 ** linear, C=1.0, random_state=16 **
            linear   10     0.957       0.952     0.959     0.956
            linear   100    0.957       0.952     0.959     0.956
            rbf      0.01   0.880       0.910     0.836     0.871
            rbf      0.1    0.920       0.969     0.863     0.913
            rbf      1      0.947       0.951     0.938     0.945
            rbf      10     0.960       0.972     0.945     0.958
            rbf      100    0.957       0.952     0.959     0.956
            '''

    # --- Train Linear SVM ---
    svm = SVC(kernel='linear', C=1, random_state=16)
    svm.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = svm.predict(X_test)

    # --- Standard metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Linear SVM (C=1) Results:")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- Confusion Matrix (values + heatmap) ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    # Derived rates
    instances = len(y_test)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # AUC using decision scores (preferred for SVM without probability=True)
    y_scores = svm.decision_function(X_test)
    auc = roc_auc_score(y_test, y_scores)

    print("\nAdditional Metrics:")
    print(f"False Positive Rate:\t{fpr:.3f}")
    print(f"False Negative Rate:\t{fnr:.3f}")
    print(f"AUC:\t\t\t{auc:.3f}")

    # Plot heatmap
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Linear SVM (C=1)")
    plt.show()

    # --- Feature Weights (coefficients) ---
    coef = svm.coef_.ravel()
    plt.figure(figsize=(9,7))
    plt.barh(range(len(coef)), coef, align='center')
    plt.yticks(np.arange(len(coef)), X_train.columns)
    plt.xlabel("Coefficient magnitude")
    plt.ylabel("Feature")
    plt.title("SVM (Linear, C=1) Feature Weights")
    plt.savefig("svm-linear-feature-importance.png")
    plt.show()


if rf:
    # random forest setup and training
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=16,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # predictions on test data
    y_pred_rf = rf.predict(X_test)

    # basic performance scores
    acc = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf, zero_division=0)
    rec = recall_score(y_test, y_pred_rf, zero_division=0)
    f1 = f1_score(y_test, y_pred_rf, zero_division=0)

    print("random forest results:")
    print(f"accuracy : {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall   : {rec:.3f}")
    print(f"f1 score : {f1:.3f}\n")

    print("classification report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    # confusion matrix values
    cm = confusion_matrix(y_test, y_pred_rf)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    # rates and auc
    instances = len(y_test)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob_rf)

    print("\nadditional metrics:")
    print(f"false positive rate:\t{fpr:.3f}")
    print(f"false negative rate:\t{fnr:.3f}")
    print(f"auc:\t\t\t{auc:.3f}")

    # confusion matrix plot
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_).plot(
        cmap="Blues", values_format="d"
    )
    plt.title("confusion matrix - random forest")
    plt.show()

    # feature importance chart
    importances = rf.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(9,7))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), X_train.columns[indices])
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.title("random forest feature importances")
    plt.tight_layout()
    plt.savefig("random-forest-feature-importance.png")
    plt.show()