import numpy as np 
from .constants import ORDER
import pandas as pd

def confusion_matrix(y_true, y_pred, classes = ORDER) -> np.array:
    classes = np.array(classes)
    n_classes = len(classes)
    
    confusion_matrix = np.zeros((n_classes, n_classes))

    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        true_index = np.where(classes == true_class)[0][0]
        pred_index = np.where(classes == pred_class)[0][0]
        confusion_matrix[true_index][pred_index] += 1

    return confusion_matrix

def classification_report(cm, classes = ORDER) -> None:
    TP = np.diag(cm)
    TN = np.sum(cm) - (np.sum(TP) + np.sum(cm.diagonal(-1)))
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    accuracy = np.sum(TP) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(pd.DataFrame({
            "classes": classes,
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score
        }).set_index("classes"), "\n\naccuracy: ", np.round(accuracy, 3))

