import numpy as np 
from .constants import ORDER
import pandas as pd

# def confusion_matrix(y_true, y_pred, class_1 = -1, class_2 = 1):
#     VP = 0
#     VN = 0
#     FP = 0
#     FN = 0
    
#     for i in range(len(y_pred)):
#         if (y_true[i] == y_pred[i]):
#             if y_pred[i] == class_1:
#                 VP += 1
#             else: # y_pred[i] == class_2:
#                 VN += 1
#         else:
#             if y_pred[i] == class_1:
#                 FP += 1
#             else: # y_pred[i] == class_2:
#                 FN += 1            
    
#     return np.array([[VP, FP],[FN, VN]])

# def classification_report(y_true, y_pred, class_1 = -1, class_2 = 1):
#     mc = confusion_matrix(y_true, y_pred, class_1, class_2)
    
#     sup_c1 = 0
#     sup_c2 = 0
#     for i in y_pred:
#         if i == class_1:
#             sup_c1 += 1
#         else:
#             sup_c2 += 1
    
    
#     VP = mc[0, 0] 
#     FP = mc[0, 1] 
#     FN = mc[1, 0] 
#     VN = mc[1, 1] 
    
#     accuracy = (VP + VN) / (VP + VN + FP + FN)
    
#     if (VP+FP) != 0:
#         precision_c1 = VP/(VP+FP)
#     else:
#         precision_c1 = 0
    
#     if (VN+FN) != 0:
#         precision_c2 = VN/(VN+FN)
#     else:
#         precision_c2 = 0 
        
#     if (VP+FN) != 0:
#         recall_c1 = VP/(VP+FN)
#     else:
#         recall_c1 = 0
    
#     if (VN+FP) != 0:
#         recall_c2 = VN/(VN+FP)
#     else:
#         recall_c2 = 0
        
#     if (precision_c1 + recall_c1) != 0:
#         f1_score_c1 = (2*precision_c1*recall_c1) / (precision_c1 + recall_c1) 
#     else:
#         f1_score_c1 = 0
    
#     if (precision_c2 + recall_c2) != 0:
#         f1_score_c2 = (2*precision_c2*recall_c2) / (precision_c2 + recall_c2)
#     else:
#         f1_score_c2 = 0
    
#     return f"              precision    recall  f1-score   support\n\n" + \
#            f"          {class_1}       {precision_c1:.2f}      {recall_c1:.2f}" + \
#            f"      {f1_score_c1:.2f}       {sup_c1}\n" + \
#            f"           {class_2}       {precision_c2:.2f}      {recall_c2:.2f}" + \
#            f"      {f1_score_c2:.2f}        {sup_c2}\n\n" + \
#            f"    accuracy                           {accuracy:.2f}       {len(y_pred)}"

def confusion_matrix(y_true, y_pred, classes = ORDER):
#     classes = ORDER
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

def classification_report(cm):
    # calcular as métricas de avaliação a partir da matriz de confusão
    TP = np.diag(cm)
    TN = np.sum(cm) - (np.sum(TP) + np.sum(cm.diagonal(-1)))
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # imprimir as métricas de avaliação
    return pd.DataFrame({
        "accuracy": np.sum(accuracy),
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score
    })
