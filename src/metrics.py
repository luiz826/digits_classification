import numpy as np 

def confusion_matrix(y_true, y_pred, class_1 = -1, class_2 = 1):
    VP = 0
    VN = 0
    FP = 0
    FN = 0
    
    for i in range(len(y_pred)):
        if (y_true[i] == y_pred[i]):
            if y_pred[i] == class_1:
                VP += 1
            else: # y_pred[i] == class_2:
                VN += 1
        else:
            if y_pred[i] == class_1:
                FP += 1
            else: # y_pred[i] == class_2:
                FN += 1            
    
    return np.array([[VP, FP],[FN, VN]])

def classification_report(y_true, y_pred, class_1 = -1, class_2 = 1):
    mc = confusion_matrix(y_true, y_pred, class_1, class_2)
    
    sup_c1 = 0
    sup_c2 = 0
    for i in y_pred:
        if i == class_1:
            sup_c1 += 1
        else:
            sup_c2 += 1
    
    
    VP = mc[0, 0] 
    FP = mc[0, 1] 
    FN = mc[1, 0] 
    VN = mc[1, 1] 
    
    accuracy = (VP + VN) / (VP + VN + FP + FN)
    
    if (VP+FP) != 0:
        precision_c1 = VP/(VP+FP)
    else:
        precision_c1 = 0
    
    if (VN+FN) != 0:
        precision_c2 = VN/(VN+FN)
    else:
        precision_c2 = 0 
        
    if (VP+FN) != 0:
        recall_c1 = VP/(VP+FN)
    else:
        recall_c1 = 0
    
    if (VN+FP) != 0:
        recall_c2 = VN/(VN+FP)
    else:
        recall_c2 = 0
        
    if (precision_c1 + recall_c1) != 0:
        f1_score_c1 = (2*precision_c1*recall_c1) / (precision_c1 + recall_c1) 
    else:
        f1_score_c1 = 0
    
    if (precision_c2 + recall_c2) != 0:
        f1_score_c2 = (2*precision_c2*recall_c2) / (precision_c2 + recall_c2)
    else:
        f1_score_c2 = 0
    
    return f"              precision    recall  f1-score   support\n\n" + \
           f"          {class_1}       {precision_c1:.2f}      {recall_c1:.2f}" + \
           f"      {f1_score_c1:.2f}       {sup_c1}\n" + \
           f"           {class_2}       {precision_c2:.2f}      {recall_c2:.2f}" + \
           f"      {f1_score_c2:.2f}        {sup_c2}\n\n" + \
           f"    accuracy                           {accuracy:.2f}       {len(y_pred)}"

def confusion_matrix_all(y_true, y_pred, real_class, class_1 = 1, class_2 = -1):
    pred = np.zeros(len(y_pred)) 
    pred[y_pred == real_class] = class_1
    pred[y_pred != real_class] = class_2
    
    true = np.zeros(len(y_true)) 
    true[y_true == real_class] = class_1
    true[y_true != real_class] = class_2
    
    VP = 0
    VN = 0
    FP = 0
    FN = 0
    
    for i in range(len(pred)):
        if (true[i] == pred[i]):
            if pred[i] == class_1:
                VP += 1
            else: # y_pred[i] == class_2:
                VN += 1
        else:
            if pred[i] == class_1:
                FP += 1
            else: # y_pred[i] == class_2:
                FN += 1            
    
    return np.array([[VP, FP],[FN, VN]])

def classification_report_all(y_true, y_pred, real_class, print_mc = True, class_1 = 1, class_2 = -1):
    mc = confusion_matrix_all(y_true, y_pred, real_class, class_1, class_2)
    
    if print_mc:
        print("Matriz de Confusão: ", mc)

    sup_c1 = 0
    sup_c2 = 0
    for i in y_pred:
        if i == class_1:
            sup_c1 += 1
        else:
            sup_c2 += 1
    
    
    VP = mc[0, 0] 
    FP = mc[0, 1] 
    FN = mc[1, 0] 
    VN = mc[1, 1] 
    
    accuracy = (VP + VN) / (VP + VN + FP + FN)
    
    if (VP+FP) != 0:
        precision_c1 = VP/(VP+FP)
    else:
        precision_c1 = 0
    
    if (VN+FN) != 0:
        precision_c2 = VN/(VN+FN)
    else:
        precision_c2 = 0 
        
    if (VP+FN) != 0:
        recall_c1 = VP/(VP+FN)
    else:
        recall_c1 = 0
    
    if (VN+FP) != 0:
        recall_c2 = VN/(VN+FP)
    else:
        recall_c2 = 0
        
    if (precision_c1 + recall_c1) != 0:
        f1_score_c1 = (2*precision_c1*recall_c1) / (precision_c1 + recall_c1) 
    else:
        f1_score_c1 = 0
    
    if (precision_c2 + recall_c2) != 0:
        f1_score_c2 = (2*precision_c2*recall_c2) / (precision_c2 + recall_c2)
    else:
        f1_score_c2 = 0
    
    return f"              precision    recall  f1-score   support\n\n" + \
           f"             {class_1}       {precision_c1:.2f}      {recall_c1:.2f}" + \
           f"      {f1_score_c1:.2f}       {sup_c1}\n" + \
           f"           {class_2}       {precision_c2:.2f}      {recall_c2:.2f}" + \
           f"      {f1_score_c2:.2f}        {sup_c2}\n\n" + \
           f"    accuracy                           {accuracy:.2f}       {len(y_pred)}"
