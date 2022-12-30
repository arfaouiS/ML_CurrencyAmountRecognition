from sklearn.metrics import f1_score, recall_score, precision_score

def mlModel_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print(f1, recall, precision)
    return [f1, recall, precision]