from sklearn.metrics import f1_score, recall_score, precision_score



'''
Evaluates the model according to the precision score, recall and f1-score
Args :
    - model : the evaluated model
    - X_test : features of the testing set
    - y_test : labels of the testing set
Returns : 
    list containing the f1-score, recall and precision
'''
def mlModel_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    print(f1, recall, precision)
    return [f1, recall, precision]


'''
Evaluates the model according to the accuracy score, recall and f1-score
Args :
    - X_test : features of the testing set
    - y_test : labels of the testing set
    - model : the evaluated model
Returns : 
    list containing the f1-score, recall and accuracy score 
'''
def dlModel_evaluation(X_test,y_test,model):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_test = y_test.argmax(axis=1)
    y_pred = model.predict(X_test).argmax(axis=1)
    recall = recall_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='micro')  
    print(F1_score, recall, accuracy)  
    return [F1_score, recall, accuracy]