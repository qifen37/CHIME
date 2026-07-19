from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def multi_class_eval(labels, pred):
    pred = [sample.argmax() for sample in pred]
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    precision = precision_score(labels, pred, average='macro', zero_division=0)
    recall = recall_score(labels, pred, average='macro', zero_division=0)

    return acc * 100, f1 * 100, precision * 100, recall * 100
