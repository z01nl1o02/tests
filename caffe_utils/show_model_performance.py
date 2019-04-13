import os,sys
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
import numpy as np
import copy,pdb
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def load_raw(input):
    num_classes = -1
    ys_true = []
    ys_pred = []
    scores_pred = []
    name_classes = []
    with open(input,'r') as f:
        for line in f:
            data = line.strip().split(' ')
            _,y_true = data[0],int(float(data[1]))
            scores = [float(x) for x in data[2:]]
            if num_classes < 0:
                num_classes = len(scores)
                for k in range(num_classes):
                    name_classes.append( k )
            if num_classes != len(scores):
                print("output number of score should be same")
                continue
            y_pred = np.argmax(scores)
            ys_true.append(y_true)
            ys_pred.append(y_pred)
            scores_pred.append( scores )
    return ys_true,ys_pred,scores_pred,name_classes


    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax    
    
if __name__=="__main__":
    ys_true,ys_pred,scores_pred,name_classes = load_raw(sys.argv[1])
    print(classification_report(ys_true, ys_pred))
    #print('confusion_matrix')
    #print(confusion_matrix(ys_true,ys_pred))
    plot_confusion_matrix(ys_true, ys_pred,name_classes)
    
    plt.show()
    
    
    

            
            
