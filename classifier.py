import pandas as pd
import numpy as np
import math
from preprocessor import extract_features_and_class
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score

NUM_FOLDS = 7
DATA_SET = 'isic_balanced'
PERCENTAGE_BENIGN = 0.5
PERCENTAGE_MALIGNANT = 1 - PERCENTAGE_BENIGN
IS_REDUCED = True

def get_data(dataset_file, reduced=False, shuffle=True):
    dataset_file += '_reduced_20.npy' if reduced else '.npy'
    print "Loading {}...".format(dataset_file)
    data_with_class = np.load(dataset_file)
    print data_with_class.shape
    return extract_features_and_class(get_split(data_with_class))

def train_classifier(data, classes):
    clf = ensemble.RandomForestClassifier(n_estimators=100) # 75%
    clf.fit(data, classes)
    return clf

def get_benign_training_prob(label_data):
    return int(math.floor(100 * \
            len(np.where(label_data == 0)[0]) / float(len(label_data))))

# Gets a shuffled split of benign to malignant images based on PERCENTAGE_BENIGN
def get_split(data):
    malignant = data[np.where(data[:,-1] == 1)]
    num_benign = int(math.ceil(PERCENTAGE_BENIGN * malignant.shape[0] / PERCENTAGE_MALIGNANT))
    benign = data[np.where(data[:,-1] == 0)]
    np.random.shuffle(benign)
    per_benign = benign[0:num_benign,:]
    split_data = np.vstack((per_benign, malignant))
    np.random.shuffle(split_data)
    return split_data

def main():
    # Load dataset as numpy arr
    X, Y = get_data(DATA_SET, reduced=IS_REDUCED)

    kf = KFold(n_splits = NUM_FOLDS, shuffle=True)

    print "Running K-folds for the classifier... K = {}".format(NUM_FOLDS)
    scores, precs, recalls = [], [], []
    for k, (train, test) in enumerate(kf.split(X, Y)):
        clf = train_classifier(X[train], Y[train])
        score = clf.score(X[test],Y[test])
        predictions =clf.predict(X[test]) 
        precision = precision_score(Y[test], predictions)
        recall = recall_score(Y[test], predictions)
        print "Score: {}%    Precision: {}%    Recall: {}%".\
                format(percentify(score), percentify(precision), percentify(recall))
        scores.append(score)
        precs.append(precision)
        recalls.append(recall)
        
    print "Mean of trials: accuracy {}% precision: {}% recall: {}%".format(percentify(np.mean(scores)), percentify(np.mean(precs)), percentify(np.mean(recalls)))
    print "Median of trials: {}%".format(percentify(np.mean(scores)))

def main_with_cnf():
    # Load dataset as numpy arr
    X, Y = get_data(DATA_SET, reduced=IS_REDUCED)

    kf = KFold(n_splits = NUM_FOLDS, shuffle=True)

    print "Running K-folds for the classifier... K = {}".format(NUM_FOLDS)
    scores, precs, recalls = [], [], []
    cnf = None
    for k, (train, test) in enumerate(kf.split(X, Y)):
        clf = train_classifier(X[train], Y[train])
        score = clf.score(X[test],Y[test])
        preds = clf.predict(X[test])
        cnf = confusion_matrix(Y[test], preds)
        predictions =clf.predict(X[test]) 
        precision = precision_score(Y[test], predictions)
        recall = recall_score(Y[test], predictions)
        print "Score: {}%    Precision: {}%    Recall: {}%".\
                format(percentify(score), percentify(precision), percentify(recall))
        scores.append(score)
        precs.append(precision)
        recalls.append(recall)
        plt.figure()
    plot_confusion_matrix(cnf, classes=['benign', 'malignant'], normalize=True, title='Confusion matrix with normalization')
    plt.show()

    print "Mean of trials: accuracy {}% precision: {}% recall: {}%".format(percentify(np.mean(scores)), percentify(np.mean(precs)), percentify(np.mean(recalls)))
    print "Median of trials: {}%".format(percentify(np.mean(scores)))

def percentify(decimal):
    return int(math.floor(100 * decimal))

def main_with_auc():
    # Load dataset as numpy arr
    X, Y = get_data(DATA_SET, reduced=IS_REDUCED)

    kf = KFold(n_splits = NUM_FOLDS, shuffle=True)

    print "Running K-folds for the classifier... K = {}".format(NUM_FOLDS)
    scores,tprs,aucs = [],[],[]
    mean_fpr = np.linspace(0, 1, 100)
    for k, (train, test) in enumerate(kf.split(X, Y)):
        clf = train_classifier(X[train], Y[train])
        score = clf.score(X[test],Y[test])
        probs = clf.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], probs[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc))
        #score = clf.score(X[test],Y[test])
        #print "Score: {}%".format(int(math.floor(100 * score)))

        scores.append(score)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False-positive rate (unnecessary biopsy)')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operator Characteristic for {} Folds'.format(NUM_FOLDS))
    plt.legend(loc="lower right")
    plt.show()

    print "Mean of trials: {}".format(np.mean(scores))
    print "Median of trials: {}".format(np.mean(scores))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    #main()
    #main_with_auc()
    main_with_cnf()
