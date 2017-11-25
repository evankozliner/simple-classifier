import pandas as pd
import numpy as np
import math
from preprocessor import extract_features_and_class
from sklearn import ensemble
from sklearn.model_selection import KFold

NUM_FOLDS = 7
DATA_SET = 'isic_balanced'
PERCENTAGE_BENIGN = 0.6
PERCENTAGE_MALIGNANT = 1 - PERCENTAGE_BENIGN
IS_REDUCED = True

def get_data(dataset_file, reduced=False, shuffle=True):
    dataset_file += '_reduced_20.npy' if reduced else '.npy'
    print "Loading {}...".format(dataset_file)
    data_with_class = np.load(dataset_file)
    return extract_features_and_class(get_split(data_with_class))

def train_classifier(data, classes):
    #clf = linear_model.LogisticRegression() # 
    clf = ensemble.RandomForestClassifier(n_estimators=100) # 75%
    #clf = ensemble.AdaBoostClassifier(n_estimators=100) # 70%
    #clf = discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    #clf = svm.LinearSVC() # 60%
    #clf = svm.SVC() # 60%
    #clf = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariances=True)
    #clf = neighbors.NearestCentroid() # 65%
    clf.fit(data, classes)
    return clf

def get_benign_training_prob(label_data):
    return int(math.floor(100 * \
            len(np.where(label_data == 0)[0]) / float(len(label_data))))

# Gets a shuffled 60/40 split of benign to malignant images 
def get_split(data):
    malignant = data[np.where(data[:,-1] == 1)]
    num_benign = int(math.ceil(PERCENTAGE_BENIGN * malignant.shape[0] / PERCENTAGE_MALIGNANT))
    print num_benign
    benign = data[np.where(data[:,-1] == 0)]
    np.random.shuffle(benign)
    per_benign = benign[0:num_benign,:]
    split_data = np.vstack((per_benign, malignant))
    np.random.shuffle(split_data)
    return split_data

def main():
    # Load dataset as numpy arr
    X, Y = get_data(DATA_SET, reduced=IS_REDUCED)
    #X_test, Y_test = get_data('test', reduced=True)
    #X_train, Y_train = get_data('train', reduced=True)
    #X = np.vstack((X_test, X_train))
    #Y = np.vstack((Y_test, Y_train))

    kf = KFold(n_splits = NUM_FOLDS, shuffle=True)

    print "Running K-folds for the classifier... K = {}".format(NUM_FOLDS)
    scores = []
    for k, (train, test) in enumerate(kf.split(X, Y)):
        #print("TRAIN:", train_index, "TEST:", test_index)
        #X_train, X_test = X[train_index], X[test_index]
        #Y_train, Y_test = Y[train_index], Y[test_index]
        clf = train_classifier(X[train], Y[train])
        score = clf.score(X[test],Y[test])
        print "Probability of benign in training set: {}%".format(\
                get_benign_training_prob(Y[train]))
        print "Probability of benign in test set: {}%".format(\
                get_benign_training_prob(Y[test]))
        print "Score: {}%".format(int(math.floor(100 * score)))
        scores.append(score)
    print "Mean of trials: {}".format(np.mean(scores))
    print "Median of trials: {}".format(np.mean(scores))

if __name__ == "__main__":
    main()
