import numpy as np
from functions import calculate_features, training, show_scatter, bayes_test

# True to show scatter of features for three classes (6, 8, 9), False otherwise
SHOW_SCATTER = True
SHUFFLE = False
# total number of images for each digit
N = 120
# training set: 100 images for each digit
Nt = 100

(feat_6, feat_8, feat_9) = calculate_features(N)

if SHUFFLE:
    feat_6_h = feat_6.transpose()
    feat_8_h = feat_8.transpose()
    feat_9_h = feat_9.transpose()
    # Shuffle the features
    np.random.shuffle(feat_6_h)
    np.random.shuffle(feat_8_h)
    np.random.shuffle(feat_9_h)

    feat_6 = feat_6_h.transpose()
    feat_8 = feat_8_h.transpose()
    feat_9 = feat_9_h.transpose()

training_set_6 = feat_6[:, 0:Nt]
training_set_8 = feat_8[:, 0:Nt]
training_set_9 = feat_9[:, 0:Nt]

if SHOW_SCATTER:
    show_scatter(training_set_6, training_set_8, training_set_9)

M1, S1, M2, S2, M3, S3 = training(training_set_6, training_set_8, training_set_9)

test_set_6 = feat_6[:, Nt:N]
test_set_8 = feat_8[:, Nt:N]
test_set_9 = feat_9[:, Nt:N]
# Test using Bayesian Hypothesis Testing and calculate confusion matrix
conf_mat = bayes_test(test_set_6, test_set_8, test_set_9, M1, S1, M2, S2, M3, S3)

print("Confusion matrix: \n")
print(conf_mat)
