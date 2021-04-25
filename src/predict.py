import numpy as np
import pandas
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import datetime
from src import load_data
import os

# This file takes in the feature-engineered CSV and will train and test
# multiple models

SOLUTION_PATH = os.path.join(os.getcwd(), os.pardir, "solution")


def train_and_score(training_data_df, truth_df):
    truth_training_df = pandas.merge(training_data_df, truth_df,
                                     how='inner',
                                     left_on=['id1', 'id2'],
                                     right_on=['ltable_id', 'rtable_id'])

    total_not_present = 0
    total_zeroed = 0
    true_labels_to_append = np.empty((0, 0))
    for idx, row in truth_df.iterrows():
        l_id = row['ltable_id']
        r_id = row['rtable_id']
        sample_exists = ((training_data_df['id1'] == l_id) & (training_data_df['id2'] == r_id)).any()
        label = row['label']
        if not sample_exists:
            total_not_present += 1
            true_labels_to_append = np.append(true_labels_to_append, label)
        if not sample_exists and label == 0:
            total_zeroed += 1

    pred_labels_to_append = np.zeros(true_labels_to_append.shape)
    print(total_not_present)
    print(total_zeroed)



    truth_training_df.drop(['id1', 'id2', 'ltable_id', 'rtable_id'], axis=1, inplace=True)
    train, test = train_test_split(truth_training_df, test_size=0.2)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = train.drop('label', axis=1)
    y_train = train.loc[:, 'label']
    y_train.reset_index(drop=True, inplace=True)
    x_test = test.drop('label', axis=1)
    y_test = test.loc[:, 'label']
    y_test.reset_index(drop=True, inplace=True)
    y_test = y_test.append(pandas.Series(true_labels_to_append), ignore_index=True)
    imp = imp.fit(x_train)

    # Impute our data, then train
    x_train_imp = imp.transform(x_train)
    scaler = StandardScaler().fit(x_train_imp)
    x_train_scale_imp = scaler.transform(x_train_imp)
    svm = SVC(kernel='rbf', C=1.0)
    rf = RandomForestClassifier(n_estimators=5000)
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()

    svm = svm.fit(x_train_scale_imp, y_train)
    rf = rf.fit(x_train_scale_imp, y_train)
    lda = lda.fit(x_train_scale_imp, y_train)
    qda = qda.fit(x_train_scale_imp, y_train)

    score_trained_classifier('SVM', svm, imp, scaler, x_test, y_test, pred_labels_to_append)
    score_trained_classifier('Random Forest', rf, imp, scaler, x_test, y_test, pred_labels_to_append)
    score_trained_classifier('LDA', lda, imp, scaler, x_test, y_test, pred_labels_to_append)
    score_trained_classifier('QDA', qda, imp, scaler, x_test, y_test, pred_labels_to_append)


def score_trained_classifier(name, classifier, imputer, scaler, x_test, y_test, pred_labels_to_append):
    print("================================")
    print(name)
    print("================================")
    x_test_imp = imputer.transform(x_test)
    x_test_scale_imp = scaler.transform(x_test_imp)
    y_pred = pandas.Series(classifier.predict(x_test_scale_imp))
    y_pred = y_pred.append(pandas.Series(pred_labels_to_append), ignore_index=True)


    print('Precision: {}'.format(precision_score(y_test, y_pred)))
    print('Recall: {}'.format(recall_score(y_test, y_pred)))
    print('F1: {}'.format(f1_score(y_test, y_pred)))
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
    print('Num Positives Real: {}'.format(y_test.value_counts()[1]))
    print('Num Positives Pred: {}'.format(y_pred.value_counts()[1]))
    print("\n")


def write_final_output(training_data_df, truth_df):

    truth_training_df = pandas.merge(training_data_df, truth_df,
                                     how='inner',
                                     left_on=['id1', 'id2'],
                                     right_on=['ltable_id', 'rtable_id'])
    truth_training_df.drop(['id1', 'id2', 'ltable_id', 'rtable_id'], axis=1,
                           inplace=True)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = truth_training_df.drop('label', axis=1)
    y_train = truth_training_df.loc[:, 'label']
    y_train.reset_index(drop=True, inplace=True)
    imp = imp.fit(x_train)

    # Impute our data, then train
    x_train_imp = imp.transform(x_train)
    scaler = StandardScaler().fit(x_train_imp)
    x_train_scale_imp = scaler.transform(x_train_imp)
    rf = RandomForestClassifier(n_estimators=5000)

    rf = rf.fit(x_train_scale_imp, y_train)

    df_to_predict = training_data_df.copy(deep=True)
    df_to_predict.drop(['id1', 'id2'], inplace=True, axis=1)

    df_to_predict_imp_scale = scaler.transform(imp.transform(df_to_predict))

    predictions = pandas.DataFrame(rf.predict(df_to_predict_imp_scale))

    indexes = predictions[predictions[0] == 1].index.values
    final_df = training_data_df.loc[indexes, ['id1', 'id2']]
    final_df.columns = ['ltable_id', 'rtable_id']
    final_df.dropna(axis=0, inplace=True)
    final_df = final_df.astype('int32')

    for idx, row in truth_df.iterrows():
        l_id = row['ltable_id']
        r_id = row['rtable_id']
        final_df = final_df[(final_df['ltable_id'] != l_id) |
                            (final_df['rtable_id'] != r_id)]

    final_df.to_csv(
        os.path.join(SOLUTION_PATH, str(datetime.datetime.now())),
        index=False)


if __name__ == '__main__':
    true_df = pandas.read_csv(load_data.TRUTH_PATH)
    train_df = pandas.read_csv(load_data.TRAIN_PATH_COS_UPDATE, index_col=0)

    should_output_scores = True
    should_write_final_output = True

    if should_output_scores:
        train_and_score(train_df, true_df)
    if should_write_final_output:
        write_final_output(train_df, true_df)