import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, PrecisionRecallDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def impute_missing(df):
    """Return Test DataFrame that has its missing values imputed in the exact same way the Train DataFrame is imputed."""

    no_nans = df.dropna(axis=0)
    knn = KNeighborsClassifier(weights='distance')

    # fit it to the dataframe with no missing education
    knn.fit(no_nans[['age', 'length_of_service']], no_nans['education'])

    # find the missing rows in education
    missing_edu = df['education'].isnull()

    # extract the rows with missing education
    df_missing_edu = df[['age', 'length_of_service']][missing_edu]

    # used the train knn to predict missing education and impute it to the dataframe
    imputed_edu = knn.predict(df_missing_edu)
    df.loc[missing_edu, 'education'] = imputed_edu

    # impute missing previous_year_rating with 0s
    df['previous_year_rating'] = df['previous_year_rating'].replace(np.nan, 0)

    return df


def run_model(model, X_train, y_train, X_test, y_test):
    """Run a model, fit it, make predictions and produce relevant metrics visualization"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    sns.set_style('white')
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    print(classification_report(y_test, preds))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)


def plot_feature_importance(features, model):
    """Plot a feature importance of a tree-based model"""
    # extract the feature importance array from the estimator
    feat_importance = model.feature_importances_

    # create a df matching the feature names with the importance
    feat_df = pd.DataFrame(index=features.columns, data=feat_importance, columns=['Importance'])

    # sort the dataframe and plot barplot
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6), dpi=100)
    sns.barplot(x=feat_df.index, y=feat_df.Importance)
    plt.xticks(rotation=90)
    plt.xlabel('Feature Names', fontsize=10)
    plt.ylabel('Feature Importance', fontsize=10)


def error_rate(model, X_train, y_train, X_test, y_test):
    """Return the error rate of a model based on accuracy score"""

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    error = 1 - accuracy_score(y_test, preds)

    return error


def plot_error_rates(errors, value_range):
    """Plot error rates based on list of errors produced by for-looping the error_rate() function above"""

    error_rates = []
    for err in errors:
        error_rates.append(err)
    
    plt.plot(value_range, error_rates)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rates')


def threshold_predict(model, threshold, X_train, X_test, y_train):
    """Return the prediction of a model given by a threshold set manually."""
    model.fit(X_train, y_train)
    pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool)

    return pred


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot the ROC Curves of all the fitted models in the dict parameter."""
    fig, ax= plt.subplots(dpi=150)
    fig.suptitle('ROC Curves')
    fig.set_figheight(6)
    fig.set_figwidth(10)
    # plot the default line
    plt.plot([0, 1], [0, 1], linestyle='--', c='black');

    # plot the ROC curves
    for name, estimator in models.items():
        RocCurveDisplay.from_estimator(estimator, X_test, y_test, name="{}".format(name), ax=ax);


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot the Confusion Matrix for each fitted model in the dict parameter"""

    fig, ax = plt.subplots(ncols=len(models), dpi=200)
    fig.set_figheight(10)
    fig.set_figwidth(12)

    for (name, estimator), order in zip(models.items(), range(len(models))):
        ConfusionMatrixDisplay.from_estimator(models[name], X_test, y_test, ax=ax[order], colorbar=False)
        ax[order].set_title(name)
    
    fig.tight_layout()


def classification_reports(models, X_test, y_test):
    """Print all classification reports for all the fitted models. All the models in the dictionary are expected to accept non-scaled features"""

    for name, estimator in models.items():
        y_pred = estimator.predict(X_test)
        print(f"Classification Report for: {name}")
        print(classification_report(y_test, y_pred))


def plot_precision_recall_disp(models: dict, X_test, y_test):
    """Plot the Confusion Matrix for each fitted model in the dict parameter"""

    fig, ax = plt.subplots(nrows=len(models), dpi=200)
    fig.suptitle('Precision Recall Display')
    fig.set_figheight(10)
    fig.set_figwidth(6)

    for (name, estimator), order in zip(models.items(), range(len(models))):
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax[order])
        ax[order].set_title(name)
    
    fig.tight_layout()

