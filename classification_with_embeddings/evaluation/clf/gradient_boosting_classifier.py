from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Note that currently the gradient boosting classifier provided by sklearn package is used.
class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params: Optional[dict] = None, objective='binary:logistic', n_rounds=100, num_class=-1):
        """Gradient boostign classifier implementation

        :param params: model parameters (use default pre-set values if None)
        :param objective:  prediction objective (only 'binary:*' and 'multi:softprob' are tested)
        :param n_rounds: number of training rounds
        :param num_class: number of different classes if performing multi-class classification
        """

        self.num_class = num_class

        self.classes_ = None  # initialized inside fit method

        # set parameters
        if params is None:
            self.params = {
                'max_depth': 4,
                'eta': 0.3,
                'silent': 1,
                'objective': objective,
            }
        else:
            self.params = params

        # set number of classes if performing multi-label classification
        if self.params['objective'][:5] == 'multi':
            if num_class == -1:
                raise (ValueError('specify number of classes when performing multiclass classification'))
            else:
                self.params['num_class'] = num_class

        self.objective = objective
        self.n_rounds = n_rounds

    def fit(self, X, y):
        """Fit classifier to training data.

        :param X: training data examples
        :param y: training data labels
        """

        # split training data into training and validation sets
        data_train, data_val, target_train, target_val = train_test_split(X, y, test_size=0.2)

        self.classes_ = sorted(set(y))

        # define train and validation sets in required format
        dtrain = xgb.DMatrix(data_train, target_train)
        dval = xgb.DMatrix(data_val, target_val)
        watchlist = [(dval, 'eval'), (dtrain, 'train')]

        # train model
        gbm = xgb.train(self.params,
                        dtrain,
                        num_boost_round=self.n_rounds,
                        evals=watchlist,
                        verbose_eval=True
                        )
        self._gbm = gbm

        return self

    def predict(self, X):
        """Predict labels of new data.

        :param X: data for which to predict classes
        """

        # return labels with highest probability
        if self.objective[:6] == 'binary':
            return np.where(self._gbm.predict(xgb.DMatrix(X)) > 0.5, 1, 0)
        elif self.objective[:5] == 'multi':
            return np.argmax(self._gbm.predict(xgb.DMatrix(X)), axis=1)
        else:
            raise (NotImplementedError('Other objectives not yet tested'))

    def predict_proba(self, X):
        """Predict probabilities of labels of new data

        :param X: data for which to predict probabilities
        """

        # Return probabilities of labels.
        if self.objective[:6] == 'binary':
            probs = self._gbm.predict(xgb.DMatrix(X))
            return np.vstack((1 - probs, probs)).T
        elif self.objective[:5] == 'multi':
            return self._gbm.predict(xgb.DMatrix(X))
        else:
            raise (NotImplementedError('Other objectives not yet tested'))

    def score(self, X, y, sample_weight=None):
        """Score predictions on test data.

        :param X: test data examples
        :param y: test data labels
        :param sample_weight: sample weights for scoring predictions
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def score_features(self, f_to_name: dict):
        """Score feature importances

        :param f_to_name: dictionary mapping feature enumerations such as 'f0', 'f1', ...
        to feature names
        :return: dictionary mapping feature names as defined in f_to_name parameter
        to their estimated importances
        """
        f_scores = self._gbm.get_fscore()
        sum_f_scores = sum(f_scores.values())
        return {f_to_name[key]: val / sum_f_scores for key, val in f_scores.items()}
