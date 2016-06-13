from __future__ import division

import numpy as np

from ..base import ClassifierMixin
from ..externals.joblib import Parallel, delayed
from ..utils import check_random_state

from .base import BaseEnsemble


MAX_INT = np.iinfo(np.int32).max

def _parallel_build_estimators(ensemble, X, y, sample_weight, random_state):
    seed = random_state.randint(MAX_INT)
    estimator = ensemble._make_estimator(append=False)

    try:  # Not all estimator accept a random_state
        estimator.set_params(random_state=seed)
    except ValueError:
        pass

    estimator.fit(X, y, sample_weight)
    return estimator

class WeightedEnsembleClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(self,
                 base_estimator=None,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(WeightedEnsembleClassifier, self).__init__(
            base_estimator=base_estimator, n_estimators=None)

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose


    def fit(self, X, y, sample_weight):
        assert(sample_weight.ndim == 2)
        assert(X.shape[0] == sample_weight.shape[0])

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        n_samples, n_estimators = sample_weight.shape
        self.n_estimators = n_estimators

        self._validate_estimator()

        random_state = check_random_state(self.random_state)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                self,
                X,
                y,
                sample_weight[:, i_model],
                random_state)
            for i_model in range(sample_weight.shape[1])
        )
        self.estimators_ = results

    def _decision_function(self, X, sample_weight=None):
        if sample_weight is not None:
            assert(sample_weight.ndim == 2)
            assert(X.shape[0] == sample_weight.shape[0])
            assert(sample_weight[0] == self.n_estimators)
        n_samples, n_estimators = X.shape[0], self.n_estimators
        predictions = np.zeros((X.shape[0], n_estimators, self.n_classes_), dtype=np.float64)
        for i_model in range(n_estimators):
            confidence = self.estimators_[i_model].decision_function(X)
            if confidence.ndim > 1:
                pred_bin = np.argmax(confidence)
                predictions[(range(n_samples), i_model, pred_bin)] = confidence
            else:
                pred_bin = (confidence > 0).astype(np.int_)
                predictions[(range(n_samples), i_model, pred_bin)] = np.abs(confidence)

        if sample_weight is not None:
            predictions *= predictions * sample_weight
        return predictions

    def decision_function(self, X, sample_weight=None):
        return self._decision_function(X, sample_weight).sum(axis=1).max(axis=1)

    def predict(self, X, sample_weight=None):
        predictions = self._decision_function(X, sample_weight=None)
        if sample_weight is not None:
            predictions *= predictions * sample_weight
        return self.classes_.take((predictions.sum(axis=1).argmax(axis=1)))
