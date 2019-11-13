import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin

def _transition(state):
    n = np.random.randint(0, len(state))
    return tuple(b if i != n else not b for i, b in enumerate(state))


class RandomWalkFeatureSelection(BaseEstimator, MetaEstimatorMixin,
                                 TransformerMixin):
    """Simulated annealing-based feature selection.

    Given an external regressor (or anything with a ``score`` method)
    RWFS attempts to find the set of features that results in the highest
    score on given dataset and cross validation technique.

    It begins by choosing a set of features at random, and then takes a
    sequence of steps where a feature is toggled on or off at random.

    If the regressor performs better using the new set of features, the
    new set is taken as the new starting point. If it performs worse,
    it will still be accepted with a small probability depending on just how
    much worse in order to avoid getting stuck in a local minimum.

    The probability of accepting a worse state decreases by each time step
    to try and make sure it eventually settles on a good solution.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``score`` method.

    cv : cross-validation generator.
        Determines the cross-validation splitting strategy.

    n_steps : int
        Number of steps.

    initial_fraction : float
        Fraction of features to include in initial state.

    temperature : float
        Initial temperature.

        A higher temperature means larger probabilty of accepting a
        set of features with worse performance.

        The best value is dependent on the scale of the scores.

    cooldown_factor : float
        Determines how fast the temperature decreases.

        Each time step the temperature is multiplied by this to get the new
        temperature.

    cache_scores : bool
        If true, cache evaluations for feature sets.

        This can speed up the process considerably, but is less robust to
        scores with high variance.  Do not use unless you are very sure
        of your CV accuracy.

    agg : function
        Function to aggregate CV scores at each step.

        Can be useful, for example, to take the min instead of the mean in
        order to to be more conservative when comparing feature sets.

    gamma : float
        Exponential decay factor for feature importances.

    Attributes
    ----------
    best_score_ : float
        CV score when evaluating best set of features found.

    best_features_ : np.array of bool
        Features found that resulted in the  best score.

        Represented as an array of bool:s, True if the feature is in the set,
        False if not.

    feature_importances_ : np.array of float
        Importance of each feature.

        Given by the fraction of accepted feature sets a feature has been
        part of, weighted exponentially giving more weight to more recent
        steps.

    diagnostics_ : dict
        Statistics collected during the process:

        - features : feature set being evaluated
        - feature_count : number of features in the set
        - scores : CV scores
        - score : aggregated CV score
        - mean-score : mean CV score
        - runs : number of evaluations of estimator
        - std : standard deviation of CV score
        - se : standard error of CV score
    """
    def __init__(self, estimator, cv, n_steps, initial_fraction=0.5,
                 temperature=1e-1, cooldown_factor=0.99, cache_scores=False,
                 agg=np.mean, gamma=0.99):
        self.estimator = estimator
        self.cv = cv
        self.n_steps = n_steps
        self.initial_fraction = initial_fraction
        self.temperature = temperature
        self.cooldown_factor = cooldown_factor
        self.cache_scores = cache_scores
        self.gamma = gamma  # decay of feature importances

        self.agg = agg

        self.best_features_ = None
        """Best set of features found."""

        self.feature_importances_ = None
        """Feature importances"""

        self.best_score_ = None
        """CV score when evaluating best set of features found. """

        self.diagnostics_ = None
        """Statistics gathered during search."""

    def fit(self, X, y, groups=None, verbose=0):
        # Handle peculiarity with StratifiedShuffleSplit and KFold
        if groups is None:
            groups = y

        # Check that we have a 2D matrix
        assert len(X.shape) == 2

        n_features = X.shape[1]
        state = tuple(np.random.uniform(0, 1) < self.initial_fraction
                      for _ in range(n_features))
        score = -np.inf
        diagnostics = None
        temperature = self.temperature

        best_state = None
        best_score = -np.inf

        feature_counts = np.zeros(X.shape[1])
        n_accepted = 0

        self.diagnostics_ = []
        visited = {}

        for i in range(self.n_steps):
            new_state = _transition(state)

            # Must have at least one feature
            while not np.any(new_state):
                new_state = _transition(new_state)

            # TODO: Add some kind of penalty for number of features (AIC?)
            # TODO: Reverse condition for clarity
            if not self.cache_scores or new_state not in visited:
                X_trial = X[:, new_state]
                trial_scores = []
                for train_index, test_index in self.cv.split(X_trial, groups,
                                                             groups):
                    self.estimator.fit(X_trial[train_index], y[train_index])
                    s = self.estimator.score(X_trial[test_index], y[test_index])

                    if verbose >= 2:
                        n = np.sum(new_state)
                        print(
                            f"Estimator score during CV for {n} feature(s): {s}")

                    trial_scores.append(s)

                trial_score = self.agg(trial_scores)

                std = np.std(trial_scores)
                n_runs = len(trial_scores)
                trial_diagnostics = {
                    'features': new_state,
                    'feature_count': np.sum(new_state),
                    'scores': trial_scores,
                    'score': trial_score,
                    'mean_score': np.mean(trial_scores),
                    'runs': n_runs,
                    'std': std,
                    'se': std / np.sqrt(n_runs)
                }

                if self.cache_scores:
                    visited[new_state] = trial_diagnostics

            else:
                trial_diagnostics = visited[new_state]
                trial_score = diagnostics['score']

            delta = trial_score - score
            p_accept = np.clip(np.exp(delta / temperature), 0,
                               1)  # Clipping isn't technically necessary
            if np.random.uniform(0, 1) < p_accept:
                if (verbose >= 1):
                    print(
                        f"Step {i + 1}: Accept new state with score {trial_score:.5f} over old score {score:.5f} (p = {p_accept:.3f})")

                state = new_state
                score = trial_score
                diagnostics = trial_diagnostics

                if score > best_score:
                    best_state = state
                    best_score = score

                # Update feature importances
                feature_counts *= self.gamma
                n_accepted *= self.gamma
                feature_counts += np.array(state)
                n_accepted += 1

                # Update diagnostics with accepted
                diagnostics['accepted'] = True
            else:
                if (verbose >= 1):
                    print(
                        f"Step {i + 1}: Reject new state with score {trial_score:.5f} over old score {score:.5f} (p = {p_accept:.3f})")

                # Update diagnostics with accepted
                diagnostics['accepted'] = False

            self.diagnostics_.append(diagnostics)
            temperature *= self.cooldown_factor

        self.best_features_, self.best_score_ = best_state, best_score
        self.feature_importances_ = feature_counts / n_accepted

        return self

    def transform(self, X, y=None):
        return X[self.best_features_]