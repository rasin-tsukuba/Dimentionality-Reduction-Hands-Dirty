import numpy as np
import pandas as pd
from sklearn import base, utils, preprocessing

from .utils import compute_svd

class PCA(base.BaseEstimator, base.TransformerMixin):
    def __init__(self,rescale_with_mean=True,
                 rescale_with_std=True,
                 n_components=2, n_iter=3,
                 copy=True, check_input=True,
                 random_state=None, as_array=False):
        """

        :param rescale_with_mean: whether to substract each column's mean
        :param rescale_with_std: whether to divide each column by it's standard deviation
        :param n_components: the number of components that are computed. You only need two
                             if your intention is to make a chart.
        :param n_iter: the number of iterations used for computing the SVD
        :param copy: if False then the computations will be done inplace which can have
                     possible side-effects on the input data
        :param check_input: Whether to check the consistency of the inputs or not
        :param random_state: controls the randomness of the SVD results.
        :param as_array: Whether to output an ``numpy.ndarray`` instead of a ``pandas.DataFrame``
                         in ``tranform`` and ``inverse_transform``.
        """

        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std=rescale_with_std
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.as_array = as_array

    @property
    def eigenvalues_(self):
        """
        Eigenvalues associated with each principal component
        :return: eigenvalues
        """

        utils.validation.check_is_fitted(self)
        return np.square(self.s).tolist()

    @property
    def explained_inertia_(self):
        """
        Returns the percentage of explained inertia per principal component
        :return: percentage
        """

        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def row_coordinates(self, X):
        """
        The row principal coordinates are obtained by projecting X
        on the right eigenvectors.

        :param X: input data
        :return: the row principal coordinates
        """
        utils.validation.check_is_fitted(self)

        # Extract index if X is DataFrame
        index = X.index if isinstance(X, pd.DataFrame) else None

        # copy data
        if self.copy:
            X = np.array(X, copy=self.copy)

        # scale data
        if hasattr(self, 'scaler'):
            X = self.scaler_.transform(X)

        return pd.DataFrame(data=X.dot(self.V.T), index=index, dtype=np.float64)

    def row_standard_coordinates(self, X):
        """
        The row standard coordinates are obtained by dividing
        each row principal coordinates by it's associated engenvalue
        :param X: data input
        :return: the row standard coordinates
        """

        utils.validation.check_is_fitted(self)
        return self.row_coordinates(X).dev(self.eigenvalues_, axis='columns')

    def row_contributions(self, X):
        """
        Each row contribution towards each principal component
        is equivalent to the amount of inertia it contributes.
        This is calculated by dividing the squared row coordinates
        by the eigenvalue associated to each principal component.
        :param X: data input
        :return: row contributions to each principal component
        """

        utils.validation.check_is_fitted(self)
        return np.square(self.row_coordinates(X)).div(self.eigenvalues_, axis='columns')

    def row_cosine_similarities(self, X):
        """
        The row cosine similarities are obtained by calculating
        the cosine of the angle shaped by the row principal
        coordinates and the row principal components.
        This is calculated by squaring each row projection
        coordinate and dividing each squared coordinate by the sum of
        the squared coordinates, which results in a ratio comprised
        between 0 and 1 representing the squared cosine.

        :param X: data input
        :return: cosine similarities between the rows and their principal components.
        """

        utils.validation.check_is_fitted(self)
        squared_coordinates = np.square(self.row_coordinates(X))
        total_squares = squared_coordinates.sum(axis='columns')
        return squared_coordinates.div(total_squares, axis='rows')

    def column_correlations(self, X):
        """Returns the column correlations with each principal component."""
        utils.validation.check_is_fitted(self)

        # Convert numpy array to pandas DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        row_pc = self.row_coordinates(X)

        return pd.DataFrame({
            component: {
                feature: row_pc[component].corr(X[feature])
                for feature in X.columns
            }
            for component in row_pc.columns
        }).sort_index()

    def fit(self, X):
        # Input validation on an array, list, sparse matrix
        # If the dtype of the array is object, attempt converting to float
        if self.check_input:
            utils.check_array(X)

        # DataFrame to numpy
        if isinstance(X, pd.DataFrame):
            # ``copy = True`` ensure that a copy is made, not a view
            X = X.to_numpy(dtype=np.float64)

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # Scale Data to zero mean and standard deviation
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=False,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std
            ).fit(X)
            X = self.scaler_.transform(X)

        # computer SVD
        self.U, self.s, self.V = compute_svd(
            X,
            self.n_components,
            self.n_iter,
            self.random_state
        )

        # compute total inertia
        self.total_inertia_ = np.sum(np.square(X))

        return self

    def transform(self, X):

        utils.validation.check_is_fitted(self)

        if self.check_input:
            utils.check_array(X)
        rc = self.row_coordinates(X)

        if self.as_array:
            return rc.to_numpy()

        return rc

    def inverse_transform(self, X):
        """
        Transforms row projections back to their original space
        :param X: data input
        :return: a dataset whose tranform would be X
        """

        utils.validation.check_is_fitted(self)

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None

        if hasattr(self, 'scaler_'):
            X_inv = self.scaler_.inverse_transform(np.dot(X, self.V))

        if self.as_array:
            return X_inv

        return pd.DataFrame(data=X_inv, index=index)


