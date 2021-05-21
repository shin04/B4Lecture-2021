import itertools
import collections

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegression:
    def __init__(self, digree=1):
        self.digree = digree
        self.v_num = 0
        self.coef = None

    def fit(self, X, y, lam=1.0, regularization=True, sample_weight=None):
        '''
        Fit linear model.

        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values
        sample_weight : array, default=None
            Individual weights for each sample
        lam : float
            hyper parameter for regularization
        regularization : bool
            adding l2 reguration (redge) or not

        Returns
        -------
        self : returns an instance of self.
        '''

        self.v_num = X.shape[1]

        poly = PolynomialFeatures(self.digree)
        phi = poly.fit_transform(X)
        # phi = np.zeros((len(X), self.digree+1))
        # for i in range(self.digree+1):
        #     p = X**i
        #     phi[:, i] = p.ravel()

        c = 0
        if regularization:
            c = lam * np.eye(phi.shape[1])

        A = np.dot(phi.T, phi) + c
        B = np.dot(phi.T, y)
        self.coef = np.dot(np.linalg.inv(A), B)
        # self.coef = np.dot(np.linalg.pinv(A), B)

        return self

    def predict(self, X):
        '''
        predict data using self.coef and input data X.

        Parameters
        ----------
        X : array
            data for using prediction

        Returns
        -------
        pred : array
            prediction
        '''

        assert self.coef is not None, 'please fit first'

        pred = []

        poly = PolynomialFeatures(self.digree)
        X_poly = poly.fit_transform(X)

        for x in X_poly:
            y = self.coef @ x
            pred.append(y)

        return pred

    def get_equation(self):
        '''
        return regression equation as string

        Returns
        -------
        reg_eq : string
            regression equation
        '''

        assert self.coef is not None, 'please fit first'

        reg_eq = 'y='
        v_list = ['x'+str(i) for i in range(self.v_num)]
        comb_list = []
        for dig in range(self.digree + 1):
            for comb in itertools.combinations_with_replacement(v_list, dig):
                c = collections.Counter(comb)
                c_str = ''
                for k, v in c.items():
                    if v == 0:
                        continue
                    elif v == 1:
                        c_str += k
                    else:
                        c_str += k + '**' + str(v)
                comb_list.append(c_str)

        assert len(comb_list) == len(self.coef), 'failed'

        for i, c in enumerate(self.coef):
            if i != 0:
                reg_eq += '+'

            reg_eq += '{:.2f}{}'.format(c, comb_list[i])

        return reg_eq
