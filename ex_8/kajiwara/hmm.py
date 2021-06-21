import numpy as np


class HMM():
    """
    Hidden Markov Model

    Attributes
    ----------
    pi : ndarray (n_states, )
        An aray of probability of initial state.
    A : ndarray (n_state, n_state)
        Transition matrix。
    B : ndarray (n_state, c_symbols)
        Output matrix of elements in each state。
    """

    def __init__(self, pi: np.array, A: np.array, B: np.array) -> None:
        """
        Parameters
        ----------
        pi : ndarray (n_states, )
            An aray of probability of initial state.
        A : ndarray (n_state, n_state)
            Transition matrix。
        B : ndarray (n_state, c_symbols)
            Output matrix of elements in each state。
        """

        self.pi = pi
        self.A = A
        self.B = B
        self.n_status = A.shape[0]

    def forward(self, outputs: np.array) -> np.array:
        """
        Parameters
        ----------
        outputs : ndarray (n_samples, l_series)
            A list of output series.

        Returns
        -------
        prob : ndarray (n_samples,)
            Observation probability.
        """

        n_samples = outputs.shape[0]
        l_series = outputs.shape[1]

        alphas = np.zeros((n_samples, l_series, self.n_status))
        for i in range(n_samples):
            o = outputs[i]

            # initialize
            alpha = np.zeros((l_series, self.n_status))
            alpha[0, :] = self.pi * self.B[:, o[0]]

            # reccurent process
            for t in range(1, l_series):
                for j in range(self.n_status):
                    alpha[t, j] = alpha[t-1] @ (self.A[:, j]*self.B[j, o[t]])
            alphas[i] = alpha

        prob = np.sum(alphas[:, -1, :], axis=1)

        return prob

    def viterbi(self, outputs: np.array) -> np.array:
        """
        Parameters
        ----------
        outputs : ndarray (n_samples, l_series)
            A list of output series.

        Returns
        -------
        prob : ndarray (n_samples,)
            Observation probability.
        """

        n_samples = outputs.shape[0]
        l_series = outputs.shape[1]

        vs = np.zeros((n_samples, l_series, self.n_status))
        for i in range(n_samples):
            o = outputs[i]
            v = np.zeros((l_series, self.n_status))
            # v[0, :] = np.log(pi*B[:, o[0]])
            v[0, :] = self.pi*self.B[:, o[0]]
            w = np.zeros((l_series-1, self.n_status))

            for t in range(1, l_series):
                for j in range(self.n_status):
                    # prob = v[t-1] + np.log((A[:,j]*B[j,o[t]]))
                    t_prob = v[t-1] @ (self.A[:, j]*self.B[j, o[t]])
                    w[t-1, j] = np.argmax(t_prob)
                    v[t, j] = np.max(t_prob)

            vs[i] = v

        prob = np.sum(vs[:, -1, :], axis=1)

        return prob
