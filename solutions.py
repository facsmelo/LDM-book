import numpy as np
import networkx as nx
import numpy.random as rnd
import numpy.linalg as la
import matplotlib.pyplot as plt

import scipy.optimize as opt

import models

from functools import reduce
from models import get_index, sample, get_item

MAX_ERROR = 1e-2
MAX_ITER = 5e3
MAX_REPS = 100
MAX_POINTS = 5e3

#  = = = = =  A U X I L I A R Y   F U N C T I O N S  = = = = = #

def softmax(x, temp=0):
    """
    softmax(x, t) creates a softmax distribution from the values in x.

    :param x: np.ndarray (1D or 2D values; the distribution is computed row-wise)
    :param temp: float > 0 (temperature)
    :return: np.ndarray (distribution; same size as x)
    """

    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except:
            raise ValueError('softmax -> invalid operand')

    sh = x.shape

    if len(sh) > 1:
        r = sh[0]
        c = sh[1]
        m = x.max(axis=1).reshape((r, 1))
    else:
        r = 1
        c = sh[0]
        m = np.max(x).reshape((r, 1))

    if temp == 0:
        p = (x == m).astype(int)
    else:
        p = np.exp((x - m) / temp)

    s = p.sum(axis=1).reshape((r, 1))
    p = p / s

    if len(sh) == 1:
        p = np.reshape(p, (c,))

    return p

# -- End: softmax

def softmin(x, temp=0):
    """
    softmin(x, t) creates a softmin distribution from the values in x.

    :param x: np.ndarray (1D or 2D values; the distribution is computed row-wise)
    :param temp: float > 0 (temperature)
    :return: np.ndarray (distribution; same size as x)
    """

    if not isinstance(x, np.ndarray):
        try:
            x = -np.array(x)
        except:
            raise ValueError('softmax -> invalid operand')
    else:
        x = -x

    return softmax(x, temp)

# -- End: softmin

def sample_beliefs(m, n):
    """
    sample(m, n) receives a POMDP m and builds a set of n sampled beliefs using a random policy. The sampling
    process is terminated is new sampled points can't be found at a distance smaller than 1e-2.

    :param m: POMDP
    :param n: int (number of beliefs to sample)
    :return: np.ndarray (belief set, one per row)
    """

    minimum_belief_distance = 1e-2

    aux = np.eye(m.n_states())

    # Depart from POMDP initial state
    m.reset()

    b_set = [m.get_current_belief()]

    i = 1
    k = 0

    while i < n and k < MAX_REPS:
        m.__next__()
        b_new = m.get_current_belief()
        d = (np.concatenate(b_set, axis=0) - b_new) ** 2
        d_min = d.sum(axis=1).min()

        if d_min >= minimum_belief_distance / n:
            i += 1
            k = 0
            b_set += [b_new]
        else:
            k += 1
            idx = rnd.choice(i)
            m.reset(new_belief=b_set[idx])

    for s in range(m.n_states()):
        d = (np.array(b_set) - aux[s, None, :]) ** 2
        d_min = d.sum(axis=1).min()

        if not np.isclose(d_min, 0, atol=1e-10):
            b_set += [aux[s, None, :]]

    print('Sampled a total of', len(b_set), 'belief points.')

    return np.concatenate(b_set, axis=0)

# -- End: sample

def pbvi_backup(m, g_curr, l_curr, s_curr, b_queue, update_all=False):
    """
    backup(m, g_curr, b_queue, all=False) computes a backup of the alpha-vectors in g_curr using the beliefs in b_queue.
    The model used is the POMDP m. If all=True, then all beliefs are updated, otherwise, beliefs whose value improves in
    one iteration are not updated.

    :param m: POMDP
    :param g_curr: np.ndarray (set of a-vectors, one per column)
    :param l_curr: list (labels) 
    :param s_curr: list of lists (successors)
    :param b_queue: np.ndarray (set of beliefs, one per row)
    :param update_all: boolean
    :return: (np.ndarray, list, list) (set of a-vectors, one per column, list of labels and list of successors)
    """

    b_queue = np.copy(b_queue)
    g_new = []
    l_new = []
    s_new = []

    quit = False

    while not quit:

        # Auxiliary successor queue
        s_aux = []

        n_points = b_queue.shape[0]

        # Select a belief from the queue

        idx = rnd.choice(n_points)
        b = b_queue[idx, :]
        b_queue = np.delete(b_queue, idx, axis=0)

        # Compute new alpha-vector

        a_new = np.zeros((m.n_states(), m.n_actions()))

        for a in m.list_actions():
            a_idx = m.get_action_index(a)
            aux = b.dot(m.p(a=a))
            aux = aux * m.o(a=a).T
            idx = aux.dot(g_curr).argmin(axis=1)

            a_next = g_curr[:, idx]

            a_new[:, a_idx] = m.c(a) + m.gamma() * m.p(a=a).dot(m.o(a=a) * a_next).sum(axis=1)
            s_aux += [idx.tolist()]

        # Test new a-vectors

        j_new = b.dot(a_new).min()
        j_old = b.dot(g_curr).min()

        if j_new <= j_old:
            idx = int(b.dot(a_new).argmin())
            aux = a_new[:, idx, None]

            if not any(np.isclose(aux, x, atol=1e-10,rtol=1e-10).all() for x in g_new):
                g_new += [aux]
                l_new += [m.get_action(idx)]
                s_new += [s_aux[idx]]
        else:
            idx = b.dot(g_curr).argmin()
            aux = g_curr[:, idx, None]

            if not any(np.isclose(aux, x, atol=1e-10,rtol=1e-10).all() for x in g_new):
                g_new += [aux]
                l_new += [l_curr[idx]]
                s_new += [s_curr[idx]]

        # Clean b-queue

        if not update_all:
            if b_queue.shape[0] != 0:
                j_old = b_queue.dot(g_curr).min(axis=1)
                j_new = b_queue.dot(np.concatenate(g_new, axis=1)).min(axis=1)

                idx = np.where(j_old >= j_new)[0]
                b_queue = np.delete(b_queue, idx, axis=0)

        quit = (b_queue.shape[0] == 0)

    return np.concatenate(g_new, axis=1), l_new, s_new

# -- End: backup

def extract_controller(m, q, l, s, qnew, lnew, snew):
    """
    extract_controller(m, q, l, s, qnew, lnew, snew) takes the a-vectors in q and the controller C defined by labels l
     and successors s and creates an updated controller by adding the nodes defined by labels l and successors s
     and removing redundant and unlinked nodes from C.

    :param m: POMDP
    :param q: np.ndarray (set of a-vectors)
    :param l: list (action labels)
    :param s: list (successors)
    :param qnew: np.ndarray (set of a-vectors)
    :param lnew: list (action labels)
    :param snew: list (successors)
    :return: Controller (updated policy)
    """

    # Round everything to avoid numerical errors
    q = np.around(q, 10)
    qnew = np.around(qnew, 10)

    # List of previous nodes
    idx_check = list(range(len(l)))

    # Initialize new nodes to previous ones
    lfinal = l.copy()
    sfinal = s.copy()

    # Track vertices to remove at the end
    to_remove = []
    to_link = []

    # Add one new node at a time
    for v_idx in range(len(lnew)):

        # Get new a-vector
        v = qnew[:, v_idx, None]

        # Check if a-vector is dominated
        # noinspection PyUnresolvedReferences
        if (q[:, idx_check] - v >= 0).all(axis=0).any():

            # Get index of dominating a-vector

            # noinspection PyUnresolvedReferences
            aux = (q[:, idx_check] - v >= 0).all(axis=0)
            idx_remove = np.where(aux)[0]
            idx_replace = idx_check[idx_remove[0]]

            if idx_remove.size > 1:
                to_remove += [idx_check[i] for i in idx_remove[1:]]
                to_link += [idx_replace] * (idx_remove.size - 1)

            # Remove from check
            for i in sorted(idx_remove, reverse=True):
                del idx_check[i]

            # Replace first dominating node
            lfinal[idx_replace] = lnew[v_idx]
            sfinal[idx_replace] = snew[v_idx]

        # Non-dominated nodes are just added
        else:
            lfinal.append(lnew[v_idx])
            sfinal.append(snew[v_idx])

 #   print('\n--> Housecleaning --')
 #   print('\tGraph (before removal):')
 #   print('Nodes:', len(lfinal))
 #   print('\tLabels:', lfinal, '- Successors:', sfinal)

    # Sort indices to be removed
    idx_sorted = sorted(range(len(to_remove)), key=lambda i: to_remove[i], reverse=True)

    for idx in idx_sorted:
        source = to_remove[idx]
        target = to_link[idx]

 #       print('\tRemoving', source, 'and replacing links by', target)

        # Replace nodes linking to it
        sfinal = [[i if i != source else target for i in s] for s in sfinal]

        # Remove from check-list
        idx_check = [i - 1 if i > source else i for i in idx_check]

        # Remove node
        del lfinal[source]
        del sfinal[source]

        # Push all successors one number down
        sfinal = [[j - 1 if j > source else j for j in s] for s in sfinal]

 #   print('\tGraph (after removal):')
 #   print('Nodes:', len(lfinal))
 #   print('\tLabels:', lfinal, '- Successors:', sfinal)
 #   print('\tJust removed', sorted(to_remove))
 #   print('--> End housecleaning --\n')

    for i in idx_check[::-1]:
        # Check if node i is linked by any other node
        is_linked = reduce(lambda x, y: x or y, map(lambda x: i in x, sfinal))

        # If is not linked, remove it and renumber subsequent nodes
        if not is_linked:
            del lfinal[i]
            del sfinal[i]
            sfinal = [[j - 1 if j > i else j for j in s] for s in sfinal]


    try:
        return Controller(m, labels=lfinal, successors=sfinal)
    except:
        raise ValueError('This blew up in creating the controler')

# -- End: extract_controller

def fib(m):
    """
    fib(M) returns the FIB Q-values for the POMDP M.

    :param m: POMDP
    :param max_iter: int
    :return: np.ndarray
    """

    def fib_update(qold):
        """
        fib_update(q) produces an updated version of q for the POMDP M.

        :param qold: np.ndarray (Q-function)
        :return: np.ndarray (updated Q-function)
        """

        qnew = np.zeros((m.n_states(),m.n_actions()))

        # DP update

        for a in m.list_actions():
            a_idx = get_index(a, m.list_actions())

            for z in m.list_observations():
                qnew[:, a_idx] += m.gamma() * (m.p(a=a) * m.o(a=a, z=z)).dot(qold).min(axis=1)

        qnew += m.c()

        return qnew

    # -- End: fib_update

    # Argument check

    if not isinstance(m, models.POMDP):
        raise ValueError('fib -> invalid argument type (POMDP expected)')

    # Stopping condition

    eps = 0.5 * MAX_ERROR / m.gamma() * (1 - m.gamma())

    # Initialization

    q = np.zeros((m.n_states(), m.n_actions()))

    it = 0
    qt = False

    # VI cycle

    while not qt:

        # Update
        qupd = fib_update(q)

        # Stopping condition
        qt = (la.norm(qupd - q, ord=2) < eps)

        q = qupd
        it += 1

    print('FIB ended after', it, 'iterations')
    return q, it

# -- End: fib

#  = = = = =  F E A T U R E   F U N C T I O N S  = = = = = #

def mdp_identity(m, x):
    """
    mdp_identity(m, x) is a feature map that receives as input a state x of the (PO)MDP m and returns its representation
     as a row vector containing a 1 in the corresponding entry.

    :param m: (PO)MDP (model)
    :param x: univ (input state)
    :return:  ndarray (resulting unitary vector)
    """

    if not isinstance(m, models.POMDP):
        raise TypeError('mdp_identity -> invalid model')

    i = np.eye(m.n_states())
    try:
        x = get_index(x, m.list_states())
    except:
        raise ValueError('mdp_identity -> invalid input state')

    return i[x, :]

# -- End: mdp_identity

def fsc_identity(c, x):
    """
    fsc_identity(c, x) is a feature map that receives as input a controller and returns its representation as a row
    vector containing a 1 in the corresponding entry.

    :param c: int (number of controller states)
    :param x: univ (input state)
    :return:  ndarray (resulting unitary vector)
    """
    if not isinstance(c, Controller):
        raise TypeError('fsc_identity -> invalid controller')

    i = np.eye(c.n_states())

    if x not in c.list_states():
        raise ValueError('fsc_identity -> invalid controller state')

    return i[x, :]

# -- End: fsc_identity

def minimizer(a_vectors, b):
    """
    minimizer(a_vectors, b) is a feature map that receives as input a set of a_vectors and retrieves the controller
     state for belief b. The number of elements of b must be the same as the number of rows of a_vectors. b is supposed
     to be a row-vector.

    :param a_vectors: np.ndarray (a-vectors)
    :param b: np.ndarray (belief)
    :return: np.ndarray (vector with index of maximizing a-vector)
    """

    if not isinstance(a_vectors, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError('minimizer -> invalid argument type')

    if len(b.shape) != 2 or len(a_vectors.shape) != 2:
        raise ValueError('minimizer -> invalid argument shape')

    if b.shape[0] != 1 or b.shape[1] != a_vectors.shape[0]:
        raise ValueError('minimizer -> argument dimensions must agree')

    i = np.eye(a_vectors.shape[1])
    x = b.dot(a_vectors).argmax()

    return i[x, :]

# -- End: n_states

#  = = = = =  P O L I C Y   C L A S S E S  = = = = = #

class Controller:
    def __init__(self, m, labels=None, successors=None, feature_map=minimizer):
        """
        Controller(M, labels=L, successors=S, feature_map=F)

        Builds a (finite-state) controller with node labels L (elements of L are actions); node successors S and
         feature map F.

        :param m: POMDP (or subclass thereof)
        :param labels: list with actions; each element corresponds to a state of the controller;
        :param successors: list with successors; each element corresponds to a list with as many elements as
         observations in m, containing the index of the successor (controller) state
        :param feature_map: function that maps inputs and outputs a controller state in the form of a column array
        """

        # 1. Check arguments

        # m

        if not isinstance(m, models.POMDP):
            raise TypeError('Controller -> invalid model type')

        self._actions = m.list_actions()
        self._na = len(self._actions)

        self._observations = m.list_observations()
        self._nz = len(self._observations)

        # Labels and successors

        if labels is not None and successors is not None:
            if len(labels) != len(successors):
                raise ValueError('Controller -> incompatible dimensions for label and successor lists')

        if labels is None and successors is None:
            labels = [rnd.choice(m.n_actions())]
            successors = [[0] * m.n_observations()]

        self._states = list(range(len(labels)))
        self._list_states = self._states
        self._ns = len(labels)

        self._dist = np.zeros((self._ns, self._na))

        for n in self._states:
            try:
                labels[n] = get_item(labels[n], self._actions)
            except:
                raise ValueError('Controller -> invalid label array')

            self._dist[n, get_index(labels[n], self._actions)] = 1

            if len(successors[n]) != self._nz:
                print(len(successors[n]), 'edges for', self._nz, 'elements')
                raise ValueError('Controller -> invalid successor array')

            for z_idx in range(self._nz):
                if successors[n][z_idx] not in self._states:
                    print('Successor', successors[n][z_idx], 'is not in', self._states)
                    raise ValueError('Controller -> invalid successor array')

        self._labels = labels
        self._successors = successors

        # feature_map

        if not callable(feature_map):
            raise ValueError('Controller -> invalid feature map')

        self._feature_map = feature_map

    # -- End: __init__

    def labels(self):
        """
        Getter.

        :return: list (labels)
        """

        return self._labels

    # -- End: labels

    def successors(self):
        """
        Getter.

        :return: np.array (successors)
        """

        return self._successors

    # -- End: labels

    def list_states(self):
        """
        C.list_states() returns a list with all states of controller C (list).

        :return: list (of states)
        """

        return self._states

    # -- End: list_states

    def list_actions(self):
        """
        C.list_actions() returns a list with all actions available to controller C (list).

        :return: list (of actions)
        """

        return self._actions

    # -- End: list_actions

    def list_observations(self):
        """
        C.list_observations() returns a list with all observations driving controller C (list).

        :return: list (of observations)
        """

        return self._observations

    # -- End: list_observations

    def n_states(self):
        """
        C.n_states() returns the number of internal controller states.

        :return: int (n. of states)
        """

        return len(self._states)

    # -- End: n_states

    def n_actions(self):
        """
        C.n_actions() returns the number of actions available.

        :return: int (n. of actions)
        """

        return len(self._actions)

    # -- End: n_actions

    def n_observations(self):
        """
        C.n_observations() returns the number of observations driving the controller.

        :return: int (n. of observations)
        """

        return len(self._observations)

    # -- End: n_observations

    def get_action(self, s):
        """
        C.get_action(s) returns the action for controller state s (univ). If multiple actions are allowed, it samples
         one randomly according to the prescribed probabilities.

        :param s: int (controller state)
        :return: univ (action)
        """

        try:
            s = get_index(s, self._list_states)
        except:
            print('State provided:', s)
            print('Possible states:', self._list_states)
            raise ValueError('Policy.get_action -> invalid state')

        return sample(self._actions, self._dist[s, :])

    # -- End: get_action

    def get_successor(self, s, z):
        """
        C.get_successor(s, z) returns the successor of s given observation z (int).

        :param s: int (controller state)
        :param z: univ (observation)
        :return: int (controller state)
        """

        if s not in self._states:
            raise ValueError('Controller.get_successor -> invalid state')

        try:
            z_idx = get_index(z, self._observations)
        except:
            raise ValueError('Controller.get_successor -> invalid observation')

        return self._successors[s][z_idx]

    # -- End: get_successor

    def probability(self, cstate=None, action=None, mstate=None):
        """
        C.probability() returns the probability matrix for controller C (ndarray).

        C.probability(cstate=s) returns the probability distribution for the controller state s (ndarray).

        C.probability(cstate=s, action=a) returns the probability of action a in (controller) state s (float).

        C.probability(action=a) returns the probability of action a for the different controller states (ndarray).

        C.probability(mstate=x) returns the probability of each action for the input x (ndarray). The input is weighted
         using the policy's feature map.

        C.probability(mstate=x, action=a) returns the probability of action a for the input x (float).

        :param cstate: int (controller state)
        :param action: univ (action)
        :param mstate: univ (input state)
        :return: ndarray or float (probability matrix/probability vector/probability)
        """

        if cstate is None:
            if mstate is None:
                p = self._dist
            else:
                try:
                    p = self._feature_map(input).dot(self._dist)
                except:
                    raise ValueError('C.probability -> invalid input')
        else:
            if mstate is None:
                p = self._dist[cstate, :]
            else:
                raise ValueError('C.probability -> incompatible arguments (state and input)')

        if action is None:
            return p
        elif len(p.shape) == 2:
            return p[:, action]
        else:
            return p[action]

    # -- End: probability

    def expectation(self, model=None, func=None):
        """
        C.expectation(func=F) computes the expectation of function F at the different controller nodes. F must return,
          for each input, a |X| x |A| vector, where X is the input space. The result is, therefore, a |S|-list where
          element s is a vector EF with x component given by

          EF_x = F(x, a(s)),

          with x an input state and s a controller state.

        C.expectation(model=M) averages the model components of M at the different controller nodes. It expects a POMDP
         model M and outputs a tuple (K, P) where

        - K is a |S|-list where the element s, Ks, is a vector with x component

          K[s][x] = M.c(x, a(s)),

          with x an input state;

        - P is a |S|-list where the element s is a |X| x |X||S|-matrix with xxs component

          P[s][x, y + s' * |X|] = P(y | x, a(s)) * O(z | y, a(s)) I(s' == suc(s, z))

          with x, y input states and s, s' controller states.

        :param model: POMDP
        :param func:  ndarray (|X| x |A| function array)
        :return: list or tuple (list, list)
        """

        if model is not None and func is not None:
            raise ValueError('Controller.expectation -> incompatible arguments (model and function)')

        if func is not None:
            if not isinstance(func, np.ndarray) or len(func.shape) != 2 or func.shape[1] != self._na:
                raise ValueError('Controller.expectation -> invalid function')

            ef = []

            for s in self._states:
                ef += [func.dot(self._dist[s, :]).reshape((func.shape[0], 1))]

            return ef

        c = []
        p = []
        nx = model.n_states()

        for s in self._states:
            c += [model.c().dot(self._dist[s,:]).reshape((model.n_states(), 1))]

            ptot = 0

            for z in model.list_observations():
                z_idx = model.get_observation_index(z)

                perm = np.zeros((nx, self._ns * nx))
                col = self._successors[s][z_idx]
                perm[:, col * nx: (col + 1) * nx] = np.eye(nx)

                paux = 0

                for a in model.list_actions():
                    a_idx = model.get_action_index(a)
                    paux += self._dist[s, a_idx] * np.dot(model.p(a=a), np.diag(model.o(a=a, z=z)))

                ptot += paux.dot(perm)

            # noinspection PyUnboundLocalVariable
            p += [ptot]

        return c, p

    # -- End: expectation

    def evaluate(self, m):
        """
        C.evaluate(M) returns the set of alpha-vectors representing the cost-to-go of controller C in the POMDP M. It
        returns is a |X| x |S| array, where column s is the alpha-vector corresponding to the node s in the controller.

        :param m: POMDP
        :return: np.ndarray (alpha-vectors)
        """

        i = np.eye(self._ns * m.n_states())
        cpi, ppi = self.expectation(model=m)

        if isinstance(cpi, list):
            cpi = np.concatenate(cpi, axis=0)

        if isinstance(ppi, list):
            ppi = np.concatenate(ppi, axis=0)

        a_vectors = la.inv(i - m.gamma() * ppi).dot(cpi)

        siz = (a_vectors.size//m.n_states(), m.n_states())
        return a_vectors.reshape(siz).T

    # -- End: evaluate

    def __eq__(self, c):
        """
        This method overrides the inherited method __eq__ for the object class. It returns True if C and C2 are the
        same controller (exactly). Labels must be the same and successors. The method does not look for controller
        homomorphisms (i.e., node relabeling).

        :param: Controller
        :return: bool
        """

        # noinspection PyUnresolvedReferences
        return self._labels == c.labels() and self._successors == c.successors()

    # -- End: __eq__

    def __str__(self):
        """
        C.__str__() returns the external representation of controller C (string).

        :return: string (representation of controller C).
        """
        s = ''

        for st in self._states:
            s += "'" + str(self._labels[st]) + "' -> {"
            for z_idx in range(self._nz):
                s += str(self._successors[st][z_idx]) + ', '

            s = s[:-2] + '}\n'

        return s

        # -- End: __str__

class Policy(Controller):

    # noinspection PyMissingConstructor
    def __init__(self, m, weights=np.array([]), feature_map=mdp_identity, immutable=False):
        """
        Policy(M, weights=W, feature_map=F)

        Builds an MDP policy with weights W and feature map F.

        :param m: POMDP (or subclass thereof)
        :param weights: list with actions or distribution over actions;
        :param feature_map: function that maps inputs and outputs a controller state in the form of a column array
        :param immutable: bool (determines whether weights are deep copied or not)
        """

        if not isinstance(m, models.POMDP):
            raise TypeError('Policy -> invalid model type')

        self._actions = m.list_actions()
        self._na = len(self._actions)

        self._list_states = m.list_states()
        self._ns = len(self._list_states)
        self._states = list(range(self._ns))

        self._observations = self._states
        self._nz = self._ns

        if not isinstance(weights, np.ndarray):
            raise TypeError('Policy -> invalid argument type')

        if np.array_equal(weights, np.array([])):
            self._dist = rnd.rand(len(self._states), len(self._actions))

        elif weights.shape != (len(self._states), len(self._actions)):
            raise ValueError('Policy -> invalid argument dimensions')

        else:
            if immutable:
                self._dist = np.copy(weights)
            else:
                self._dist = weights

            self._labels = self._dist

        self._successors = [list(range(self._ns))] * self._ns

        # feature_map

        if not callable(feature_map):
            raise ValueError('Policy -> invalid feature map')

        self._feature_map = feature_map

    # -- End: Policy

    def expectation(self, model=None, func=None):
        """
        P.expectation(func=F) computes the expectation of function F given policy P. F must return, for each input, a
         1 x |A| vector, where A is the action space. The result is a |X| x 1
         EF with x component given by

          EF_x = F(x, a(x)),

          with x an input state.

        C.expectation(model=M) averages the model components of M given policy P. It expects a (PO)MDP model M and
         outputs a tuple (K, P) where

        - K is a |X| x 1 vector with x component

          K[x] = M.c(x, a(x)),

          with x an input state;

        - P is a |X| x |X| matrix with xy component

          P[x, y] = P(y | x, a(x))

          with x, y input states.

        :param model: POMDP
        :param func:  ndarray (|X| x |A| function array)
        :return: list or tuple (list, list)
        """

        if model is not None and func is not None:
            raise ValueError('Policy.expectation -> incompatible arguments (model and function)')

        if func is not None:
            if not isinstance(func, np.ndarray) or \
                            len(func.shape) != 2 or \
                            func.shape[0] != self._ns or \
                            func.shape[1] != self._na:
                raise ValueError('Policy.expectation -> invalid function')

            ef = (self._dist * func).sum(axis=1)

            return ef

        if not isinstance(model, models.POMDP):
            raise TypeError('Policy.expectation -> invalid argument type')

        c = (self._dist * model.c()).sum(axis=1)
        p = np.zeros((self._ns, self._ns))

        for a in model.list_actions():
            a_idx = model.get_action_index(a)
            p += self._dist[:, a_idx].reshape((model.n_states(), 1)) * model.p(a=a)

        return c, p

    # -- End: expectation

    def evaluate(self, m, cost=True):
        """
        P.evaluate(M) returns the cost-to-go function for policy P in the MDP M. The result is a |X| x 1 array,
        corresponding to J.

        P.evaluate(M, cost=False) instead returns the Q-function for policy P.

        :param m: MDP
        :param cost: bool (if False returns Q-function instead of cost-to-go)
        :return: np.ndarray (cost-to-go or Q-function)
        """

        cpi, ppi = self.expectation(model=m)

        j = np.dot(la.inv(np.eye(m.n_states()) - m.gamma() * ppi), cpi)

        if cost:
            return j

        q = np.zeros((m.n_states(), m.n_actions()))

        for a in m.list_actions():
            a_idx = m.get_action_index(a)

            q[:, a_idx] = m.c(a) + m.gamma() * np.dot(m.p(a=a), j)

        return q

    # -- End: evaluate

    def __eq__(self, p):
        """
        This method overrides the inherited method __eq__ for the object class. It returns True if C and C2 are the
        same policy (exactly).

        :param: Policy
        :return: bool
        """

        # noinspection PyUnresolvedReferences
        return (self._dist == p.probability()).all()

    # -- End: __eq__

    def __str__(self):
        """
        P.__str__() returns the external representation of policy P (string).

        :return: string (representation of policy P).
        """

        s = ''

        for x in self._states:
            s = s + str(self._list_states[x]) + ' -> '

            # Find maximal actions

            p = self._dist[x, :]
            idx = np.argwhere(p == np.max(p))
            idx = idx.flatten()

            if len(idx) > 1:
                s += '{'

            for i in idx:
                s = s + str(self._actions[i]) + ', '

            s = s[:-2]

            if len(idx) > 1:
                s += '}'

            s += '\n'

        return s

        # -- End: __str__

class Uniform(Policy):

    def __init__(self, m, feature_map=mdp_identity):
        """
        Uniform(M, feature_map=F)

        Builds a uniform MDP policy with feature map F.

        :param m: POMDP (or subclass thereof)
        :param feature_map: function that maps inputs to a column array of features
        """

        weights = np.ones((m.n_states(), m.n_actions())) / m.n_actions()

        super().__init__(m, weights=weights, feature_map=feature_map, immutable=True)

        # -- End: __init__

class Random(Policy):
    def __init__(self, m, feature_map=mdp_identity):
        """
        Random(M, feature_map=F)

        Builds a random MDP policy with feature map F.

        :param m: POMDP (or subclass thereof)
        :param feature_map: function that maps inputs to a column array of features
        """

        weights = rnd.rand(m.n_states(), m.n_actions())
        s = np.reshape(np.sum(weights, axis=1), (m.n_states(), 1))
        weights /= s

        super().__init__(m, weights=weights, feature_map=feature_map, immutable=True)

        # -- End: __init__

class Boltzmann(Policy):

    def __init__(self, m, weights, t=1, feature_map=mdp_identity, immutable=False):
        """
        Boltzmann(M, W, t=T, feature_map=F)

        Builds an Boltzmann policy for MDPs from the weights W using temperature T and using feature map F.

        :param m: POMDP (or subclass thereof)
        :param weights: value over actions for each input state;
        :param feature_map: function that maps inputs to a column array of features
        :param immutable: bool (determines whether weights are deep copied or not)
        """

        if t < 0:
            raise ValueError('Boltzmann -> invalid parameter')

        w = softmin(weights, t)

        super().__init__(m, weights=w, feature_map=feature_map, immutable=immutable)

        self._values = weights
        self._temp = t

    # -- End: __init__

    def temperature(self):
        """
        B.temperature() returns the temperature value associated with the Boltzmann policy B.

        :return: float (temperature)
        """

        return self._temp

    # -- End: temperature

    def set_temperature(self, t):
        """
        B.set_temperature(t) sets the value of the temperature parameter to t (and recomputes the policy).

        :param t: float (temperature)
        """

        if not isinstance(t, (int, float)) or t < 0:
            raise ValueError('Boltzmann.set_temperature -> invalid temperature parameter')

        self._temp = t

        w = softmin(self._values, t)
        self._dist = w
        self._labels = self._dist

        # -- End: set_temperature

class Greedy(Policy):

    def __init__(self, m, weights, eps=0, feature_map=mdp_identity, immutable=False):
        """
        Greedy(M, W, eps=E, feature_map=F)

        Builds an (eps-)greedy policy for MDPs from the weights W using exploration parameter E and feature map F.

        :param m: POMDP (or subclass thereof)
        :param weights: value over actions for each input state;
        :param feature_map: function that maps inputs to a column array of features
        :param immutable: bool (determines whether weights are deep copied or not)
        """

        if eps < 0:
            raise ValueError('Greedy -> invalid parameter')

        p = weights.min(axis=1).reshape((m.n_states(), 1))
        p = (weights == p).astype(int)
        s = p.sum(axis=1).reshape((m.n_states(), 1))
        p = p / s

        # Add exploration

        p = (1 - eps) * p + eps / m.n_actions()

        # noinspection PyTypeChecker
        super().__init__(m, weights=p, feature_map=feature_map, immutable=immutable)

        self._eps = eps
        self._values = weights

    # -- End: __init__

    def exploration(self):
        """
        G.exploration() returns the exploration parameter associated with the Greedy policy G.

        :return: float (exploration)
        """

        return self._eps

    # -- End: exploration

    def set_exploration(self, e):
        """
        G.set_exploration(t) sets the value of the exploration parameter to e (and recomputes the policy).

        :param e: float (exploration)
        """

        if not isinstance(e, (int, float)) or e < 0:
            raise ValueError('Greedy.set_exploration -> invalid exploration parameter')

        p = (self._dist - self._eps / self.n_actions()) / (1 - self._eps)

        self._eps = e

        self._dist = (1 - e) * p + e / self.n_actions()
        self._labels = self._dist

        # -- End: set_exploration

#  = = = = =  S O L V E R   F U N C T I O N S  = = = = = #

def value_iteration(m, pol=None, cost=True, as_mdp=False, max_iter=MAX_ITER):
    """
    value_iteration(M) returns the optimal Q-values for the (PO)MDP M.
    value_iteration(M, P) returns the cost-to-go associated with the policy P for the (PO)MDP M.
    value_iteration(M, P, cost=False) returns the Q-function associated with the policy P form the (PO)MDP M.
    value_iteration(M, P, as_mdp=True) returns the cost-to-go function associated with the policy P form the MDP M
     (even ifM is a POMDP).

    :param m: (PO)MDP
    :param pol: Controller/Policy
    :param cost: bool
    :param as_mdp: bool
    :param max_iter: int
    :return: np.ndarray
    """

    def value_update(qold):
        """
        value_update(m, q) produces an updated version of q for the MDP M. This update is for MDPs.

        :param qold: np.ndarray (Q-function)
        :return: np.ndarray (updated Q-function)
        """

        qnew = np.zeros([m.n_states(), m.n_actions()])

        # VI update
        for a in m.list_actions():
            act = get_index(a, m.list_actions())
            qnew[:, act] = m.c(a) + m.gamma() * np.dot(m.p(a=a), qold.min(axis=1))

        return qnew

    # -- End: value_update

    # Argument check

    if not isinstance(m, models.POMDP):
        raise ValueError('value_iteration -> invalid argument type (MDP expected)')

    if isinstance(m, models.MDP):
        is_mdp = True
    else:
        is_mdp = as_mdp

    # Prediction

    if pol is not None:
        if not isinstance(pol, (Controller, Policy)):
            raise TypeError('vi -> invalid policy')

        if is_mdp:
            return pol.evaluate(m, cost=cost)
        else:
            return pol.evaluate(m)

    # Control

    eps = 0.5 * MAX_ERROR / m.gamma() * (1 - m.gamma())

    # Initialization

    if is_mdp:
        q = np.zeros((m.n_states(), m.n_actions()))
    else:
        q = None

    it = 0
    qt = False

    # VI cycle

    while not qt:

        # Update
        if is_mdp:
            qupd = value_update(q)
        else:
            qupd, lb, _ = incremental_pruning(m, vectors=q, pruned=True)

        # Stopping condition
        if q is not None and (q.shape == qupd.shape):
            qt = la.norm(qupd - q, ord=2) < eps
        else:
            qt = (it + 1) >= max_iter

        q = qupd
        it += 1

    if is_mdp:
        return q, it
    else:
        return q, lb, it

# -- End: value_iteration

def policy_iteration(m, as_mdp=False, max_iter=MAX_ITER):
    """
    policy_iteration(m) receives a POMDP m and runs policy iteration on m. If m is an MDP, it returns a standard
     policy. If m is a POMDP, it returns a FSC. The maximum number of iterations is limited by MAX_ITER. The function
     also returns the number of iterations.

    :param m: model.POMDP
    :param as_mdp: bool
    :param max_iter: int
    :return: (Controller, int) (optimal policy or FSC for m and number of iterations)
    """

    if not isinstance(m, models.POMDP):
        raise ValueError('policy_iteration -> invalid argument type (MDP expected)')

    # Initialization

    if isinstance(m, models.MDP) or as_mdp:
        pol = Uniform(m)
        is_mdp = True
    else:
        pol = Controller(m)
        is_mdp = False

    it = 0
    qt = False

    # PI cycle

    while not qt:
        if is_mdp:
            # Policy evaluation
            q = pol.evaluate(m, cost=False)

            # Policy improvement
            # noinspection PyTypeChecker
            polnew = Greedy(m, q)
        else:
            # Policy evaluation
            l = pol.labels()
            s = pol.successors()
            q = pol.evaluate(m)

            # Policy improvement
            qnew, lnew, snew = incremental_pruning(m, vectors=q, pruned=True)
            # noinspection PyTypeChecker
            polnew = extract_controller(m, q, l, s, qnew, lnew, snew)

        it += 1
        qt = (pol == polnew) or (it >= max_iter)
        pol = polnew

    return pol, it

# -- End: policy_iteration

# noinspection PyUnresolvedReferences
def linear_programming(m):
    """
    linear_programming(m) receives a (PO)MDP m as an argument and produces the optimal value function using linear
     programming. It also provides the number of iterations used to compute it.

    :param m: POMDP
    :return: (np.ndarray, int) (optimal Q-function and n. of iterations)
    """

    if not isinstance(m, models.POMDP):
        raise ValueError('linear_programming -> invalid argument type (MDP expected)')



    if sol.status == 0:
        # print('\nLP: ' + sol.message)
        # print('    Finished in', sol.nit, 'iterations.')

        j = sol.x

        q = np.zeros((m.n_states(), m.n_actions()))

        for a in m.list_actions():
            act = get_index(a, m.list_actions())
            q[:, act] = m.c(a) + m.gamma() * np.dot(m.p(a=a), j)

        return q, sol.nit  # Number of iterations
    else:
        raise ValueError('linear_programming -> ' + sol.message)

# -- End: linear_programming

def incremental_pruning(m, vectors=None, pruned=True):
    """
    incremental_prunning(m, vectors=V, labels=L, pruned=P) receives as input a POMDP m, a set of a-vectors V and
     corresponding action labels L and computes an iteration of incremental prunning. It returns the updated set of
     vectors, the corresponding labels, and a set of successor nodes (for policy graph representations).

    :param m: POMDP
    :param vectors: np.ndarray (a-vectors, one per column)
    :param pruned: bool (indicates whether the updated set of a-vectors should be pruned or not
    :return: (np.ndarray, list, list of lists)
    """

    def remove_duplicates(a_arr):
        """
        remove_duplicates(a_arr) receives a set of a-vectors a_arr and returns that same set after removing duplicate
        entries. It also returns a list of indices of the vectors kept (for external use).

        :param a_arr: np.ndarray (a-vectors, one per column)
        :return: (np.ndarray, list)
        """

        # Sort columns lexicographically
        output_idx = np.lexsort(a_arr[::-1])
        a_arr = a_arr[:, output_idx]

        # Adjacent column difference (duplicate columns yield an all-zeros column)
        diff = np.diff(a_arr, axis=1)

        # Initialize index vector all to true with as many entries as columns
        idx = np.ones(a_arr.shape[1], 'bool')

        # Excluding the first column (diff has one less column), set all-zeros column to false
        idx[1:] = (diff != 0).any(axis=0)

        # Finally, return selected columns
        return a_arr[:, idx], output_idx[idx]

    # -- End: remove_duplicates

    # noinspection PyUnresolvedReferences
    def dominate(v, a_set):
        """
        dominate(v, a_set) receives an a-vector v and a set of a-vectors a_set and returns a witness for v (i.e., a
         point in the ns-simplex in which v dominates a_set (ns is the dimension of the a-vectors). If no such point is
         found, the function returns None.

        :param v: np.ndarray (a-vector)
        :param a_set: np.ndarray (set of a-vectors)
        :return: np.ndarray (witness)
        """

        # Remove a from aset
        # noinspection PyUnresolvedReferences
        a_set = a_set[:, (a_set != v).any(axis=0)]

        # Number of states and constraints
        ns, nc = a_set.shape

        # Cost vector
        c = np.zeros(ns + 1)
        c[-1] = -1

        # Find witness
        if nc > 0:
            # . domination constraint
            a_ub = np.append(v - a_set, np.ones((1, nc)), axis=0).transpose()
            b_ub = np.zeros(nc)
        else:
            x = np.zeros((1, ns))
            x[0, 0] = 1
            return x

        # . distribution constraints
        a_eq = np.ones((1, ns + 1))
        a_eq[0, -1] = 0
        b_eq = np.array([[1]])

        # Collect matrices

        sol = opt.linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq)

        if sol.success:
            if sol.x[-1] > 0:
                return sol.x[:-1]
            else:
                return None
        else:
            return None

    # -- End: dominate

    def prune(f):
        """
        prune(f) receives a set of a-vectors and returns that set after pruning dominated vectors. It also returns an
         index list for external indexation.

        :param f: np.ndarray (set of a-vectors)
        :return: (np.ndarray, list)
        """

        if not (isinstance(f, np.ndarray)):
            raise TypeError('prune -> invalid argument type')

        if len(f.shape) != 2:
            raise ValueError('prune -> invalid argument dimensions')

        if f.shape[1] == 0:
            return f

        # Dimensions
        ns = f.shape[0]
        base_idx = list(range(f.shape[1]))

        # Save minima at the simplex vertices
        idx = np.unique(f.argmin(axis=1))
        w_out = np.reshape(f[:, idx], (ns, idx.size))
        output_idx = idx.tolist()

        f = np.delete(f, idx, axis=1)
        for i in sorted(idx, reverse=True):
            del(base_idx[i])

        while f.shape[1] > 0:
            f0 = np.reshape(f[:, 0], (ns, 1))

            # Find witness beliefs for each vector in f
            b = dominate(f0, w_out)

            if b is None:
                f = f[:, 1:]
                del(base_idx[0])
            else:
                idx = np.dot(b, f).argmin()
                f_idx = np.reshape(f[:, idx], (ns, 1))

                w_out = np.append(w_out, f_idx, axis=1)

                # noinspection PyUnresolvedReferences
                output_idx.append(base_idx[idx])

                f = np.delete(f, idx, axis=1)
                del(base_idx[idx])

        return w_out, output_idx

    # -- End: prune

    def oplus(a, b):
        """
        oplus(a, b) receives two sets of a-vectors, a and b, and outputs the cross sum of all vectors in a and b,
         prunning those that are strictly dominated. It also returns a double list with the indices of a and b that
         were used in each of the output vectors.

        :param a: np.ndarray (set of a-vectors)
        :param b: np.ndarray (set of a-vectors)
        :return: (np.ndarray, [list, list])
        """

        # -- Naive outer sum of a and b

        ns1, na = a.shape
        ns2, nb = b.shape

        if ns1 != ns2:
            raise ValueError('oplus -> vector dimensions must agree')
        else:
            ns = ns1

        a_idx = list(range(na))
        b_idx = list(range(nb))

        if na == 0:
            d = b

        elif nb == 0:
            d = a

        else:
            # Get all possible combinations of indices

            a_idx, b_idx = np.meshgrid(a_idx, b_idx)
            a_idx = a_idx.flatten().tolist()
            b_idx = b_idx.flatten().tolist()

            # Sum
            foo = a[:, a_idx] + b[:, b_idx]

            # Remove duplicate entries
            d, i = remove_duplicates(np.array(foo))
            a_idx = [a_idx[j] for j in i]
            b_idx = [b_idx[j] for j in i]

        if not pruned:
            return d, [a_idx, b_idx]

        # -- Empty set

        hat_d = np.ndarray((ns, 0))
        hat_a = []
        hat_b = []

        # -- Add and prune

        while d.shape[1] > 0:
            f0 = np.reshape(d[:, 0], (ns, 1))

            # Find witness belief for each vector in d
            b = dominate(f0, hat_d)
            if b is None:
                d = d[:, 1:]
                del a_idx[0]
                del b_idx[0]
            else:
                idx = np.dot(b, d).argmin()
                f_idx = np.reshape(d[:, idx], (ns, 1))

                hat_d = np.concatenate((hat_d, f_idx), axis=1)
                hat_a.append(a_idx[idx])
                hat_b.append(b_idx[idx])

                d = np.delete(d, idx, axis=1)
                del a_idx[idx]
                del b_idx[idx]

        return hat_d, [hat_a, hat_b]

    # -- End: oplus

    ns = m.n_states()

    # First iteration is treated separately

    if vectors is None:
        g = m.c()
        l = m.list_actions()
        s = [[] * len(l)]

        return g, l, s

    # Check parameters

    if vectors.shape[0] != ns:
        raise ValueError('incremental_pruning -> invalid a-vector dimension')

    nv = vectors.shape[1]

    # Generate Gamma_az

    g_aux = vectors
    s_idx = np.array(range(nv), ndmin=2)

    g = None
    l = None
    s = None

    for a in m.list_actions():
        a_idx = m.get_action_index(a)

        for z in m.list_observations():
            z_idx = m.get_observation_index(z)

            caux = np.reshape(m.c(a), (ns, 1))
            oaux = np.reshape(m.o(a=a, z=z), (ns, 1))

            g_az = caux / m.n_observations() + m.gamma() * m.p(a=a).dot(oaux * g_aux)

            if z_idx == 0:
                g_a = g_az
                s_a = np.copy(s_idx)
            else:
                g_a, idx = oplus(g_a, g_az)
                s_a = np.concatenate([s_a[:, idx[0]], s_idx[:, idx[1]]], axis=0)

        if a_idx == 0:
            g = g_a
            l = [a] * g_a.shape[1]
            s = s_a.T.tolist()
        else:
            g = np.concatenate([g, g_a], axis=1)
            l += [a] * g_a.shape[1]
            s += s_a.T.tolist()

    if pruned:
        g, idx = prune(g)
        l = [l[j] for j in idx]
        s = [s[j] for j in idx]

    return g, l, s

# -- End: incremental_pruning

def perseus(m, pointset = None, successors_out=False, max_iter = MAX_ITER, max_samples=MAX_POINTS):
    """
    perseus(m, pointset=None) receives as input a POMDP m and returns the set of alpha-vectors computed using PERSEUS.
    The algorithm builds a set of sampled belief points used in the computation. Only alpha-points relevant at those
    sample beliefs are used. The set of samples can be pre-specified as the pointset argument.

    References:

    M. Spaan, N. Vlassis. Perseus: Randomized point-based value iteration for POMDPs. In "Journal of Artificial
    Intelligence Research", vol. 24, pp. 195-220, 2005.

    :param m: POMDP
    :param pointset: np.ndarray (belief points, one per row)
    :param successors_out: boolean (if True, function returns the successor states)
    :param max_iter: int
    :return: np.ndarray (alpha-vectors, one per column)
    """

    # Stopping condition

    #eps = 0.5 * MAX_ERROR / m.gamma() * (1 - m.gamma())
    eps = 0.01

    # Get beliefs

    if pointset is None:
        print('Sampling...', flush=True, end=' ')
        pointset = sample_beliefs(m, max_samples)
        print('Complete. Running...', flush=True)
    else:
        print('Running...', flush=True)

    # Initialize alpha-vectors as an upper-bound

    aux = m.c().max(axis=0)
    aux = aux.max()
    act = int(aux.argmax())
    g_curr = np.ones((m.n_states(), 1)) * aux / (1 - m.gamma())
    l_curr = [m.get_action(act)]
    s_curr = [[0] * m.n_observations()]

    # Initialize cycle

    quit = False
    i = 0

    while not quit:

        # Current function
        j = pointset.dot(g_curr).min(axis=1)

        # Perform backup
        g_new, l_new, s_new = pbvi_backup(m, g_curr, l_curr, s_curr, pointset)

        # Check if stopping condition is met
        j_new = pointset.dot(g_new).min(axis=1)

        err = la.norm(j_new - j)

        quit = err < eps or i + 1 == max_iter
        i += 1
        g_curr = g_new
        l_curr = l_new
        s_curr = s_new

        if i % 1000 == 0:
            print('Completed', i, 'iterations so far, using', g_curr.shape[1], 'vectors. Current error is', err, '- still running...', flush=True)

    print('Complete after', i, 'iterations.', flush=True)

    if successors_out:
        return g_curr, l_curr, s_curr, i
    else:
        return g_curr, l_curr, i

# -- End: PERSEUS

def pbpi(m, pointset = None, max_iter=MAX_ITER):
    """
    policy_iteration(m) receives a POMDP m and runs policy iteration on m. If m is an MDP, it returns a standard
     policy. If m is a POMDP, it returns a FSC. The maximum number of iterations is limited by MAX_ITER. The function
     also returns the number of iterations.

    :param m: model.POMDP
    :param pointset: np.ndarray (belief points, one per row)
    :param max_iter: int
    :return: (Controller, int) (optimal policy or FSC for m and number of iterations)
    """

    # Get beliefs

    if pointset is None:
        print('Sampling...', flush=True, end=' ')
        pointset = sample_beliefs(m, MAX_POINTS)
        print('Complete. Running...', flush=True)
    else:
        print('Running...', flush=True)

    pol = Controller(m)

    quit = False
    i = 0

    # PI cycle

    while not quit:

        g_curr = pol.evaluate(m)
        l_curr = pol.labels()
        s_curr = pol.successors()

        # Policy improvement
        g_new, l_new, s_new = pbvi_backup(m, g_curr, l_curr, s_curr, pointset, update_all=True)
        polnew = extract_controller(m, g_curr, l_curr, s_curr, g_new, l_new, s_new)

        i += 1
        quit = (pol == polnew) or (i >= max_iter)
        pol = polnew

        if i % 25 == 0:
            print('Completed', i, 'iterations so far. Still running...', flush=True)

    print('Complete after', i, 'iterations.', flush=True)

    return pol, i

# -- End: PBPI

def draw(c, fname=None, with_labels=True, node_size=800, arrows=True):
    """
    Draws the controller f as a graph.

    :param c: Controller
    :param fname: str
    """

    g = nx.DiGraph()
    l = {}
    for s in range(c.n_states()):
        g.add_node(s)
        l[s] = str(c.get_action(s))[:6]

    for s in range(c.n_states()):
        for z in range(c.n_observations()):
            new_edge = s, c.get_successor(s, z)
            if new_edge in g.edges():
                g[s][c.get_successor(s, z)]['obs']+=', ' + str(c.list_observations()[z])
            else:
                g.add_edge(s, c.get_successor(s, z), obs=str(c.list_observations()[z]))

    pos = nx.spring_layout(g)

    plt.figure()

    n = nx.draw_networkx_nodes(g, pos, node_size=node_size)
    n.set_edgecolor('k')
    n.set_facecolor('w')
    nx.draw_networkx_edges(g, pos, width=0.5, arrows=arrows)

    if with_labels:
        nx.draw_networkx_labels(g, pos, l, font_size=8)

    plt.axis('off')
    plt.show()

    if fname is not None:
        plt.savefig(fname + '.pdf')
        nx.write_graphml(g, fname + '.graphml')

    plt.close()

# -- End: draw
