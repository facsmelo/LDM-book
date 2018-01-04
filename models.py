import numpy as np
import numpy.random as rnd


#  = = = = =  C O N S T A N T S  = = = = =  #

PREAMBLE_LABELS = ('discount:', 'values:', 'states:', 'actions:', 'observations:')
INITIAL_LABELS = ('start:', 'start')
TOLERANCE = 1e-8

#  = = = = =  A U X I L I A R Y   F U N C T I O N S  = = = = =  #

def is_equal(x, y):
    """ is_equal : num x num -> bool

        is_equal(x, y) returns true if the difference between the two numbers, x and y, is smaller than some given
        parameter."""

    return abs(x - y) < TOLERANCE

# -- End: is_equal

def is_valid_item(item, item_list):
    """ is_valid_item : univ x tuple -> bool
    
        is_valid_item(i, l) returns True if i is a valid item in l or a valid
        index for an item in l."""

    return item in item_list or (isinstance(item, int) and 0 <= item < len(item_list))

# -- End: is_valid_item


def get_item(item, item_list):
    """ get_item : univ x tuple -> univ
    
        get_item(i, l) returns the item in l corresponding to i. If i is already
        an item, it returns i. If it's an index, it returns the corresponding 
        item."""
    
    if not is_valid_item(item, item_list):
        raise ValueError('get_item -> invalid index')
        
    if item in item_list:
        return item
    
    return item_list[item]
    
# -- End: get_item


def get_index(item, item_list):
    """ get_index : univ x tuple -> univ
    
        get_index(i, l) returns the index of item i in l. If i is already an 
        index, it returns i. If it's an item in l, it returns the corresponding 
        index."""

    if not is_valid_item(item, item_list):
        raise ValueError('get_index -> invalid item')

    if item in item_list:
        return item_list.index(item)
    
    return item
    
# -- End: get_index


def sample(items, p=None):
    """ sample : tuple x list -> univ
        sample(i, p) samples an item from i using distribution p."""
    
    if p is None:
        p = [1/len(items)] * len(items)

    aux = rnd.choice(list(range(len(items))), size=1, p=p)[0]
    return items[aux]

# -- End: sample
    

def is_stochastic_matrix(m, n, p):
    """ is_stochastic_matrix : int x int x ndarray -> bool
    
        is_stochastic_matrix(m, n, P) returns True if P is a stochastic matrix
        and False otherwise."""
    
    if not isinstance(p, np.ndarray):
        return False
    
    if p.shape != (m, n):
        return False

    # noinspection PyTypeChecker
    if not all(is_equal(np.sum(p, axis=1), 1)):
        print(np.sum(p, axis=1))
        return False
    
    return True

# -- End: is_stochastic_matrix


# noinspection PyTypeChecker
def is_valid_cost(m, n, c):
    """ is_valid_cost : int x int x ndarray -> bool
    
        is_valid_cost(m, n, C) returns True if C is a valid cost function
        and False otherwise."""
    
    if not isinstance(c, np.ndarray):
        return False
    
    if c.shape != (m, n):
        return False
            
    if np.any(np.any(c > 1) == 1):
        return False
    
    if np.any(np.any(c < 0) == 1):
        return False

    return True

# -- End: is_valid_cost


def get_stochastic_family(items, m, n, p):
    """ get_stochastic_family : tuple x int x int x ndarray -> dict
    
        get_stochastic_family(P) returns a dictionary containing in each entry 
        the stochastic matrix in element i of the 3d array P."""
    
    if not isinstance(p, np.ndarray):
        return dict()
            
    if len(items) == 0:
        if p.shape != (m, n):
            return dict()
        
        if not is_stochastic_matrix(m, n, p):
            return dict()
        
        return p
    else:        
        if p.shape != (len(items), m, n):
            return dict()

        family = dict()
        
        for a in range(len(items)):
            if not is_stochastic_matrix(m, n, p[a]):
                return dict()
            
            family[items[a]] = p[a]
        
        return family

# -- End: get_stochastic_family


def get_enumeration(s):
    """ get_enumeration : {int, seq} -> tuple

        get_enumeration(n) returns an enumeration of all integers between 0 and
        n.
        
        get_enumeration(e_1,...,e_n) returns the tuple (e_1, ..., e_n)."""
    
    if isinstance(s, int):       
        return tuple(range(s))

    if isinstance(s, (tuple, list)):
        return tuple(s)

    return -1

# -- End: get_enumeration


def get_initialization(items, initial):
    """ get_initialization : tuple x {int, ndarray} -> ndarray
        
        gt_initialization(i, p) builds a valid distribution over the set of 
        items i from p."""
        
    if np.array_equal(initial, np.array([])):
        return np.ones(len(items)) / len(items)

    if not isinstance(initial, np.ndarray):
        return np.array([])
        
    if initial.shape != (len(items),):
        return np.array([])
        
    if not is_equal(np.sum(initial), 1):
        return np.array([])
    
    return initial
        
# -- End: get_initialization


def belief_update(m, b, a, z):
    """
    belief_update(m, b, a, z) updates belief b given the action a and observation z.

    :param m: pomdp
    :param b: np.ndarray
    :param a: int or action
    :param z: int or observation
    :return: np.ndarray
    """

    if not isinstance(m, POMDP):
        raise ValueError('belief_update -> invalid POMDP model')

    if not isinstance(b, np.ndarray):
        raise ValueError ('belief_update -> invalid belief vector')

    if b.shape != (m.n_states(),) and b.shape != (1, m.n_states()):
        raise ValueError('belief_update -> invalid belief vector')

    if b.sum() != 1:
        raise ValueError('belief_update -> invalid belief vector')

    try:
        a = m.get_action(a)
    except:
        raise ValueError('belief_update -> invalid action')

    try:
        z = m.get_observation(z)
    except:
        raise ValueError('belief_update -> invalid observation')

    b_new = b.dot(m.p(a=a)) * m.o(a=a, z=z)
    return b_new / b_new.sum()

# -- End: belief_update

#  = = = = =  P O M D P   C L A S S  = = = = = #
#
# This is a huge class that yields
#
# . MDPs
# . HMMs
# . MCs
#
# as subclasses.
#
#  = = = = = = = = = = = = = = = = = = = = = = #


class POMDP:
    """ POMDP class.
    
        POMDP : enumeration x enumeration x enumeration x ndarray x ndarray 
                                          x ndarray x float

        POMDP : enumeration x enumeration x enumeration x ndarray x ndarray 
                                          x ndarray x float x ndarray
                                          
        POMDP(X, A, Z, P, O, C, gamma) defines a POMDP with states X, actions A,
        observations Z, transition probabilities P, observation probabilities O,
        cost function C and discount gamma. In particular:
        
        - X is an enumeration (tuple or list) of states;
        - A is an enumeration (tuple or list) of actions;
        - Z is an enumeration (tuple or list) of observations;
        - P is a 3D numpy array, where P[a][x][y] = P(y | x, a)
        - O is a 3D numpy array, where O[a][x][z] = O(z | x, a)
        - C is a 2D numpy array, where C[x][a] = c(x, a)
        - gamma is a float, corresponding to the discount.
        
        POMDP(k, m, n, P, O, C, gamma) defines a POMDP where X, A and Z are now 
        the integers between 0 and k, m and n, respectively.
        
        POMDP(., ., ., P, O, C, gamma, init) additionally initializes the state
        distribution to init, where init is a 1D numpy array.
        
        *IMPORTANT NOTE* 
        The constructor does not build copies of the transition, observation and
        cost matrices, P, O, C. Instead, the actual matrices are linked to the 
        model."""
    
    def __init__(self, states, actions, observations, p, o, c, discount, init=np.array([])):
        """ __init__ : enumeration x enumeration x enumeration x ndarray 
                                   x ndarray x ndarray x float -> POMDP

            POMDP(X, A, Z, P, O, C, gamma) defines a POMDP with states X, 
            actions A, observations Z, transition probabilities P, observation 
            probabilities O, cost function C and discount gamma."""
        
        # Check states
        
        aux = get_enumeration(states)
        
        if aux == -1:
            raise ValueError('pomdp -> invalid operand')
        
        self._states = aux
        self._nX = len(aux)
        
        # Check actions
        
        aux = get_enumeration(actions)
        
        if aux == -1:
            raise ValueError('pomdp -> invalid operand')
        
        self._actions = aux
        self._nA = len(aux)
        
        # Check observations
        
        aux = get_enumeration(observations)
        
        if aux == -1:
            raise ValueError('pomdp -> invalid operand')
        
        self._observations = aux
        self._nZ = len(aux)
        
        # Check transition probabilities
        
        aux = get_stochastic_family(self._actions, self._nX, self._nX, p)
        
        if aux == {}:
            raise ValueError('pomdp -> invalid operand')
        
        self._P = aux

        # Check observation probabilities
        
        aux = get_stochastic_family(self._actions, self._nX, self._nZ, o)
        
        if aux == {}:
            raise ValueError('pomdp -> invalid operand')
        
        self._O = aux
        
        # Check cost
        
        if not is_valid_cost(self._nX, self._nA, c):
            raise ValueError('pomdp -> invalid operand')
        
        self._c = c
        
        # Check discount
        
        if not isinstance(discount, (int, float)):
            raise ValueError('pomdp -> invalid operand')
        
        if discount < 0 or discount > 1:
            raise ValueError('pomdp -> invalid operand')
        
        self._gamma = discount
        
        # Check initial distribution
        
        aux = get_initialization(self._states, init)
        
        if np.array_equal(aux, np.array([])):
            raise ValueError('pomdp -> invalid operand')
        
        self._init = aux
        self._B = aux.reshape((1, self._nX))
        self._X = sample(self._states, self._B.flatten().tolist())
        
        self._sampling_policy = None
        
    # -- End: __init__
    
    
    def n_states(self):
        """ n_states : POMDP -> int

            M.n_states() returns the number of states in M."""
        
        return self._nX
    
    # -- End: n_states
    
    
    def n_actions(self):
        """ n_actions : POMDP -> int

            M.n_actions() returns the number of actions in M."""
        
        return self._nA
    
    # -- End: n_actions


    def n_observations(self):
        """ n_observations : POMDP -> int

            M.n_observations() returns the number of observations in M."""
        
        return self._nZ
    
    # -- End: n_observations


    def list_states(self):
        """ list_states : POMDP -> tuple

            M.list_states() returns a tuple listing the states of M."""
        
        return self._states
    
    # -- End: list_states
    
        
    def list_actions(self):
        """ list_actions : POMDP -> tuple

            M.list_actions() returns a tuple listing the actions of M."""
        
        return self._actions

    # -- End: list_actions
    
    
    def list_observations(self):
        """ list_observations : POMDP -> tuple

            M.list_observations() returns a tuple listing the observations of 
            M."""
        
        return self._observations

    # -- End: list_observations
    
    
    def get_state(self, x):
        """ get_state : POMDP x univ -> univ

            M.get_state(x) returns the state representation for x. If x is 
            already a state representation, then it returns x. If it's a state
            index, it returns the xth state."""

        try:
            return get_item(x, self._states)
        except:
            raise ValueError('get_state -> invalid index')
                    
    # -- End: get_state
        

    def get_action(self, a):
        """ get_action : POMDP x univ -> univ

            M.get_action(a) returns the action representation for a. If a is 
            already an action representation, then it returns a. If it's an 
            action index, it returns the ath action."""
        
        try:
            return get_item(a, self._actions)
        except:
            raise ValueError('get_action -> invalid index')

    # -- End: get_action
    

    def get_observation(self, z):
        """ get_observation : POMDP x univ -> univ

            M.get_observation(z) returns the observation representation for z. 
            If z is already an observation, then it returns z. If it's an index,
            it returns the zth observation."""
        
        try:
            return get_item(z, self._observations)
        except:
            raise ValueError('get_observation -> invalid index')
        
    # -- End: get_observation


    def get_state_index(self, x):
        """ get_state_index : POMDP x univ -> int

            M.get_state_index(x) returns the state index for x. If x is already 
            a state index, then it returns x. If it's a state representation, it 
            returns the corresponding index."""
        
        try:
            return get_index(x, self._states)
        except:
            raise ValueError('get_state_index --> invalid item')
        
    # -- End: get_state_index
    
        
    def get_action_index(self, a):
        """ get_action_index : POMDP x univ -> int

            M.get_action_index(a) returns the action index for a. If a is 
            already an action index, then it returns a. If it's an action, it 
            returns the corresponding index."""

        try:
            return get_index(a, self._actions)
        except:
            raise ValueError('get_action_index --> invalid item')

    # -- End: get_action_index


    def get_observation_index(self, z):
        """ get_observation_index : POMDP x univ -> int

            M.get_observation_index(z) returns the observation index for z. If z 
            is already an observation index, then it returns z. If it's an 
            observation, it returns the corresponding index."""

        try:
            return get_index(z, self._observations)
        except:
            raise ValueError('get_observation_index --> invalid item')
    
    # -- End: get_observation_index


    def p(self, **kwargs):
        """ P : POMDP x dict -> {dict, ndarray, float}
        
            M.P() returns the transition probabilities for M (a dictionary)
            
            M.P(a = A) returns the transition probabilities P_A, for action 
            A (a 2D numpy array)
            
            M.P(a = A, x = X) returns the transition probabilities P(. | X, A),
            for action A and state X (a 1D numpy array)
            
            M.P(a = A, x = X, y = Y) returns the transition probabilities 
            P(Y | X, A) (a float)."""
        
        if len(kwargs) == 0:
            return self._P
        
        if 'a' in kwargs:
            a = self.get_action(kwargs['a'])
            if len(kwargs) == 1:
                return self._P[a]
        else:
            raise ValueError('P -> invalid indexation (action missing)')
        
        if 'x' not in kwargs:
            raise ValueError('P -> invalid indexation (state missing)')
        
        x = self.get_state_index(kwargs['x'])
        
        if 'y' not in kwargs:
            return self._P[a][x]
        else:
            y = self.get_state_index(kwargs['y'])            
            return self._P[a][x][y]
    
    # -- End: P

 
    def o(self, **kwargs):
        """ O : POMDP x dict -> {dict, ndarray, float}
        
            M.O() returns the observation probabilities for M (a dictionary)
            
            M.O(a = A) returns the observation probabilities O_A, for action 
            A (a 2D numpy array)
            
            M.O(a = A, x = X) returns the observation probabilities O(. | X, A),
            for action A and state X (a 1D numpy array)
            
            M.O(a = A, z = Z) returns the probability of observation Z, O(Z | ., A),
            given action A (a 1D numpy array)

            M.O(a = A, x = X, z = Z) returns the observation probabilities
            O(Z | X, A) (a float)."""
        
        if len(kwargs) == 0:
            return self._O
        
        if 'a' in kwargs:
            a = self.get_action(kwargs['a'])
            if len(kwargs) == 1:
                return self._O[a]
        else:
            raise ValueError('O -> invalid indexation (action missing)')
        
        if 'x' not in kwargs and 'z' not in kwargs:
            raise ValueError('O -> invalid indexation (missing state or observation)')
        
        if 'x' in kwargs:
            x = self.get_state_index(kwargs['x'])
        
            if 'z' not in kwargs:
                return self._O[a][x]
            else:
                z = self.get_observation_index(kwargs['z'])
                return self._O[a][x][z]
        else:
            z = self.get_observation_index(kwargs['z'])
            return self._O[a][:, z]

    # -- End: O
    
        
    def c(self, *args):
        """ c : POMDP x tuple -> {ndarray, float}
        
            M.c() returns the cost function for M (a ndarray)
            
            M.c(A) returns the cost vector c_A, for action A (a 1D numpy array)
            
            M.c(X, A) returns the cost c(X, A) (a float)."""

        if len(args) == 0:
            return self._c
        
        if len(args) == 1:
            a = self.get_action_index(args[0])

            return self._c[:, a]
        
        if len(args) == 2:
            x = self.get_state_index(args[0])
            a = self.get_action_index(args[1])
            
            return self._c[x, a]

    # -- End: c
    
    
    def gamma(self):
        """ gamma : POMDP -> float

            M.gamma() returns the discount in M."""
        
        return self._gamma
    
    # -- End: gamma
    
    
    def get_current_state(self):
        """ get_current_state : POMDP -> univ

            M.get_current_state() returns the current state of M."""
        
        return self._X
    
    # -- End: get_current_state
    
    
    def get_current_belief(self):
        """ get_current_belief : POMDP -> ndarray

            M.get_current_belief() returns the current belief of M."""
        
        return self._B
    
    # -- End: get_current_belief


    def reset(self, new_belief=None):
        """
        Resets the state of the POMDP.

        :param new_belief: np.ndarray
        :return: None.
        """

        if new_belief is None:
            self._B = self._init.reshape((1,self._nX))
        else:
            if not isinstance(new_belief, np.ndarray) or new_belief.shape != (1, self._nX):
                raise ValueError('POMDP.reset -> invalid belief')

            self._B = new_belief

        self._X = sample(self._states, self._B.flatten().tolist())

    # -- End: reset


    def set_sampling_policy(self, pol):
        """
        m.set_sampling_policy(pol) sets the current sampling policy for m to pol.

        :param pol: Policy (solutions module)
        :return: None
        """

        self._sampling_policy = pol

    # -- End: set_sampling_policy


    def sample_transition(self, *args):
        """ sample_transition : POMDP x tuple -> tuple

            M.sample_transition(A) returns a transition obtained with action A. 
            The transition is a tuple (X, A, C, Y, Z), where X, Y are states, A 
            is an action, C is a cost and Z is an observation.
            
            M.sample_transition(X, A) returns a transition obtained from state X
            with action A."""            
            
        if args == ():
            raise ValueError('sample_transition --> action missing')
        
        if len(args) == 1:
            a = self.get_action(args[0])
            x = self._X

        elif len(args) == 2:
            x = self.get_state_index(args[0])
            a = self.get_action(args[1])
        else:
            raise ValueError('sample_transition -> invalid function call')

        y = sample(self.list_states(), list(self.p(x=x, a=a)))
        z = sample(self.list_observations(), list(self.o(x=y, a=a)))
        cost = self.c(x, a)
        
        return self.get_state(x), self.get_action(a), cost, self.get_state(y), self.get_observation(z)
    
    # -- End: sample_transition


    # -- Specialized Functions -- #
    
    def __len__(self):
        """ __len__ : POMDP -> int

            len(M) returns the dimension of M."""
        
        return self._nX * self._nA * self._nZ
    
    # -- End: __len__


    def __iter__(self):
        """ __iter__ : POMDP -> iterable

            iter(M) returns an iterable associated with M."""
            
        return self
    
    # -- End: __iter__
    
    
    def __next__(self):
        """ __next__ : POMDP -> tuple

            next(POMDP) iterates the POMDP (simulates a transition) using the
            current sampling policy. If the sampling policy is undefined, it
            selects a random action. It returns a pair (X, Z), corresponding to
            the new state and next observation."""
        
        if self._sampling_policy is None:
            action = sample(self.list_actions())
        else:
            action = self._sampling_policy.get_action(self.get_current_state())
            
        current_state = self.get_current_state()
        next_state = sample(self.list_states(), self.p(x=current_state, a=action).tolist())
        next_observation = sample(self.list_observations(), self.o(x=next_state, a=action).tolist())
        
        self._X = next_state

        b_new = self._B.dot(self.p(a=action)) * self.o(a=action, z=next_observation)
        self._B = b_new / b_new.sum()

        return next_state, next_observation
    
    # -- End: __next__

        
    def __str__(self):
        """ __str__ returns the "pretty" (external) representation of a 
            POMDP."""
        
        s = '('
        s = s + str(set(self._states)) + ', '
        s = s + str(set(self._actions)) + ', '
        s = s + str(set(self._observations)) + ', P, O, c, ' 
        s = s + str(self._gamma) + ')'
        
        return s
    
    # -- End: __str__

    # noinspection PyTypeChecker
    def __repr__(self):
        """ __repr__ returns the functional (internal) representation of a 
            POMDP."""
        
        s = '\nPOMDP('
        s = s + str(self._states) + ',\n'
        s = s + str(self._actions) + ',\n'
        s = s + str(self._observations) + ',\n'
        
        p = []
        o = []
        
        for action in self._actions:
            p += [self._P[action]]
            o += [self._O[action]]
            
        p = np.array(p)
        o = np.array(o)
        
        s = s + repr(p) + ',\n'
        s = s + repr(o) + ',\n'
        s = s + repr(self._c) + ',\n'
        
        s = s + repr(self._gamma) + ')'

        return s
    
    # -- End: __repr__
    

#  = = = = =  M D P   C L A S S  = = = = = #
#
# POMDP subclass that, in turn, yields
#
# . MCs
#
# as subclasses.
#
#  = = = = = = = = = = = = = = = = = = = = #

class MDP (POMDP):
    """ MDP class (inherits from POMDP)
    
        MDP : enumeration x enumeration x ndarray x ndarray x float

        MDP : enumeration x enumeration x ndarray x ndarray x float x ndarray
                                          
        MDP(X, A, P, C, gamma) defines an MDP with states X, actions A, 
        transition probabilities P, cost function C and discount gamma. In 
        particular:
        
        - X is an enumeration (tuple or list) of states;
        - A is an enumeration (tuple or list) of actions;
        - P is a 3D numpy array, where P[a][x][y] = P(y | x, a)
        - C is a 2D numpy array, where C[x][a] = c(x, a)
        - gamma is a float, corresponding to the discount.
        
        MDP(m, n, P, C, gamma) defines a MDP where X and A are now the integers 
        between 0 and m and n, respectively.
        
        MDP(., ., P, C, gamma, init) additionally initializes the state
        distribution to init, where init is a 1D numpy array."""
    
    def __init__(self, states, actions, p, c, discount, init=np.array([])):
        """ __init__ : enumeration x enumeration x ndarray x ndarray x float
                                          
            MDP(X, A, P, C, gamma) defines an MDP with states X, actions A, 
            transition probabilities P, cost function C and discount gamma."""
        
        # Build auxiliary observation matrix
        
        aux_s = get_enumeration(states)
        
        if aux_s == -1:
            raise ValueError('mdp -> invalid operand')
        
        aux_a = get_enumeration(actions)

        if aux_a == -1:
            raise ValueError('mdp -> invalid operand')
        
        o = [np.eye(len(aux_s))] * len(aux_a)
                     
        o = np.array(o)
        
        # Call super
        
        super().__init__(states, actions, states, p, o, c, discount, init)
        
    # -- End: __init__
    

    def sample_transition(self, *args):
        """ sample_transition : MDP x tuple -> tuple

            M.sample_transition(A) returns a transition obtained with action A. 
            The transition is a tuple (X, A, C, Y), where X, Y are states, A 
            is an action and C is a cost.
            
            M.sample_transition(X, A) returns a transition obtained from state X
            with action A."""            
        
        return super().sample_transition(*args)[:-1]

    # -- End: sample_transition
    
    
    # -- Specialized Functions -- #
    
    def __len__(self):
        """ __len__ : MDP -> int

            len(M) returns the dimension of M."""
        
        return self._nX * self._nA
    
    # -- End: __len__    
    
    def __next__(self):
        """ __next__ : MDP -> tuple

            next(MDP) iterates the MDP (simulates a transition) using the
            current sampling policy. If the sampling policy is undefined, it
            selects a random action. It returns a new state X."""

        return super().__next__()[0]
    
    # -- End: __next__

        
    def __str__(self):
        """ __str__ returns the "pretty" (external) representation of an MDP."""
        
        s = '('
        s = s + str(set(self.list_states())) + ', '
        s = s + str(set(self.list_actions())) + ', P, c, ' 
        s = s + str(self.gamma()) + ')'
        
        return s
    
    # -- End: __str__

    # noinspection PyTypeChecker
    def __repr__(self):
        """ __repr__ returns the functional (internal) representation of an 
            MDP."""
        
        s = '\nMDP('
        s = s + str(self.list_states()) + ',\n'
        s = s + str(self.list_actions()) + ',\n'
        
        p = []
        
        for action in self.list_actions():
            p += [self.p(a=action)]
            
        p = np.array(p)
        
        s = s + repr(p) + ',\n'
        s = s + repr(self.c()) + ',\n'
        
        s = s + repr(self.gamma()) + ')'

        return s
    
    # -- End: __repr__
    
    
    
    
#  = = = = =  H M M   C L A S S  = = = = = #
#
# Subclass of POMDPs
#
#  = = = = = = = = = = = = = = = = = = = = #


class HMM (POMDP):
    """ HMM class (inherits from POMDP).
    
        HMM : enumeration x enumeration x ndarray x ndarray

        HMM : enumeration x enumeration x ndarray x ndarray x ndarray
                                          
        HMM(X, Z, P, O) defines an HMM with states X, observations Z, 
        transition probabilities P and observation probabilities O. In 
        particular:
        
        - X is an enumeration (tuple or list) of states;
        - Z is an enumeration (tuple or list) of observations;
        - P is a 3D numpy array, where P[a][x][y] = P(y | x, a)
        - O is a 3D numpy array, where O[a][x][z] = O(z | x, a).
        
        HMM(m, n, P, O) defines an HMM where X and Z are now the integers 
        between 0 and m and n, respectively.
        
        HMM(., ., P, O, init) additionally initializes the state distribution to 
        init, where init is a 1D numpy array."""
    
    def __init__(self, states, observations, p, o, init=np.array([])):
        """ HMM : enumeration x enumeration x ndarray x ndarray

            HMM(X, Z, P, O) defines an HMM with states X, observations Z, 
            transition probabilities P and observation probabilities O."""
            
        # Check states
        
        aux_s = get_enumeration(states)
        
        if aux_s == -1:
            raise ValueError('hmm -> invalid operand')
        
        # Check observations
        
        aux_o = get_enumeration(observations)
        
        if aux_o == -1:
            raise ValueError('hmm -> invalid operand')
        
        # Simulate actions        
       
        actions = (0,)
        
        # Rebuild transition and observation matrices
        
        p = np.array([p])
        o = np.array([o])
        
        # Simulate cost & discount
        
        c = np.zeros((len(aux_s), 1))
        discount = 1
        
        super().__init__(states, actions, observations, p, o, c, discount, init)
        
    # -- End: __init__


    def p(self, **kwargs):
        """ P : HMM x dict -> {ndarray, float}

            M.P() returns the transition probabilities for M (a ndarray)

            M.P(x = X) returns the transition probabilities P(. | X),
            for state X (a 1D numpy array)

            M.P(x = X, y = Y) returns the transition probabilities
            P(Y | X) (a float)."""

        kwargs['a'] = 0

        super().o(**kwargs)

    # -- End: o


    def o(self, **kwargs):
        """ O : HMM x dict -> {ndarray, float}

            M.O() returns the observation probabilities for M (a ndarray)

            M.O(x = X) returns the observation probabilities P(. | X),
            for state X (a 1D numpy array)

            M.O(z = Z) returns the probabilities P(Z | .),
            for observation Z (a 1D numpy array)

            M.O(x = X, z = Z) returns the observation probabilities
            P(Y | X) (a float)."""

        kwargs['a'] = 0

        super().p(**kwargs)


    # -- End: o


    def sample_transition(self, *args):
        """ sample_transition : HMM x tuple -> tuple

            M.sample_transition() returns an HMM transition. The transition is a
            tuple (X, Y, Z), where X, Y are states and Z is an observation.
            
            M.sample_transition(X) returns a transition from state X."""

        if len(args) == 0:
            x = self._X

        elif len(args) == 1:
            x = self.get_state_index(args[0])

        else:
            raise ValueError('sample_transition -> invalid function call')
            
        trans = super().sample_transition(x, 0)
        return trans[0], trans[3], trans[4]
    
    # -- End: sample_transition        
    
    
    # -- Specialized Functions -- #
    
    def __str__(self):
        """ __str__ returns the "pretty" (external) representation of an HMM."""
        
        s = '('
        s = s + str(set(self._states)) + ', '
        s = s + str(set(self._observations)) + ', P, O)' 
        
        return s
    
    # -- End: __str__
    
    def __repr__(self):
        """ __repr__ returns the functional (internal) representation of an 
            HMM."""
        
        s = 'HMM('
        s = s + str(self._states) + ',\n'
        s = s + str(self._observations) + ',\n'
        
        s = s + repr(self._P) + ',\n'
        s = s + repr(self._O) + ')'

        return s
    
    # -- End: __repr__

#  = = = = =  M A R K O V   C H A I N  = = = = = #   
#
# MDP subclass 
#
#  = = = = = = = = = = = = = = = = = = = = = = = #

class MarkovChain (MDP):
    """ MarkovChain class (inherits from MDP).
    
        MarkovChain : enumeration x ndarray 

        MarkovChain : enumeration x ndarray x ndarray 
                                          
        MarkovChain(X, P) defines a Markov chain with states X and transition 
        probabilities P. In particular:
        
        - X is an enumeration (tuple or list) of states;
        - P is a 3D numpy array, where P[a][x][y] = P(y | x, a).
        
        MarkovChain(n, P) defines a Markov chain where X is now the integers 
        between 0 and n.
        
        MarkovChain(., P, init) additionally initializes the state distribution 
        to init, where init is a 1D numpy array."""
    
    def __init__(self, states, p, init=np.array([])):
        """ MarkovChain : enumeration x ndarray 

            MarkovChain(X, P) defines a Markov chain with states X and transition 
            probabilities P."""
        
        # Check states
        
        aux_s = get_enumeration(states)
        
        if aux_s == -1:
            raise ValueError('hmm -> invalid operand')
        
        # Simulate actions        
       
        actions = (0,)
        
        # Rebuild transition matrix
        
        p = np.array([p])
        
        # Simulate cost & discount
        
        c = np.zeros((len(aux_s), 1))
        discount = 1        

        # Call super
        
        super().__init__(states, actions, p, c, discount, init)
        
    # -- End: __init__
    

    def sample_transition(self, *args):
        """ sample_transition : MarkovChain x tuple -> tuple

            M.sample_transition() returns an Markov chain transition. The 
            transition is a tuple (X, Y), where X and Y are states.
            
            M.sample_transition(X) returns a transition from state X."""

        if len(args) == 0:
            x = self._X

        elif len(args) == 1:
            x = self.get_state_index(args[0])

        else:
            raise ValueError('sample_transition -> invalid function call')
            
        trans = super().sample_transition(x, 0)
        return trans[0], trans[3]

    # -- End: sample_transition
    
    
    # -- Specialized Functions -- #
    
    def __str__(self):
        """ __str__ returns the "pretty" (external) representation of a MC."""
        
        s = '('
        s = s + str(set(self.list_states())) + ', P)'
        
        return s
    
    # -- End: __str__
    
    def __repr__(self):
        """ __repr__ returns the functional (internal) representation of a MC."""
        
        s = '\nMarkovChain('
        s = s + str(self.list_states()) + ',\n'        
        s = s + repr(self._P) + ')'

        return s
    
    # -- End: __repr__
