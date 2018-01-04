import numpy as np
from scipy.stats import poisson
import models

from models import is_valid_item, get_index, is_stochastic_matrix, is_equal

def two_states():

    # Build states

    states = ('A', 'B')

    # Build actions

    actions = ('a', 'b')

    # Build transitions

    pa = np.array([[1, 0],
                   [1, 0]])

    pb = np.array([[0, 1],
                   [0, 1]])

    p = np.array([pa, pb])

    # Build cost

    c = np.eye(2)

    # Discount

    g = 0.5

    # Model

    return models.MDP(states, actions, p, c, g)



def three_states():
    
    # Build states
    
    states = ('0','A','B')    
    
    # Build actions
    
    actions = ('a', 'b')
    
    # Build transitions
    
    pa = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

    pb = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [0, 0, 1]])
    
    p = np.array([pa, pb])
    
    # Build cost
    
    c = np.array([[1.0, 0.5],
                  [0.0, 0.0],
                  [1.0, 1.0]])
    
    # Discount
    
    g = 0.99
    
    # Model
    
    return models.MDP(states, actions, p, c, g)

# -- End: three_states

def seven_states():

    # Build states

    states = ('I', 'A1', 'A2', 'B', 'C', 'D', 'E')

    # Build actions

    actions = ('a', 'b', 'c')

    # Build observations

    observations = ('I', 'A', 'B', 'C', 'D', 'E')

    # Build transitions

    pa = np.array([[0, 0.5, 0.5, 0, 0, 0, 0],
                   [0, 0,   0,   0, 0, 1, 0],
                   [0, 0,   0,   0, 0, 0, 1],
                   [0, 1,   0,   0, 0, 0, 0],
                   [0, 0,   1,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0]])

    pb = np.array([[0, 0.5, 0.5, 0, 0, 0, 0],
                   [0, 0,   0,   0, 0, 0, 1],
                   [0, 0,   0,   0, 0, 1, 0],
                   [0, 1,   0,   0, 0, 0, 0],
                   [0, 0,   1,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0]])

    pc = np.array([[0, 0.5, 0.5, 0, 0, 0, 0],
                   [0, 0,   0,   1, 0, 0, 0],
                   [0, 0,   0,   0, 1, 0, 0],
                   [0, 1,   0,   0, 0, 0, 0],
                   [0, 0,   1,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0],
                   [1, 0,   0,   0, 0, 0, 0]])

    p = np.array([pa, pb, pc])

    # Build observations

    oa = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])

    p = np.array([pa, pb, pc])
    o = np.array([oa, oa, oa])

    # Build cost

    c = 2 * np.array([[0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5],
                  [0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.5]])

    # Discount

    g = 0.95

    # Model

    return models.POMDP(states, actions, observations, p, o, c, g)

# -- End: seven_states

def hiring(N):
    
    if N < 2:
        raise ValueError('hiring -> N must be at least 2')
    
    # Build states
    
    states = ('(B,1)',)
    
    for n in range(1, N):
        curr = '(B,' + str(n + 1) + ')'
        states += curr,

        curr = '(~B,' + str(n + 1) + ')'
        states += curr,

    states += 'H',
        
    b1 = 0
    h = 2 * N - 1
        
    # Build actions
        
    actions = ('H', '~H')
        
    # Build transitions and cost
    
    ph = np.zeros((2 * N, 2 * N))
    pnh = np.zeros((2 * N, 2 * N))
    
    ch = np.zeros(2 * N)
    cnh = np.zeros(2 * N)

    ph[b1][h] = 1
    ph[h][h] = 1
    
    pnh[h][h] = 1
    pnh[b1][1] = 0.5
    pnh[b1][2] = 0.5
    
    ch[b1] = (N - 1) / N
    ch[h] = 0
    
    cnh[b1] = 0
    cnh[h] = 0
    
    for s in range(1, 2 * N - 1):
        ph[s][h] = 1
        cnh[s] = 0
        
        n = (s - 1) // 2 + 2
        b = (s % 2) != 0
                
        if n == N:
            pnh[s][h] = 1
            if b:
                cnh[s] = 0
                ch[s] = 0
            else:
                cnh[s] = 1
                ch[s] = 1
        else:        
            # Next best candidate 
            s_bnext = 2 * n - 1
            
            # Next worst candidate
            s_nbnext = 2 * n
            
            # Transition probabilities
            pnh[s][s_bnext] = 1/(n + 1)
            pnh[s][s_nbnext] = n/(n + 1)
            
            # Cost
            if not b:
                ch[s] = 1
            else:
                ch[s] = (N - n)/N
    
    p = np.array([ph, pnh])
    c = np.array([ch, cnh]).transpose()
    
    g = 0.95
    
    return models.MDP(states, actions, p, c, g)
            
# -- End: hiring


def epidemic(N):
    
    # Problem constants
    
    l0 = 1/N
    u0 = 1/N
    k = l0 / (l0 * (N - 1) + u0)
    
    cq = 0.05
    ci = 0.1
    ct = 0.05
    
    # Build states

    states = tuple(range(N + 1))
    
    # Build actions
    
    actions = ((0,0),   (0,10),   (0,50),  (0,100),
               (10,0),  (10,10),  (10,50),  (10,100),
               (50,0),  (50,10),  (50,50),  (50,100),
               (100,0), (100,10), (100,50), (100,100))
    
    # Build transitions and cost
    
    p = []
    c = []

    for a in actions:
        paux = np.zeros((N + 1, N + 1))
        caux = np.zeros(N + 1)

        for x in states:
            aq = a[0] / 100
            at = a[1] / 100
            
            li = (N - x) * l0 * np.exp(-3 * aq)
            lc = (N - x) * x * l0 * np.exp(-3 * at)
            u = x * u0 * at * (2 - at)

            if x < N:
                dp = k * (li + lc)
                paux[x][x + 1] = dp
            else:
                dp = 0
                
            if x > 0:
                dm = k * u
                paux[x][x - 1] = dm
            else:
                dm = 0
                
            paux[x][x] = 1 - dp - dm
            
            caux[x] = min(cq * aq ** 2 + ct * at * (x / N) + ci * (x / N) ** 3, 1)

        p += [paux]
        c += [caux]
            
    p = np.array(p)
    c = np.array(c).transpose()
    
    g = 0.95
    
    return models.MDP(states, actions, p, c, g)

# -- End: Epidemic

def fault(N, pf, ti):
    
    # Check probability vector
    
    if len(pf) != N or \
       any(map(lambda u: u > 1, pf)) or \
       any(map(lambda u: u < 0, pf)):
        raise ValueError('fault -> invalid probability vector')
    
    # Check cost vector
    
    if len(ti) != N or \
       any(map(lambda u: u > 1, ti)) or \
       any(map(lambda u: u < 0, ti)):
        raise ValueError('fault -> invalid cost vector')

    # Build states
    
    uf = list(range(1, 2 ** N))
    
    for i in range(len(uf)):
        uf[i] = np.binary_repr(uf[i], N).replace('1', '?').replace('0', 'O')
        
    states = tuple(uf)
    
    uf = list(range(2 ** (N - 1)))
    
    for i in range(len(uf)):
        uf[i] = np.binary_repr(uf[i], N - 1).replace('1', '?').replace('0', 'O')
        
    for i in range(N):
        for s in uf:
            aux = s[:i] + 'X' + s[i:]

            states += aux,
    
    # Build actions
    
    actions = tuple(range(1, N + 1))
    
    # Build transitions and costs
    
    p = []
    c = []
    
    for act in actions:
        a = act - 1
        
        paux = np.zeros((len(states), len(states)))
        caux = np.zeros(len(states))
        
        for i in range(len(states)):
            for j in range(len(states)):
                x = states[i]
                y = states[j]
                
                if x[:a] != y[:a] or x[a + 1:] != y[a + 1:]:
                    aux = 0
                else:
                    aux = 1

                if 'X' not in x:
                    if x[a] == '?' and y[a] == 'X':
                        index = [idx for idx, ltr in enumerate(x) if ltr == '?']
                        ptot = sum([pf[idx] for idx in index])
                        
                        aux = aux * pf[a] / ptot
                    elif x[a] == '?' and y[a] == 'O':
                        index = [idx for idx, ltr in enumerate(x) if ltr == '?']
                        ptot = sum([pf[idx] for idx in index])

                        aux *= 1 - pf[a] / ptot
                    elif x[a] == '?':
                        aux *= 0
                    elif y[a] != x[a]:
                        aux *= 0
                else:
                    if x[a] == '?' and y[a] == 'X':
                        aux *= 0
                    elif x[a] == '?' and y[a] == 'O':
                        aux *= 1
                    elif x[a] == '?':
                        aux *= 0
                    elif y[a] != x[a]:
                        aux *= 0

                paux[i][j] = aux

                if 'X' in x:
                    caux[i] = 0
                else:
                    caux[i] = min(0.5 + ti[a], 1)
                    
            aux = np.sum(paux[i])
            paux[i] /= aux

        p += [paux]
        c += [caux]
                
    p = np.array(p)
    c = np.array(c).transpose()
    
    g = 0.95
    
    return models.MDP(states, actions, p, c, g)

# -- End: fault


def bus(N):
    
    # Constants
    
    la = .3  # Type : float
    lb = .7  # Type : float

    d = 2.   # Type : float
    
    kw = (la + lb) / (2 * N)  # Type : float
    kr = 0.05                 # Type : float
    
    # Build states
    
    states = ()
    
    for i in range(N + 1):
        for j in range(N + 1):
            states += (i, j),
            
    # Build actions
    
    actions = ('W', 'S')
    
    # Build transitions and costs
    
    pw = np.zeros((len(states),len(states)))
    ps = np.zeros((len(states),len(states)))
    
    cw = np.zeros(len(states))
    cs = np.zeros(len(states))
    
    for i in range(len(states)):
        m = states[i][0]
        n = states[i][1]

        for j in range(len(states)):
            k = states[j][0]
            l = states[j][1]
            
            if k == m and n < N and l == n + 1:
                pw[i][j] = la / (la + lb)
            elif m < N and k == m + 1 and l == n:
                pw[i][j] = lb / (la + lb)
            elif m == N and n == N and k == m and l == n:
                pw[i][j] = 1
            else:
                pw[i][j] = 0
                
            ps[i][j] = poisson.pmf(k, la * d) * poisson.pmf(l, lb * d / 2)
            
        aux = np.sum(pw[i])
        pw[i] /= aux

        aux = np.sum(ps[i])
        ps[i] /= aux

        # noinspection PyTypeChecker
        cw[i] = kw * (m + n) / (la + lb)
        cs[i] = kr + 0.5 * kw * d ** 2 * (n / d + la + lb/2)  # Type : float
        
    p = np.array([pw, ps])
    c = np.array([cw, cs]).transpose()
    
    g = 0.95
    
    return models.MDP(states, actions, p, c, g)

# -- End: bus


def divergent():
    
    states = (1, 2, 3, 4, 5, 6)
    actions = ('a', 'b')
    
    pa = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0.05, 0.95, 0, 0, 0, 0],
                   [0.05, 0.95, 0, 0, 0, 0],
                   [0.05, 0.95, 0, 0, 0, 0],
                   [0.05, 0.95, 0, 0, 0, 0]])
    
    pb = pa
    
    p = np.array([pa, pb])
    
    # Build cost
    
    c = np.array([[0.0, 1.0],
                  [0.0, 1.0],
                  [0.0, 1.0],
                  [0.0, 1.0],
                  [0.0, 1.0],
                  [0.0, 1.0]])
    
    # Discount
    
    g = 0.99
    
    # Model
    
    return models.MDP(states, actions, p, c, g)

# -- End: divergent


def household_robot():

    # States
    #          0    1    2     3     4    5    6    7    8
    states = ('B', 'D', 'H', 'H1', 'H2', 'K', 'L', 'P', 'W')

    # Actions
    #           0    1    2    3    4
    actions = ('U', 'D', 'L', 'R', 'S')

    # Observations

    observations = ('0000', '0001', '0010', '0011',
                    '0100', '0101', '0110', '0111',
                    '1000', '1001', '1010', '1011',
                    '1100', '1101', '1110', '1111')

    # Transition probabilities

    ps = np.eye(len(states))
    pu = np.eye(len(states))
    pd = np.eye(len(states))
    pl = np.eye(len(states))
    pr = np.eye(len(states))

    # Up:

    pu[3, 3] = 0
    pu[3, 0] = 1

    pu[2, 2] = 0
    pu[2, 6] = 1

    pu[6, 6] = 0
    pu[6, 0] = 1

    pu[4, 4] = 0
    pu[4, 6] = 1

    pu[1, 1] = .4
    pu[1, 3] = .6

    pu[7, 7] = 0
    pu[7, 3] = 1

    # Down:

    pd[6, 6] = 0
    pd[6, 4] = 1

    pd[0, 0] = 0
    pd[0, 3] = 1

    # Left

    pl[5, 5] = 0
    pl[5, 7] = 1

    pl[1, 1] = 0
    pl[1, 5] = 1

    pl[3, 3] = 0
    pl[3, 7] = 1

    pl[6, 6] = .4
    pl[6, 3] = .6

    pl[2, 2] = .4
    pl[2, 4] = .6

    pr[4, 4] = 0
    pr[4, 8] = 1

    # Right

    pr[0, 0] = 0
    pr[0, 6] = 1

    pr[5, 5] = 0
    pr[5, 1] = 1

    pr[6, 6] = 0
    pr[6, 2] = 1

    pr[7, 7] = 0
    pr[7, 5] = 1

    pr[8, 8] = 0
    pr[8, 4] = 1

    # Observation probabilities

    # Observations:    0     1
    oto = np.array([[0.05, 0.95],   # B
                    [0.90, 0.10],   # D
                    [0.05, 0.95],   # H
                    [0.80, 0.20],   # H1
                    [0.90, 0.10],   # H2
                    [0.05, 0.95],   # K
                    [0.05, 0.95],   # L
                    [0.05, 0.95],   # P
                    [0.05, 0.95]])  # W

    obo = np.array([[0.90, 0.10],   # B
                    [0.05, 0.95],   # D
                    [0.05, 0.95],   # H
                    [0.80, 0.20],   # H1
                    [0.05, 0.95],   # H2
                    [0.05, 0.95],   # K
                    [0.80, 0.20],   # L
                    [0.05, 0.95],   # P
                    [0.05, 0.95]])  # W

    ole = np.array([[0.05, 0.95],   # B
                    [0.90, 0.10],   # D
                    [0.90, 0.10],   # H
                    [0.90, 0.10],   # H1
                    [0.90, 0.10],   # H2
                    [0.80, 0.20],   # K
                    [0.90, 0.10],   # L
                    [0.05, 0.95],   # P
                    [0.05, 0.95]])  # P

    ori = np.array([[0.90, 0.10],   # B
                    [0.05, 0.95],   # D
                    [0.05, 0.95],   # H
                    [0.90, 0.10],   # H1
                    [0.90, 0.10],   # H2
                    [0.90, 0.10],   # K
                    [0.90, 0.10],   # L
                    [0.90, 0.10],   # P
                    [0.90, 0.10]])  # W

    o = np.zeros((len(states), len(observations)))

    for i, obs in enumerate(observations):
        to = int(obs[0])
        ri = int(obs[1])
        bo = int(obs[2])
        le = int(obs[3])

        o[:, i] = oto[:, to] * obo[:, bo] * ole[:, le] * ori[:, ri]

    o = np.array([o] * len(actions))

    # Costs

    c = np.ones((len(states), len(actions)))
    c[5, :] = 0

    p = np.array([pu, pd, pl, pr, ps])

    g = 0.95

    return models.POMDP(states, actions, observations, p, o, c, g)

# -- End: household_robot


def parse(fname):
    """ parse : string -> POMDP

        parse(s) parses the file named s into a POMDP. The file is expected to adhere to Tony Cassandra's POMDP file
        format, described in www.pomdp.org."""

    def parse_index(element, domain):
        """ parse_index : str x list -> int

            parse_index(e, l) parses whether the string e corresponds to an element of the list l. If so, it returns
            the corresponding index. If e already corresponds to an index, it returns it. Otherwise, it raises an
            exception."""

        try:
            x = int(element)
        except ValueError:
            x = element

        if is_valid_item(x, domain):
            return get_index(x, domain)
        else:
            raise ValueError('parse: invalid element')

    # -- End: parse_index


    def process_preamble(tok):
        """ process_preamble : list -> dict

            process_preamble(l) goes over the list l searching for the POMDP prb information and fills in a dictionary
            structure with that info. Preamble lines are removed from l (destructively). The returned dictionary
            includes the fields:

            . states       -> a list of entities corresponding to the POMDP states.
            . actions      -> a list of entities corresponding to the POMDP actions.
            . observations -> a list of entities corresponding to the POMDP observations.
            . discount     -> the value of the discount for the POMDP.
            . cost         -> +1 or -1, depending on whether the POMDP is defined as rewards or costs.

            If any of the fields is not provided, the function raises an exception. """

        fields = ['states', 'actions', 'observations', 'discount', 'cost']
        prb = dict()

        i = 0
        while i < len(tok):
            line = tok[i]

            if line[0] == 'states:':
                if len(line) == 1:
                    line = line + tok[i + 1]
                    del (tok[i + 1])

                if len(line) > 2:
                    prb['states'] = line[1:]
                elif line[1].isdigit():
                    prb['states'] = list(range(int(line[1])))
                else:
                    raise ValueError('process_preamble -> invalid specification')
                del(tok[i])

            elif line[0] == 'actions:':
                if len(line) > 2 or len(line) == 1:
                    if len(line) == 1:
                        line = line + tok[i+1]
                        del(tok[i+1])
                    prb['actions'] = line[1:]
                elif line[1].isdigit():
                    prb['actions'] = list(range(int(line[1])))
                else:
                    raise ValueError('process_preamble -> invalid specification')
                del (tok[i])

            elif line[0] == 'observations:':
                if len(line) > 2 or len(line) == 1:
                    if len(line) == 1:
                        line = line + tok[i+1]
                        del(tok[i+1])
                    prb['observations'] = line[1:]
                elif line[1].isdigit():
                    prb['observations'] = list(range(int(line[1])))
                else:
                    raise ValueError('process_preamble -> invalid specification')
                del (tok[i])

            elif line[0] == 'discount:':
                if len(line) > 2:
                    raise ValueError('process_preamble -> invalid specification')

                prb['discount'] = float(line[1])
                del (tok[i])

            elif line[0] == 'values:':
                if len(line) > 2:
                    raise ValueError('process_preamble -> invalid specification')

                if line[1] == 'reward':
                    prb['cost'] = -1
                elif line[1] == 'cost':
                    prb['cost'] = 1
                else:
                    raise ValueError('process_preamble -> invalid specification')
                del (tok[i])

            else:
                i += 1

        for fld in fields:
            if fld not in prb:
                raise ValueError('process_preamble -> incomplete specification (missing %s)' % f)

        print('Preamble: processed successfully.')

        return prb

    # -- End: process_preamble


    def process_start(tok, states):
        """ process_start : list x tuple -> np.array

            process_start(l, t) goes over the list l and processes the information regarding the POMDP's initial
            distribution over the states in t. It returns a 1D numpy array with that distribution. If no distribution is
            specified, the uniform distribution is returned."""

        # Initial state or distribution over X

        sdist = None
        i = 0

        while i < len(tok):
            line = tok[i]

            if line[0] == 'start:':

                # Distribution

                if len(line) == 1:
                    line = line + tok[i + 1]
                    del (tok[i + 1])

                if len(line) > 2:
                    if sdist is None:
                        sdist = []

                    for p in line[1:]:
                        try:
                            sdist += [float(p)]
                        except:
                            raise ValueError('process_start -> invalid specification')

                    if len(sdist) != len(states) or sum(sdist) != 1:
                        raise ValueError('process_start -> invalid specification')
                    else:
                        sdist = np.array(sdist)

                # Uniform

                elif line[1] == 'uniform':
                    sdist = np.ones(len(states)) / len(states)

                # Single state

                else:
                    # Figure out state

                    try:
                        sdist = parse_index(line[1], states)
                    except ValueError:
                        raise ValueError('process_start -> invalid specification')

                del(tok[i])

            # Distribution over subset of X

            elif line[0] == 'start':
                if len(line) < 3:
                    raise ValueError('process_start -> invalid specification')

                if tok[1] == 'include:':
                    sdist = np.zeros(len(states))

                elif tok[1] == 'exclude:':
                    sdist = np.ones(len(states))

                else:
                    raise ValueError('process_start -> invalid specification')

                for x in line[2:]:
                    try:
                        x = int(x)
                    except ValueError:
                        pass

                    if not is_valid_item(x, states):
                        raise ValueError('process_start -> invalid specification')
                    else:
                        sdist[get_index(x, states)] = 1 - sdist[get_index(x, states)]

                del(tok[i])

            else:
                i += 1

        if sdist is None:
            sdist = np.ones(len(states)) / len(states)

        if not is_equal(sum(sdist), 1.0):
            raise ValueError('process_start -> invalid specification')

        print('Start: processed successfully.')

        return np.array(sdist)

    # -- End: process_start


    def process_stochastic_matrix(tok, label, rows, columns, actions):
        """ process_stochastic_matrix : list x str x list x list x list -> list

            process_stochastic_matrix(l, s, x, y, a) goes over the list l searching for the stochastic matrix with
            label s. The matrix is indexed by the elements of x in its rows and the elements of y in its columns. The
            elements in a correspond to the actions according to which the matrix will be built. Probability lines are
            removed from l (destructively). Each element of the returned list corresponds to a stochastic matrix for
            the corresponding action (the matrices are ordered according to the actions in a).

            If any of the single-action probabilities is not provided, the function raises an exception. """

        m = [None] * len(actions)

        i = 0
        while i < len(tok):
            line = tok[i]

            if line[0] == label:
                if len(line) == 7:
                    if line[2] != ':' or line[4] != ':':
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    if line[3] == '*':
                        row = rows
                    else:
                        try:
                            row = [parse_index(line[3], rows)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    if line[5] == '*':
                        column = columns
                    else:
                        try:
                            column = [parse_index(line[5], columns)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    for a in act:
                        a_idx = get_index(a, actions)
                        if m[a_idx] is None:
                            m[a_idx] = np.zeros((len(rows),len(columns)))

                        for x in row:
                            x_idx = get_index(x, rows)

                            for y in column:
                                y_idx = get_index(y, columns)
                                m[a_idx][x_idx][y_idx] = np.double(line[6])

                    del(tok[i])

                elif len(line) == 4:
                    if line[2] != ':':
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    if line[3] == '*':
                        row = rows
                    else:
                        try:
                            row = [parse_index(line[3], rows)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    try:
                        values = tok[i + 1]
                        del (tok[i + 1])
                    except IndexError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                    if values == ['uniform']:
                        values = [1/len(columns)] * len(columns)

                    for a in act:
                        a_idx = get_index(a, actions)
                        if m[a_idx] is None:
                            m[a_idx] = np.zeros((len(rows), len(columns)))

                        for x in row:
                            x_idx = get_index(x, rows)

                            for y_idx in range(len(columns)):
                                try:
                                    m[a_idx][x_idx][y_idx] = np.double(values[y_idx])
                                except IndexError:
                                    raise ValueError('process_stochastic_matrix -> invalid specification')

                    del(tok[i])

                elif len(line) == 2:

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_stochastic_matrix -> invalid specification')

                    try:
                        if tok[i + 1] == ['uniform']:
                            values = np.ones((len(rows), len(columns))) / len(columns)
                            del(tok[i + 1])
                        elif tok[i + 1] == ['identity']:
                            values = np.eye(len(rows),len(columns))
                            del(tok[i + 1])
                        else:
                            values = tok[i + 1:i + len(rows) + 1]
                            del(tok[i + 1:i + len(rows) + 1])

                    except IndexError:
                        raise ValueError('process_stochastic_matrix -> invalid specification')

                    for a in act:
                        a_idx = get_index(a, actions)
                        if m[a_idx] is None:
                            m[a_idx] = np.zeros((len(rows),len(columns)))

                        for x_idx in range(len(rows)):
                            for y_idx in range(len(columns)):
                                try:
                                    m[a_idx][x_idx][y_idx] = np.double(values[x_idx][y_idx])
                                except IndexError:
                                    raise ValueError('process_stochastic_matrix -> invalid specification')

                    del(tok[i])
                else:
                    raise ValueError('process_stochastic_matrix -> invalid specification')
            else:
                i += 1

        for a in range(len(actions)):
            if not is_stochastic_matrix(len(rows), len(columns), m[a]):
                print('action:', a)
                raise ValueError('process_stochastic_matrix -> invalid specification (incomplete or incorrect P)')

        print('Stochastic matrix %s processed successfully.' % label)

        return np.array(m)

    # -- End: process_transition


    def process_reward(tok, states, actions, observations, p, o, inv = 1):
        """ process_reward : list x list x list x list x list x list x int -> np.array

            process_reward(l, s, a, z, p, o, inv) goes over the list l searching for a cost matrix. The matrix is 4D,
            depending on actions (a), states (s), next states and observations (z). The function averages out both
            next states and observations to provide as a result a cost matrix that is a function of s and a. For the
            next state and observation to be averaged out, the function requires the transition and observation
            probabilities (p and o). Finally, the cost is also normalized, for which reason it is also required the bit
            inv (+1 or -1) to turn rewards into costs in the [0,1] interval.

            If the reward is mis-specified in the list tok, the function raises an exception. """

        cst = np.zeros((len(states), len(actions)))

        i = 0
        while i < len(tok):
            line = tok[i]

            if line[0] == 'R:':
                if len(line) >= 9:
                    if line[2] != ':' or line[4] != ':' or line[6] != ':':
                        raise ValueError('process_reward -> invalid specification')

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[3] == '*':
                        state_init = states
                    else:
                        try:
                            state_init = [parse_index(line[3], states)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[5] == '*':
                        state_end = states
                    else:
                        try:
                            state_end = [parse_index(line[5], states)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[7] == '*':
                        obs = observations
                    else:
                        try:
                            obs = [parse_index(line[7], observations)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    for a in act:
                        a_idx = get_index(a, actions)
                        for x in state_init:
                            x_idx = get_index(x, states)
                            c_aux = cst[x_idx][a_idx]

                            for y in state_end:
                                y_idx = get_index(y, states)

                                for z in obs:
                                    z_idx = get_index(z, observations)
                                    c_aux += float(line[8]) * p[a_idx][x_idx][y_idx] * o[a_idx][y_idx][z_idx]

                            cst[x_idx][a_idx] = c_aux

                    del (tok[i])

                elif len(line) == 6:
                    if line[2] != ':' or line[4] != ':':
                        raise ValueError('process_reward -> invalid specification')

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[3] == '*':
                        state_init = states
                    else:
                        try:
                            state_init = [parse_index(line[3], states)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[5] == '*':
                        state_end = states
                    else:
                        try:
                            state_end = [parse_index(line[5], states)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    try:
                        values = tok[i + 1]
                        del (tok[i + 1])
                    except IndexError:
                        raise ValueError('process_reward -> invalid specification')

                    for a in act:
                        a_idx = get_index(a, actions)
                        for x in state_init:
                            x_idx = get_index(x, states)
                            c_aux = cst[x_idx][a_idx]

                            for y in state_end:
                                y_idx = get_index(y, states)

                                for z_idx in range(len(observations)):
                                    c_aux += float(values[z_idx]) * p[a_idx][x_idx][y_idx] * o[a_idx][y_idx][z_idx]

                            cst[x_idx][a_idx] = c_aux

                    del (tok[i])

                elif len(line) == 4:
                    if line[2] != ':':
                        raise ValueError('process_reward -> invalid specification')

                    # Identify action

                    if line[1] == '*':
                        act = actions
                    else:
                        try:
                            act = [parse_index(line[1], actions)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    if line[3] == '*':
                        state_init = states
                    else:
                        try:
                            state_init = [parse_index(line[3], states)]
                        except ValueError:
                            raise ValueError('process_reward -> invalid specification')

                    try:
                        values = tok[i + 1:i + len(states) + 1]
                        del (tok[i + 1:i + len(states) + 1])
                    except IndexError:
                        raise ValueError('process_reward -> invalid specification')

                    for a in act:
                        a_idx = get_index(a, actions)
                        for x in state_init:
                            x_idx = get_index(x, states)
                            c_aux = 0

                            for y_idx in range(len(states)):
                                for z_idx in range(len(observations)):
                                    c_aux += float(values[y_idx][z_idx]) * p[a_idx][x_idx][y_idx] * \
                                             o[a_idx][y_idx][z_idx]

                            cst[x_idx][a_idx] = c_aux

                    del (tok[i])

            else:
                i += 1

        cst *= inv

        for a in range(len(actions)):
            m = cst.max().max()
            n = cst.min().min()

            cst = (cst - n) / (m - n)

        print('Cost matrix: processed successfully.')

        return cst

    # -- End: process_reward


    f = open(fname, 'r')
    t = f.readlines()
    f.close()

    # Clean lines and delete comment lines

    tokens = []
    for lin in t:

        # Remove comments

        if lin[0] == '#':
            continue

        # Clean white space characters and end-of-line comments

        aux = lin.split()
        if aux:
            for j in range(len(aux)):
                if aux[j][0] == '#':
                    aux = aux[:j]
                    break
            tokens += [aux]

    # Process file

    preamble = process_preamble(tokens)
    start = process_start(tokens, preamble['states'])
    pr_matrix = process_stochastic_matrix(tokens, 'T:',
                                          preamble['states'],
                                          preamble['states'],
                                          preamble['actions'])

    ob_matrix = process_stochastic_matrix(tokens, 'O:',
                                          preamble['states'],
                                          preamble['observations'],
                                          preamble['actions'])

    c = process_reward(tokens,
                       preamble['states'],
                       preamble['actions'],
                       preamble['observations'],
                       pr_matrix,
                       ob_matrix,
                       inv=preamble['cost'])

    print('File', fname, 'processed successfully.')

    return models.POMDP(preamble['states'],
                        preamble['actions'],
                        preamble['observations'],
                        pr_matrix,
                        ob_matrix,
                        c,
                        preamble['discount'],
                        init = start)

# -- End: parse