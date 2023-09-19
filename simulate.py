import numpy as np
import cvxpy as cp

def sim(
    L: list,
    G: list,
    B: np.array,
    p_g_int_max: np.array,
    p_g_dis_min: np.array,
    p_g_dis_max: np.array,
    epsilon: np.array,
    price: np.array,
    demand: np.array,
    b_total: float,
    b_duration: float,
    p_line_max: np.array,
    cost_lin: np.array,
    cost_quad: np.array,
    generator_ratio: np.array
) -> cp.Problem:
    '''
    Arguments:
    L -- buses with loads
    G -- buses with generators
    B (N x N) -- susceptance matrix (p.u.)
    p_g_int_max (N_G x T) -- intermittent generation upper bound in fully intermittent case (p.u.)
    p_g_dis_min (N_G x T) -- dispatchable generation lower bound in fully dispatchable gase (p.u.)
    p_g_dis_max (N_G x T) -- dispatchable generation upper bound in fully dispatchable gase (p.u.)
    epsilon (N_L x T) -- price elasticity of demand
    price (N_L x T) -- the price associated with `demand`
    demand (N_L x T) -- the demand associated with `price`
    b_total -- the total storage capacity (MWh)
    b_duration -- time for storage unit to fully charge/discharge (hours)
    p_line_max (N x N) -- thermal line limits (MW)
    cost_lin (N) -- linear coefficients of generator cost
    cost_quad (N) -- quadratic coefficients of generator cost
    generator_ratio (N_G x T) -- proportion of generator capacity coming from intermittent sources

    Returns `opf`, a solved `cp.Problem` instance which contains the following properties of interest:
    opf.variables() in order:
    p_d (N_L x T) -- load (p.u)
    p_g_dis (N_G x T) -- dispatchable generation
    p_g_int (N_G x T) -- intermittent generation
    p_b (N x T) -- storage consumption  (load convention)
    delta (N x T) -- voltage angles (radians)
    b_0 (N) -- initial state-of-charge (MWh)
    b (N) -- storage capacity
    '''

    # The number of nodes and buses, respectively
    N = B.shape[0]
    T = epsilon.shape[1]

    # These matrices to map from the generator/load space to the bus space, acting as reverse indices
    AG = np.zeros((N,len(G)))
    AG[G,:] = np.eye(len(G))
    AL = np.zeros((N,len(L)))
    AL[L,:] = np.eye(len(L))

    # The coefficient of the demand-side objective function
    K = price/demand**(1/epsilon)

    # CVXPY variables
    p_d = cp.Variable((len(L), T), name='p_d')
    p_g_int = cp.Variable((len(G), T), name='p_g_int')
    p_g_dis = cp.Variable((len(G),T), name='p_g_dis')
    p_b = cp.Variable((N,T), name='p_b')
    b = cp.Variable(N, name='b')
    b_0 = cp.Variable(N, name='b_0')
    delta = cp.Variable((N, T), name='delta')

    # CVXPY constraints
    constraints = [
        AG@(p_g_dis+p_g_int)-AL@p_d-p_b == -B@delta,
        p_g_int <= p_g_int_max*generator_ratio,
        p_g_int >= 0,
        p_g_dis <= p_g_dis_max[:,np.newaxis]*(1-generator_ratio),
        p_g_dis >= p_g_dis_min[:,np.newaxis]*(1-generator_ratio),
        cp.cumsum(p_b, axis=1) >= -b_0[:,np.newaxis],
        cp.cumsum(p_b, axis=1) <= (b-b_0)[:,np.newaxis],
        p_b <= b[:,np.newaxis]/b_duration,
        p_b >= -b[:,np.newaxis]/b_duration,
        cp.sum(b) == b_total,
        b_0 >= 0,
        b_0 <= b
    ] + [
        cp.multiply(B, delta_t[:,np.newaxis]-delta_t[np.newaxis,:]) <= p_line_max
        for delta_t in delta.T
    ]
    
    # Co-optimize over the supply- and demand-side cost functions. We use cost convention, so this is a minimization problem.
    opf = cp.Problem(
        cp.Minimize(
            cp.sum(
                cost_lin@p_g_dis+cost_quad@cp.square(p_g_dis)
            ) - cp.sum(
                cp.multiply(K*epsilon/(1+epsilon), cp.exp(cp.multiply((1/epsilon+1),cp.log(p_d))))
            )
        ),
        constraints
    )

    opf.solve(max_iters=10000)

    return opf