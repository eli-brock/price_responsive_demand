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
    generator_ratio: np.array,
    voll: np.array
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
    # TODO: allow b as parameter
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

    # CVXPY variables
    p_d_low = cp.Variable((len(L), T), name='p_d_low')
    p_d_high = cp.Variable((len(L), T), name='p_d_high')
    p_g_int = cp.Variable((len(G), T), name='p_g_int')
    p_g_dis = cp.Variable((len(G),T), name='p_g_dis')
    p_b = cp.Variable((N,T), name='p_b')
    b = cp.Variable(N, name='b')
    b_0 = cp.Variable(N, name='b_0')
    delta = cp.Variable((N, T), name='delta')
    z = cp.Variable((len(L),T), boolean=True)

    # The maximum consumption at the value of lost load
    p_voll = (voll/price)**epsilon*demand

    # Numerical infinity
    infinity = 1e2

    # CVXPY constraints
    constraints = [
        AG@(p_g_dis+p_g_int)-AL@(p_d_low+p_d_high)-p_b == -B@delta,
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
        b_0 <= b,
        p_d_low <= p_voll,
        p_d_high >= 0,
        p_d_low >= cp.multiply(p_voll,z),
        p_d_high <= z*infinity
    ] + [
        cp.multiply(B, delta_t[:,np.newaxis]-delta_t[np.newaxis,:]) <= p_line_max
        for delta_t in delta.T
    ]

    objective = cp.Minimize(
            cp.sum(
                cost_lin@p_g_dis+cost_quad@cp.square(p_g_dis)
            ) - cp.sum(
                voll*p_d_low + cp.multiply(
                    price/demand**(1/epsilon)*epsilon/(1+epsilon),
                    cp.bmat(
                        [
                            [p_ij**((1/epsilon+1)[i,j]) for j, p_ij in enumerate(p_i)] 
                            for i, p_i in enumerate(p_voll+p_d_high)
                        ]
                    ) - p_voll**(1/epsilon+1)
                )
            )
        )

    # Co-optimize over the supply- and demand-side cost functions. We use cost convention, so this is a minimization problem.
    cp.Problem(objective, constraints).solve()

    # Re-solve using the optimal value of z as a parameter. This will give an equivalent result, but the SCIP solver used for the
    # mixed-integer problem does not recover dual variables.
    constraints[14:16] = [
        p_d_low >= p_voll*z.value,
        p_d_high <= z.value*infinity
    ]
    opf = cp.Problem(objective, constraints)
    opf.solve()

    return opf