import numpy as np
import cvxpy as cp

class Network:
    '''A power network for a specified timespan'''
    def __init__(
            self,
            loads,
            generators,
            B,
            p_g_int_max,
            p_g_dis_min,
            p_g_dis_max,
            epsilon,
            price,
            demand,
            b_total,
            b_duration,
            p_line_max,
            cost_lin,
            cost_quad,
            int_gen_ratio,
            voll
        ):
        self.loads = loads
        self.generators = generators
        self.B = B
        self.p_g_int_max = p_g_int_max
        self.p_g_dis_min = p_g_dis_min
        self.p_g_dis_max = p_g_dis_max
        self.epsilon = epsilon
        self.price = price
        self.demand = demand
        self.b_total = b_total
        self.b_duration = b_duration
        self.p_line_max = p_line_max
        self.cost_lin = cost_lin
        self.cost_quad = cost_quad
        self.int_gen_ratio = int_gen_ratio
        self.voll = voll

        self.n = int(np.unique(B.shape).squeeze())
        self.t = int(epsilon.shape[1])
        self.n_g = self.generators.size
        self.n_l = self.loads.size

        self.delta = self.demand*self.voll**self.epsilon/(self.price**self.epsilon-self.voll**self.epsilon)

        self.opf = None

    def solve(self):
        '''Optimize the dipatch of generators, loads, and storage.'''
        TG = np.zeros((self.n,self.n_g))
        TG[self.generators,:] = np.eye(self.n_g)
        TL = np.zeros((self.n,self.n_l))
        TL[self.loads,:] = np.eye(self.n_l)

        # CVXPY variables
        p_d = cp.Variable((self.n_l, self.t), name='p_d')
        p_g_int = cp.Variable((self.n_g, self.t), name='p_g_int')
        p_g_dis = cp.Variable((self.n_g,self.t), name='p_g_dis')
        p_b = cp.Variable((self.n,self.t), name='p_b')
        b = cp.Variable(self.n, name='b')
        b_0 = cp.Variable(self.n, name='b_0')
        angle = cp.Variable((self.n, self.t), name='angle')

        constraints = [
            TG@(p_g_dis+p_g_int)-TL@(p_d)-p_b == -self.B@angle,
            p_g_int <= self.p_g_int_max*self.int_gen_ratio,
            p_g_int >= 0,
            p_g_dis <= self.p_g_dis_max[:,np.newaxis]*(1-self.int_gen_ratio),
            p_g_dis >= self.p_g_dis_min[:,np.newaxis]*(1-self.int_gen_ratio),
            cp.cumsum(p_b, axis=1) >= -b_0[:,np.newaxis],
            cp.cumsum(p_b, axis=1) <= (b-b_0)[:,np.newaxis],
            p_b <= b[:,np.newaxis]/self.b_duration,
            p_b >= -b[:,np.newaxis]/self.b_duration,
            cp.sum(b) == self.b_total,
            b_0 >= 0,
            b_0 <= b,
        ] + [
            cp.multiply(self.B, angle_t[:,np.newaxis]-angle_t[np.newaxis,:]) <= self.p_line_max
            for angle_t in angle.T
        ]

        self.opf = cp.Problem(
            cp.Minimize(
                cp.sum(
                    self.cost_lin@p_g_dis+self.cost_quad@cp.square(p_g_dis)
                ) - cp.sum(
                    cp.multiply(self.epsilon*self.price,cp.exp(cp.multiply(1/self.epsilon+1,cp.log(p_d+self.delta)))-self.delta**(1/self.epsilon+1))
                    /(self.epsilon+1)/(self.demand+self.delta)**(1/self.epsilon)
                )
            ),
            constraints
        )

        self.opf.solve()
        