import numpy as np
import cvxpy as cp

class Network:
    '''A power network for a specified timespan'''
    def __init__(
            self,
            B,
            line_lims,
            profile,
            dis_max,
            total_storage,
            storage_cycle_hours,
            cost_coeffs
        ):
        self.B = B
        self.line_lims = line_lims
        self.profile = profile
        self.dis_max = dis_max
        self.total_storage = total_storage
        self.storage_cycle_hours = storage_cycle_hours
        self.cost_coeffs = cost_coeffs

    def solve(self):
        '''Optimize the dipatch of generators, loads, and storage.'''

        # Define helpful variables
        generators = self.profile.nom_load.columns
        loads = self.profile.int_gen.columns
        n = self.B.shape[0]
        n_g = len(generators)
        n_l = len(loads)
        t = len(self.profile.index)
        TG = np.zeros((n,n_g))
        TG[generators-1,:] = np.eye(n_g)
        TL = np.zeros((n,n_l))
        TL[loads-1,:] = np.eye(n_l)

        # CVXPY variables
        load = cp.Variable((n_l, self.t), name='load')
        generation = cp.Variable((self.n_g, self.t), name='generation')
        storage_load = cp.Variable((self.n,self.t), name='storage_load')
        storage_capacity = cp.Variable(self.n, name='storage_capacity')
        initial_soc = cp.Variable(self.n, name='initial_soc')
        angle = cp.Variable((self.n, self.t), name='angle')

        constraints = [
            TG@generation-TL@load-storage_load == -self.B@angle,
            generation <= self.dis_max + self.profile.int_gen,
            generation >= 0,
            cp.cumsum(storage_load, axis=1) >= -storage_capacity[:,np.newaxis],
            cp.cumsum(storage_load, axis=1) <= (storage_capacity-initial_soc)[:,np.newaxis],
            storage_load <= storage[:,np.newaxis]/self.storage_cycle_timesteps,
            storage_load >= -storage[:,np.newaxis]/self.storage_cycle_timesteps,
            cp.sum(storage_capacity) == self.total_storage,
            initial_soc >= 0,
            initial_soc <= storage_capacity,
        ] + [
            cp.multiply(self.B, angle_t[:,np.newaxis]-angle_t[np.newaxis,:]) <= self.line_lims
            for angle_t in angle.T
        ]

        dis_cost = lambda p: p@cp.diag(self.cost[0,:]) + cp.square(p)@cp.diag(self.cost[1,:])
        cost = lambda p: cp.maximum(0,dis_cost(p-self.profile.int_gen))
        utility = (
            lambda p: self.profile.nom_price*self.profile.elasticity
            *cp.exp(cp.multiply(1 + 1/self.profile.elasticity, cp.log(p+delta)))
            /(self.profile.nom_load + delta)**(1/self.profile.elasticity)
            /(self.profile.elasticity + 1)
        )

        self.opf = cp.Problem(
            cp.Minimize(
                cp.sum(cost(generation)-utility(load))
            ),
            constraints
        )

        self.opf.solve()
        