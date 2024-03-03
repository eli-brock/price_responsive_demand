import numpy as np
import cvxpy as cp
import pandas as pd

def outer_difference(x):
    '''Helper function used to compute line flows'''
    return x[:,np.newaxis]-x[np.newaxis,:]


class Network:
    '''A power network for a specified timespan'''
    def __init__(
            self,
            B,
            line_lims,
            profile,
            dis_max,
            total_storage,
            storage_cycle_timesteps,
            cost_coeffs,
            cycle_storage
        ):
        self.B = B.copy()
        self.line_lims = line_lims.copy()
        self.profile = profile.copy()
        self.dis_max = dis_max.copy()
        self.total_storage = total_storage
        self.storage_cycle_timesteps = storage_cycle_timesteps
        self.cost_coeffs = cost_coeffs.copy()
        self.storage_capacity = None
        self.initial_charge = None
        self.cycle_storage = cycle_storage
        self.delta = (
            self.profile.nom_load*self.profile.voll**self.profile.elasticity
            /(
                self.profile.nom_price**self.profile.elasticity
                - self.profile.voll**self.profile.elasticity
            )
        )

    def quantity_demanded(self, pi): 
        return (
            (pi/self.profile.nom_price)**self.profile.elasticity*(self.profile.nom_load + self.delta) 
            - self.delta
        )
    
    def quantity_supplied(self, pi):
        return (
            self.profile.int_gen 
            + np.minimum(
                np.maximum(0,pi - self.cost_coeffs[0,:])/2/self.cost_coeffs[1,:],
                self.dis_max
            )
        )

    def dis_cost(self,p):
        '''Helper function that gives the generation cost as a function of _dispatchable_ generation `p`'''
        return cp.hstack([p, cp.square(p)])@cp.vstack([np.diag(self.cost_coeffs[0,:]), np.diag(self.cost_coeffs[1,:])])

    def cost(self, p):
        return cp.maximum(0, self.dis_cost(p-self.profile.int_gen))

    def utility(self, p):
        return cp.bmat(
            [
                [
                    self.profile.nom_price.iloc[i,j]
                    *self.profile.elasticity.iloc[i,j]
                    /(self.profile.nom_load.iloc[i,j]+self.delta.iloc[i,j])**(1/self.profile.elasticity.iloc[i,j])
                    /(self.profile.elasticity.iloc[i,j] + 1)
                    *(
                        (p[i,j]+self.delta.iloc[i,j])**(1+1/self.profile.elasticity.iloc[i,j])
                        - self.delta.iloc[i,j]**(1+1/self.profile.elasticity.iloc[i,j])
                    )
                    for j in range(p.shape[1])
                ]
                for i in range(p.shape[0])
            ]
        )

    def solve(self):
        '''Optimize the dipatch of generators, loads, and storage.'''

        # Helpful variables
        generators = np.array(self.profile.int_gen.columns)
        loads = np.array(self.profile.nom_load.columns)
        n = self.B.shape[0]
        n_g = len(generators)
        n_l = len(loads)
        t = len(self.profile.index)
        TG = np.zeros((n,n_g))
        TG[generators-1,:] = np.eye(n_g)
        TL = np.zeros((n,n_l))
        TL[loads-1,:] = np.eye(n_l)

        # CVXPY variables
        load = cp.Variable((t,n_l))
        generation = cp.Variable((t,n_g))
        storage_load = cp.Variable((t,n))
        storage_capacity = cp.Variable(n)
        initial_charge = cp.Variable(n)
        angle = cp.Variable((t,n))

        # Define constraints
        constraints = [
            TG@generation.T-TL@load.T-storage_load.T == -self.B@angle.T,
            generation <= self.dis_max + self.profile.int_gen,
            generation >= 0,
            cp.cumsum(storage_load, axis=0) >= -initial_charge[np.newaxis,:],
            cp.cumsum(storage_load, axis=0) <= (storage_capacity-initial_charge)[np.newaxis,:],
            storage_load <= storage_capacity[np.newaxis,:]/self.storage_cycle_timesteps,
            storage_load >= -storage_capacity[np.newaxis,:]/self.storage_cycle_timesteps,
            cp.sum(storage_capacity) == self.total_storage,
            initial_charge >= 0,
            initial_charge <= storage_capacity,
        ] + [
            cp.multiply(self.B, outer_difference(angle_t)) <= self.line_lims
            for angle_t in angle
        ]
        constraints = (
            constraints + [cp.sum(storage_load,axis=0) == 0] if self.cycle_storage
            else constraints
        )

        opt_cost = self.cost(generation)
        opt_utility = self.utility(load)

        opf = cp.Problem(
            cp.Minimize(
                (cp.sum(opt_cost)-cp.sum(opt_utility))
            ),
            constraints
        )

        # Solve with CVXPY
        opf.solve()

        # Extract the prices, which are the dual of the power flow constraints
        price = -opf.constraints[0].dual_value.T

        # Construct an output dataframe to be concatenated with the input profile
        output = pd.concat(
            {
                'load': pd.DataFrame(load.value,columns=loads),
                'generation': pd.DataFrame(generation.value,columns=generators),
                'storage_load': pd.DataFrame(storage_load.value,columns=range(1,n+1)),
                'angle': pd.DataFrame(angle.value,columns=range(1,n+1)),
                'price': pd.DataFrame(price,columns=range(1,n+1)),
                'cost': pd.DataFrame(opt_cost.value,columns=generators),
                'utility': pd.DataFrame(opt_utility.value,columns=loads)
            },
            axis='columns'
        )
        output.index = self.profile.index

        # Update the object with simulation outputs
        self.profile = (
            self.profile.drop(columns=output.columns)
            if output.columns.isin(self.profile.columns).all()
            else self.profile
        )
        self.profile = pd.concat((self.profile,output),axis='columns')
        self.storage_capacity = storage_capacity.value
        self.initial_charge = initial_charge.value
        