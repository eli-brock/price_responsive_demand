import numpy as np
import cvxpy as cp
import pandas as pd

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
            cost_coeffs
        ):
        self.B = B
        self.line_lims = line_lims
        self.profile = profile
        self.dis_max = dis_max
        self.total_storage = total_storage
        self.storage_cycle_timesteps = storage_cycle_timesteps
        self.cost_coeffs = cost_coeffs

    def dis_cost(self,p):
        return (
            p@np.diag(self.cost_coeffs[0,:])
            + cp.square(p)@np.diag(self.cost_coeffs[1,:])
        )
    
    def cost(self,p):
        return cp.maximum(0, self.dis_cost(p-self.profile.int_gen))
    
    def utility(self,p):
        delta = (
            self.profile.nom_load*self.profile.voll**self.profile.elasticity
            /(
                self.profile.nom_price**self.profile.elasticity
                - self.profile.voll**self.profile.elasticity
            )
        )

        return cp.bmat(
            [
                [
                    self.profile.nom_price.iloc[i,j]
                    *self.profile.elasticity.iloc[i,j]
                    /(self.profile.nom_load.iloc[i,j]+delta.iloc[i,j])**(1/self.profile.elasticity.iloc[i,j])
                    /(self.profile.elasticity.iloc[i,j] + 1)
                    *(
                        (p[i,j]+delta.iloc[i,j])**(1+1/self.profile.elasticity.iloc[i,j])
                        - delta.iloc[i,j]**(1+1/self.profile.elasticity.iloc[i,j])
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
        load = cp.Variable((t,n_l), name='load')
        generation = cp.Variable((t,n_g), name='generation')
        storage_load = cp.Variable((t,n), name='storage_load')
        storage_capacity = cp.Variable(n, name='storage_capacity')
        initial_soc = cp.Variable(n, name='initial_soc')
        angle = cp.Variable((t,n), name='angle')

        # Define constraints
        constraints = [
            TG@generation.T-TL@load.T-storage_load.T == -self.B@angle.T,
            generation <= self.dis_max + self.profile.int_gen,
            generation >= 0,
            cp.cumsum(storage_load, axis=0) >= -initial_soc[np.newaxis,:],
            cp.cumsum(storage_load, axis=0) <= (storage_capacity-initial_soc)[np.newaxis,:],
            storage_load <= storage_capacity[np.newaxis,:]/self.storage_cycle_timesteps,
            storage_load >= -storage_capacity[np.newaxis,:]/self.storage_cycle_timesteps,
            cp.sum(storage_capacity) == self.total_storage,
            initial_soc >= 0,
            initial_soc <= storage_capacity,
        ] + [
            cp.multiply(self.B, angle_t[:,np.newaxis]-angle_t[np.newaxis,:]) <= self.line_lims
            for angle_t in angle
        ]

        opf = cp.Problem(
            cp.Minimize(
                (cp.sum(self.cost(generation))-cp.sum(self.utility(load)))
            ),
            constraints
        )

        opf.solve(verbose=True)

        price = -opf.constraints[0].dual_value.T
        output = pd.concat(
            {
                'load': pd.DataFrame(load.value,columns=loads),
                'generation': pd.DataFrame(generation.value,columns=generators),
                'storage_load': pd.DataFrame(storage_load.value,columns=range(1,n+1)),
                'angle': pd.DataFrame(angle.value,columns=range(1,n+1)),
                'price': pd.DataFrame(price,columns=range(1,n+1)),
                'consumer_surplus': pd.DataFrame(self.utility(load.value).value-price[:,loads-1]*load.value,columns=loads),
                'producer_surplus': pd.DataFrame(price[:,generators-1]*generation.value-self.cost(generation).value,columns=generators)
            },
            axis='columns'
        )
        output.index = self.profile.index

        self.profile = pd.concat((self.profile,output),axis='columns')
        