from scipy import optimize


import numpy as np
import scipy as sp
from scipy import linalg
from scipy import optimize
import sympy as sm
from types import SimpleNamespace

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from scipy.optimize import Bounds, minimize


#Due to time, full documentation of this code file is pending...

def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result

class SolowModelClass():
    def __init__(self, s=0.2, g=0.02, n=0.01, alpha=1/3, delta=0.1):
        self.s = s
        self.g = g
        self.n = n
        self.alpha = alpha
        self.delta = delta

    def output(self, k):
        return k**self.alpha
    
    def capital_acc(self, k):
     return 1/((1+self.n)*(1+self.g)) * (self.s*self.output(k)+(1-self.delta)*k)
    

    def solve_ss(self):
        #output:
        f = lambda k: k**self.alpha

        #steady state:
        obj_kss = lambda kss: kss - (self.s*f(kss) + (1-self.delta)*kss)/((1+self.g)*(1+self.n))
        result = optimize.root_scalar(obj_kss,bracket=[0.0001,100000],method='brentq')

        return result.root
    
    def simulate(self, kmin=0.0001, kmax=5, T=100000):
        
        #Defining transition path for k starting below steady state:
        k_low = np.zeros(T)
        y_low = np.zeros(T)
        k_t_plus_one_low= np.zeros(T)
        #Setting initial values:
        k_low[0] = kmin 
        y_low[0] = kmin 
        k_t_plus_one_low[0] = self.capital_acc(k_low[0])

        for t in range(1, T):
            k_low[t]= self.capital_acc(k_low[t-1])
            y_low[t] = self.output(k_low[t])
            k_t_plus_one_low[t] = self.capital_acc(k_low[t])

        #Defining transition path for k starting above steady state:
        k_high = np.zeros(T)
        y_high = np.zeros(T)
        k_t_plus_one_high= np.zeros(T)
        #Setting initial values:
        k_high[0]=kmax
        y_high[0] = self.output(k_high[0]) 
        k_t_plus_one_high[0] = self.capital_acc(k_high[0])

        for t in range(1, T):
            k_high[t]= self.capital_acc(k_high[t-1])
            y_high[t] = self.output(k_high[t])
            k_t_plus_one_high[t] = self.capital_acc(k_high[t])


        #Combining:
        k_vec = np.concatenate((k_low,k_high))
        y_vec = np.concatenate((y_low,y_high))
        k_t_plus_one_vec = np.concatenate((k_t_plus_one_low, k_t_plus_one_high))

        return k_vec, y_vec, k_t_plus_one_vec
    



    #CREATING INTERACTIVE FUNCTION FOR CAPITAL TRANSITION
def interactive(kmin, kmax, s, g, n, alpha, delta):

    #Calling class with given parameters:
    model = SolowModelClass(s, g, n, alpha, delta)
    k, y , k_plus_one = model.simulate(kmin, kmax) #Setting initial capital, and number of accumulation iterations
    kss = model.solve_ss()


    #Plotting k and k_t+1, as well as a 45-degree line
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(k, k_plus_one, linewidth=1, color='blue') 
    ax.axline([0,0],[1,1], linewidth=0.7, color='black')
    ax.set_xlabel('$k_t$')
    ax.set_ylabel('$k_{t+1}$')
    ax.annotate(f'Steady state, $k_{{t+1}}=k_t$ = {np.round(kss,3)}', xy=(kss, kss), #I've added the SS function directly as arrow values
    xycoords='data',
            xytext=(-100,60), textcoords='offset points',
            arrowprops=dict(arrowstyle='fancy',fc='0.6',
                            connectionstyle="angle3,angleA=0,angleB=-90"))
    ax.set_title('Capital Transition in the general Solow model')




#CREATING INTERACTIVE FUNCTION FOR OUTPUT/CAPITAL
def interactive_output(s, g, n, alpha, delta):

    #Calling class with given parameters:
    model = SolowModelClass(s, g, n, alpha, delta)
    k, y , k_plus_one = model.simulate(kmin=0.000001) #Setting initial capital, and number of accumulation iterations
    kss = model.solve_ss()


    #Plotting k and k_t+1, as well as a 45-degree line
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(k[0:50000], y[0:50000], linewidth=1, color='blue') 
    ax.set_xlabel('$k_t$')
    ax.set_ylabel('$y_t$')
    ax.set_title('The per capita production function in General Solow')



class SolowModelClass_HC():
    def __init__(self, s_h=0.2,s_k=0.3, g=0.02, n=0.01, alpha=1/3, delta=0.1, phi=0.2, k0=1, h0=1):
        self.s_h = s_h
        self.s_k = s_k
        self.g = g
        self.n = n
        self.alpha = alpha
        self.delta = delta
        self.phi = phi
        self.k0 = k0 #initial capital
        self.h0 = h0 #initinal human capital
        self.y0 = k0**self.alpha * h0**self.phi
        

    
    def output(self, k, h):
        return k**self.alpha * h**self.phi
    
    def capital_acc(self, k, h):
        return 1/((1+self.n)*(1+self.g)) * (self.s_k*self.output(k, h)+(1-self.delta)*k)
    
    def human_capital_acc(self,k,h):
        return 1/((1+self.n)*(1+self.g)) * (self.s_h*self.output(k,h)+(1-self.delta)*h)
    

    #We tried many different objective functions and methods to solve the steady state values using different scipy method, but were unable to figure it out
    # def solve_ss(self):
    #     #steady state:
    #     bounds = Bounds([0,0], [2000,2000])
    #     x0=[self.k0, self.h0]
    #     obj_kss = lambda kss: kss - self.capital_acc(kss, x0[1])
    #     result = optimize.minimize(obj_kss, x0, bounds=bounds)

    #     return result.x

    def hardcoded_steady_states(self):
        k_ss = (self.s_k**(1-self.phi)*self.s_h**self.phi / (self.n + self.g + self.delta + self.n*self.g))**(1/(1-self.alpha-self.phi))
        h_ss = (self.s_k**(self.alpha)*self.s_h**(1-self.alpha) / (self.n + self.g + self.delta + self.n*self.g))**(1/(1-self.alpha-self.phi))
        return k_ss, h_ss
    
    def simulate(self, T=100000):
        h_vec = np.zeros(T)
        k_vec = np.zeros(T)
        y_vec = np.zeros(T)

        h_vec[0]=self.h0
        k_vec[0]=self.k0
        y_vec[0]=self.output(k_vec[0], h_vec[0])


        for t in range(1,T):
            k_vec[t]=self.capital_acc(k_vec[t-1], h_vec[t-1])
            h_vec[t]=self.human_capital_acc(k_vec[t-1], h_vec[t-1])
            y_vec[t]=self.output(k_vec[t], h_vec[t])
       

        return k_vec, h_vec, y_vec
    




def interactive_solow_HC(s_h,s_k, g, n, alpha, delta, phi):
    T=1000
    #Calling class with given parameters:
    model = SolowModelClass_HC(s_h,s_k, g, n, alpha, delta, phi, k0=0.01, h0=0.01)
    k_vec, h_vec , y_vec = model.simulate(T=T) #Setting initial capital, and number of accumulation iterations
    kss, hss = model.hardcoded_steady_states()
    
    k_index  = np.where(np.round(k_vec,1) == np.round(kss,1))[0][0]

    h_index  = np.where(np.round(h_vec,1) == np.round(hss,1))[0][0]

    print(f'Capital steady state = {kss}, Human capital steady state {hss}')


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    ax1.plot(range(0,T), k_vec, linewidth=1, color='blue')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$k_t$')
    ax1.set_title('How many periods for $k_t$ to converge to steady state')
    # ax1.annotate(f'$k^*$ = {np.round(kss,3)}, periods for convergence = {k_index}', xy=(k_index, kss), #I've added the SS function directly as arrow values
    # xycoords='data',
    #         xytext=(-100,60), textcoords='offset points',
    #         arrowprops=dict(arrowstyle='fancy',fc='0.6',
    #                         connectionstyle="angle3,angleA=0,angleB=-90"))
    # ax1.set_title('How many periods for $k_t$ to converge to steady state')

    ax2.plot(range(0,T), h_vec, linewidth=1, color='red')
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$h_t$')
    # ax2.annotate(f'$h^*$ = {np.round(hss,3)}, periods for convergence = {h_index}', xy=(h_index, kss), #I've added the SS function directly as arrow values
    # xycoords='data',
    #         xytext=(-100,60), textcoords='offset points',
    #         arrowprops=dict(arrowstyle='fancy',fc='0.6',
    #                         connectionstyle="angle3,angleA=0,angleB=-90"))
    ax2.set_title('How many periods for $h_t$ to converge to steady state')