#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

from ctypes.wintypes import VARIANT_BOOL
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_model(my_ID, n_agents):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(12,1))
    _va = model.set_variable(var_type='_tvp', var_name='v_a', shape=(2,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(4,1))

    # x_var_name = 'x' + str(i+1)
    # u_var_name = 'u' + str(i+1)
    # G_var_name = 'G' + str(my_ID+1)
    # H_var_name = 'H' + str(my_ID+1)
    # # _x = model.set_variable(var_type='_x', var_name=x_var_name, shape=(12,1))
    # # _u = model.set_variable(var_type='_u', var_name=u_var_name, shape=(4,1))
    # # x_next = _x + delT*(A@_x+B@_u)
    # # model.set_rhs(x_var_name, x_next)
    # _G = model.set_variable(var_type='_tvp', var_name=G_var_name, shape=(6,3))
    # _H = model.set_variable(var_type='_tvp', var_name=H_var_name, shape=(6,1))

    for i in (j for j in range(n_agents) if j!=my_ID):
        
        lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(i+1)
        _lambda = model.set_variable(var_type='_u', var_name=lambda_var_name, shape=(6,1))
        G_var_name = 'G' + str(i+1)
        H_var_name = 'H' + str(i+1)
        _G = model.set_variable(var_type='_tvp', var_name=G_var_name, shape=(6,3))
        _H = model.set_variable(var_type='_tvp', var_name=H_var_name, shape=(6,1))

    n_samples = 10


    for j in (j for j in range(n_agents) if j!= my_ID):

        lambda_theta_var_name = 'lambda_theta' + str(my_ID+1) + '_for_agent' + str(j+1)
        t_var_name = 't' + str(my_ID+1) + '_for_agent' + str(j+1)
        w_star_var_name = 'w*_' + str(j+1)
        zj_var_name = 'zj_' + str(j+1)
        
        _lambda_theta = model.set_variable(var_type='_u', var_name=lambda_theta_var_name, shape=(1,1))
        _t = model.set_variable(var_type='_u', var_name=t_var_name, shape=(1,1))
        _w_star = model.set_variable(var_type='_tvp', var_name=w_star_var_name, shape=(n_samples,3))

        _zj = model.set_variable(var_type='_tvp', var_name=zj_var_name, shape=(3,1))

        _s = model.set_variable(var_type='_tvp', var_name='sum' + str(my_ID+1) + '_for_agent' + str(j+1))

       
        for n in range(n_samples):
            lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
            _lambda = model.set_variable(var_type='_u', var_name=lambda_var_name, shape=(6,1))
            

            s_n_var_name = 'si' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
            _s = model.set_variable(var_type='_u', var_name=s_n_var_name, shape=(1,1))

            G_var_name = 'G_scenario' + str(j+1) + '_for_sample' + str(n)
            H_var_name = 'H_scenario' + str(j+1) + '_for_sample' + str(n)
            _G = model.set_variable(var_type='_tvp', var_name=G_var_name, shape=(6,3))
            _H = model.set_variable(var_type='_tvp', var_name=H_var_name, shape=(6,1))


        



        

            

    


    
    g = 9.81
    delT = 0.1
    
    A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,g,0,0,0,0],
                  [0,0,0,0,0,0,-g,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0],])
    
    B = np.array([[0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [1,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1],])
    
    x_tilde = SX.sym('x_tilde',12,1)
    x_tilde[0] = _x[3]
    x_tilde[1] = _x[4]
    x_tilde[2] = _x[5]
    x_tilde[3]= (g+_u[0])*sin(_x[7])#g*sin(_x[7])
    x_tilde[4] = -(g+_u[0])*sin(_x[6])#-g*sin(_x[6])
    x_tilde[5] = _u[0]*cos(_x[6])*cos(_x[7])
    x_tilde[6] = _x[9]
    x_tilde[7] = _x[10]
    x_tilde[8] = _x[11]
    x_tilde[9] = _u[1]
    x_tilde[10] = _u[2]
    x_tilde[11] = _u[3]
    

    # x_next = _x + delT*(A@_x+B@_u)
    x_next = _x + delT*(x_tilde)
    model.set_rhs('x', x_next)

    model.setup()

    return model
