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

# from typing_extensions import final
import re
import numpy as np
from numpy.linalg import norm
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
# from main import *
import settings
from quadprog import solve_qp



def template_mpc(model,my_ID,n_agents,final_pos):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)
    n_horizon = 20
    delT = 0.1
    n_samples = 10
    cvar_cons = True
    scenario = False
    deterministic = False

    soft = True

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': n_horizon,
        't_step': delT,
        'state_discretization': 'discrete',
        'store_full_solution':True,
    }

    theta_p = 0.001
    alpha = 0.05
    d_min = 0.01

    tgt = 1e5
    max_vl = inf

    risk_tol = 0.0

    mpc.set_param(**setup_mpc)
    # mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'MA27'})
    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0, 'ipopt.linear_solver': 'MA27'}
    no_suppress_ipopt = {'ipopt.linear_solver': 'MA27'}
    if my_ID != 0:
        mpc.set_param(nlpsol_opts = suppress_ipopt)
    else:
        mpc.set_param(nlpsol_opts = suppress_ipopt)
    

    # if my_ID!=1:
    #     suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    #     mpc.set_param(nlpsol_opts = suppress_ipopt)
    
    X = model.x['x', 0]
    Y = model.x['x', 1]
    Z = model.x['x', 2]
    X_dot = model.x['x', 3]
    Y_dot = model.x['x', 4]
    Z_dot = model.x['x', 5]
    Phi = model.x['x', 6]
    Theta = model.x['x', 7]
    Psi = model.x['x', 8]
    Phi_dot = model.x['x', 9]
    Theta_dot = model.x['x',10]
    Psi_dot = model.x['x', 11]
    
    _x = model.x['x']
    _u = model.u['u']
    _z = np.array([[_x[0]],[_x[1]],[_x[2]]])

    #_lambda = *model.u['lambda']
    
    # Q = np.zeros((12,12))
    # Q[0,0] = 1
    # Q[1,1] = 1
    # Q[2,2] = 1
    # Q[3,3] = 1
    # Q[4,4] = 1
    # Q[5,5] = 1
    # Q[6,6] = 5
    # Q[7,7] = 5
    # Q[8,8] = 5
    # Q[9,9] = 1
    # Q[10,10] = 1
    # Q[11,11] = 1
    
    # R = np.zeros((4,4))
    # R[0,0] = 0.1
    # R[1,1] = 5
    # R[2,2] = 5
    # R[3,3] = 5
    
    
    # Q = np.zeros((12,12))
    # Q[0,0] = .3
    # Q[1,1] = .3
    # Q[2,2] = .3
    # Q[3,3] = .5
    # Q[4,4] = .5
    # Q[5,5] = .5
    # Q[6,6] = .5
    # Q[7,7] = .5
    # Q[8,8] = 5
    # Q[9,9] = 2
    # Q[10,10] = 2
    # Q[11,11] = 2
    
    # R = np.zeros((4,4))
    # R[0,0] = 5
    # R[1,1] = 1
    # R[2,2] = 1
    # R[3,3] = 1

    Q = np.zeros((12,12))
    Q[0,0] = 3
    Q[1,1] = 3
    Q[2,2] = 3
    Q[3,3] = 5
    Q[4,4] = 5
    Q[5,5] = 5
    Q[6,6] = 5
    Q[7,7] = 5
    Q[8,8] = 5
    Q[9,9] = 2
    Q[10,10] = 2
    Q[11,11] = 2
    
    R = np.zeros((4,4))
    R[0,0] = 5
    R[1,1] = 15
    R[2,2] = 15
    R[3,3] = 15
    
    
    
    
    x_ref = np.zeros((12,1))
    x_ref[2] = final_pos[2]
    x_ref[1] = final_pos[1]
    x_ref[0] = final_pos[0]
    err_x = _x - x_ref
    
    mterm = (err_x.T)@Q@err_x #+ (_u.T)@R@_u
    lterm = mterm + (_u.T)@R@_u

    # for i in (j for j in range(n_agents) if j!= my_ID):
    #     extra_var_name = 'e_ORCA' + str(i+1)
    #     e = model.tvp[extra_var_name]
    #     lterm += 1000*e*e

    #(_u.T)@R@_u
    
    # mterm = model.aux['cost']
    # lterm = model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=5*np.ones((4,1)))


    # max_x = np.array([[30.0], [30.0], [30.0], [1.5], [1.5], [0.5], [0.3], [0.3], [0.3], [0.5], [0.5], [0.5]])
    # max_u = np.array([[2],[1],[1],[1]])
    max_x = np.array([[20.0], [20.0], [12.0], [1.75], [1.75], [1.5], [1.5], [1.5], [1.5], [1.5], [1.5], [1.5]])
    max_u = np.array([[5],[2],[2],[2]])
    #min_u = np.array([[-0.2],[-2],[2],[2]])

    mpc.bounds['lower','_x','x'] = -max_x
    mpc.bounds['upper','_x','x'] =  max_x
    

    mpc.bounds['lower','_u','u'] = -max_u
    mpc.bounds['upper','_u','u'] =  max_u

    

    
    


    tvp_temp = mpc.get_tvp_template()
    # get_Odom(model, my_ID, n_agents,OdomStruct)
    def get_Odom(model):
        # import settings
        from main import get_G_and_H, dist_from_poly
        for i in (j for j in range(n_agents) if j!= my_ID):
            G_var_name = 'G' + str(i+1)
            H_var_name = 'H' + str(i+1)

            zj_var_name = 'zj_' + str(i+1)

            sum_var_name = 'sum' + str(my_ID+1) + '_for_agent' + str(i+1)

            for k in range(n_horizon+1):
                try:

                    (G,H) = get_G_and_H(settings.PredStruct[i][:,k+1]) ####today
                    # (G,H) = get_G_and_H(settings.PredStruct[i][:,k,0])
                    # (G,H) = get_G_and_H(settings.PredStruct[i][:,k]) ### original
                    # if my_ID == 0:
                    # print("got the predictions")
                except: 
                    (G,H) = get_G_and_H(settings.OdomStruct[i])
                    # if my_ID == 0:
                    # print("did not get the predictions")
                
                (G_,H_) = get_G_and_H(settings.OdomStruct[i])
                
                tvp_temp['_tvp',k,G_var_name] = G_
                tvp_temp['_tvp',k,H_var_name] = H_

                try:
                    if k == 0:
                        zj = np.zeros((3,1)) #### condition on k==0 introduced today
                    elif k==n_horizon:
                        zj = settings.PredStruct[i][0:3,k].reshape((3,1)) - settings.OdomStruct[i][0:3].reshape((3,1)) ### today additional condition for k
                    else:    
                        zj = settings.PredStruct[i][0:3,k+1].reshape((3,1)) - settings.OdomStruct[i][0:3].reshape((3,1)) ### original and always
                    # print('prediction shape is ',settings.PredStruct[i].shape)
                    # print(str(zj)+' '+str(k))
                    # print('zj working')
                except:
                    zj = settings.OdomStruct[i][0:3].reshape((3,1))
                    print('zj exception')
                # if my_ID == 0:
                #     print('*** printing zj at ',k)
                #     print(zj.T)
                tvp_temp['_tvp',k,zj_var_name] = zj
                tvp_temp['_tvp',k,sum_var_name] = 0

                if (scenario):
                    for w in range(n_samples):
                        try: 
                            G_var_name = 'G_scenario' + str(i+1) + '_for_sample' + str(w)
                            H_var_name = 'H_scenario' + str(i+1) + '_for_sample' + str(w)
                            (g,h) = get_G_and_H(settings.ErrorLog[i][k-1][-1-w].reshape(3)+settings.PredStruct[i][:,k][0:3].reshape(3))
                            tvp_temp['_tvp',k,G_var_name] = g
                            tvp_temp['_tvp',k,H_var_name] = h
                        except:
                            tvp_temp['_tvp',k,G_var_name] = G
                            tvp_temp['_tvp',k,H_var_name] = H
                    






                #############  tvp code for error data set 
                w_star_var_name = 'w*_' + str(i+1)
                w_star_list = []
                

                for w in range(n_samples):
                    try:
                        if k!=0:
                            w_star_list.append(settings.ErrorLog[i][k-1][-1-w].reshape(3))
                            # if (k==1 or k==0 or k==2) and my_ID == 0:
                            #     print('error @'+str(k)+' sample '+str(w)+' is ')
                            #     print(settings.ErrorLog[i][k-1][-1-w].reshape(3))
                            
                            # print('working!')
                        else:
                            w_star_list.append(settings.ErrorLog[i][0][-1-w].reshape(3))

                        # w_star_list.append(np.array([0,0,0]))
                    except: 
                        w_star_list.append(np.array([0,0,0]))
                        # print('appending zero!')
                        # w_star_list.append(settings.ErrorLog[i][k][-1-w].reshape(3))
                        # pass
                        # if k == 1:
                        #     print('appending zero! @'+str(k)+' sample '+str(w))
                
                # print('error samples')
                # if my_ID == 0:
                #     print(w_star_list[-2])
                w_star = np.array(w_star_list)
                tvp_temp['_tvp',k,w_star_var_name] = w_star # w_star shape (n_samples,3)


            # """ COLLISION CHECK """
            # (qp_G,qp_H) = get_G_and_H(settings.OdomStruct[i])
            # p = np.array([settings.OdomStruct[my_ID][0],settings.OdomStruct[my_ID][1],settings.OdomStruct[my_ID][2]])
            # dist = dist_from_poly(qp_G,qp_H,p)
            # if(my_ID==0):
            #     print('dist: ',dist)
            #     print('pos: ', p.T)
            #     print('qp_H: ', qp_H.T)
            #     # print('qp_G: ', qp_G.T)

            
            # if dist < 0.1:
            #     print('***** COLLISION dist: ',dist)
        return tvp_temp        


    

    # mpc.set_tvp_fun(get_Odom(model,my_ID,n_agents,OdomStruct))
    mpc.set_tvp_fun(get_Odom)

    # for j in (j for j in range(n_agents) if j!= my_ID):
    #     lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1)
    #     G_var_name = 'G' + str(j+1)
    #     H_var_name = 'H' + str(j+1)
    #     G = model.tvp[G_var_name]
    #     H = model.tvp[H_var_name]
    #     _lambda = model.u[lambda_var_name]

    #     lambda_lower_value = 0.03*np.ones((6,1))
    #     lambda_lower_value[1] = 0.06
    #     lambda_lower_value[3] = 0.06
    #     lambda_lower_value[5] = 0.06
    #     mpc.bounds['lower','_u',lambda_var_name] = lambda_lower_value


    #     _obst_con = -((G@_z - H).T)@_lambda
    #     GL = (G.T)@_lambda
    #     _obst_con_abs = norm_2(GL)
    #     # if my_ID == 1:
    #     #     mpc.set_nl_cons("obst_con"+str(i), _obst_con, ub=0,  soft_constraint=True, penalty_term_cons=100)
    #     #     mpc.set_nl_cons("obst_con_abs"+str(i), _obst_con_abs, ub=1, soft_constraint=True, penalty_term_cons=100)

    #     if deterministic:
    #         mpc.set_nl_cons('obst_con'+str(j), _obst_con, ub=-d_min,  soft_constraint=soft,  penalty_term_cons=tgt,maximum_violation=max_vl)
    #         mpc.set_nl_cons('obst_con_abs'+str(j), _obst_con_abs, ub=1, soft_constraint=soft,  penalty_term_cons=tgt,maximum_violation=max_vl)

    #     if scenario:
    #         for w in range(n_samples):
    #             lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(w)
    #             G_var_name = 'G_scenario' + str(j+1) + '_for_sample' + str(w)
    #             H_var_name = 'H_scenario' + str(j+1) + '_for_sample' + str(w)

    #             G = model.tvp[G_var_name]
    #             H = model.tvp[H_var_name]
    #             _lambda = model.u[lambda_var_name]

    #             lambda_lower_value = 0.03*np.ones((6,1))
    #             lambda_lower_value[1] = 0.06
    #             lambda_lower_value[3] = 0.06
    #             lambda_lower_value[5] = 0.06
    #             mpc.bounds['lower','_u',lambda_var_name] = lambda_lower_value


    #             _obst_con = -((G@_z - H).T)@_lambda
    #             GL = (G.T)@_lambda
    #             _obst_con_abs = norm_2(GL)
    #             mpc.set_nl_cons('obst_con_scenario'+str(j)+'sample'+str(w), _obst_con, ub=-d_min,  soft_constraint=soft, penalty_term_cons=1000)
    #             mpc.set_nl_cons('obst_con_abs_scenario'+str(j)+'sample'+str(w), _obst_con_abs, ub=1, soft_constraint=soft, penalty_term_cons=1000)


    

    # for j in (j for j in range(n_agents) if j!= my_ID):
    #     lambda_theta_var_name = 'lambda_theta' + str(my_ID+1) + '_for_agent' + str(j+1)
    #     t_var_name = 't' + str(my_ID+1) + '_for_agent' + str(j+1)
    #     mpc.bounds['lower','_u',lambda_theta_var_name] = 0
    #     for n in range(n_samples):
    #         lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
    #         lambda_lower_value = 0.03*np.ones((6,1))
    #         lambda_lower_value[1] = 0.06
    #         lambda_lower_value[3] = 0.06
    #         lambda_lower_value[5] = 0.06
    #         mpc.bounds['lower','_u',lambda_var_name] = lambda_lower_value

    #         s_n_var_name = 'si' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
    #         mpc.bounds['lower','_u',s_n_var_name] = 0


        
    ## parameters
    


    for j in (j for j in range(n_agents) if j!= my_ID):
        # _sum = SX.sym('_sum')
        # _sum = 0



        sum_var_name = 'sum' + str(my_ID+1) + '_for_agent' + str(j+1)
        _sum = model.tvp[sum_var_name]


        lambda_theta_var_name = 'lambda_theta' + str(my_ID+1) + '_for_agent' + str(j+1)
        _lambda_theta = model.u[lambda_theta_var_name]


        t_var_name = 't' + str(my_ID+1) + '_for_agent' + str(j+1)
        _t = model.u[t_var_name]

        
        w_star_var_name = 'w*_' + str(j+1)
        w_star = model.tvp[w_star_var_name]

        zj_var_name = 'zj_' + str(j+1)
        _zj = model.tvp[zj_var_name]
        print(_zj)

        G_var_name = 'G' + str(j+1)
        H_var_name = 'H' + str(j+1)
        G = model.tvp[G_var_name]
        H = model.tvp[H_var_name]

        mpc.bounds['lower','_u',lambda_theta_var_name] = 0

        
        for n in range(n_samples):
            lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
            s_n_var_name = 'si' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
            

            

            

            lambda_lower_value = 0.03*np.ones((6,1))
            lambda_lower_value[1] = 0.06
            lambda_lower_value[3] = 0.06
            lambda_lower_value[5] = 0.06


            # lambda_lower_value = 0.01*np.ones((6,1))
            # lambda_lower_value[1] = 0.02
            # lambda_lower_value[3] = 0.02
            # lambda_lower_value[5] = 0.02
            mpc.bounds['lower','_u',lambda_var_name] = lambda_lower_value

            mpc.bounds['lower','_u',s_n_var_name] = 0

            _lambda = model.u[lambda_var_name]
            _s = model.u[s_n_var_name]
            _sum = _sum + _s


            GL = (G.T)@_lambda
            _obst_con_abs = norm_2(GL)

            # _constraintB = d_min + _t - _s - (((G@_z - H).T)@_lambda) + _lambda.T@G@(w_star[n,:].reshape((3,1)))
            _constraintB = d_min + _t - _s - (((G@(_z-_zj) - H).T)@_lambda) + _lambda.T@G@(w_star[n,:].reshape((3,1)))

            if cvar_cons:
                mpc.set_nl_cons('cons_b'+str(j)+'sample'+str(n), _constraintB, ub=0, soft_constraint=False, penalty_term_cons=tgt,maximum_violation=max_vl)
                mpc.set_nl_cons('obst_con_abs'+str(j)+'sample'+str(n), _obst_con_abs, ub=1, soft_constraint=False, penalty_term_cons=tgt,maximum_violation=max_vl)
                mpc.set_nl_cons('cons_theta'+str(j)+'sample'+str(n), _obst_con_abs-_lambda_theta, ub=0, soft_constraint=False, penalty_term_cons=tgt,maximum_violation=max_vl)


        


        _constraintA = _lambda_theta*theta_p - _t*alpha + (1/n_samples)*(_sum)
        if cvar_cons:
            mpc.set_nl_cons('cons_a'+str(j), _constraintA, ub=risk_tol, soft_constraint=soft, penalty_term_cons=tgt,maximum_violation=max_vl)





    mpc.setup()

    return mpc


