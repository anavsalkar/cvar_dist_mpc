#!/home/atharva/BTP/btp/bin/python

# importing the mp module
import multiprocessing as mp
import os
import tarfile
from main import *
import rospy
from num_sim import *


if __name__ == "__main__":

    n_sim = 25
    coll_array = []
    is_collision = mp.Value('d', 0.0)
    im_done = mp.Value('d', 0.0)

    sol_array = []
    n_agents = 2

    for i in range(n_agents):
        sol_array.append(mp.Value('d',1.0))
    sol_array.append(mp.Value('d',0.0))  ### value to indicate when I get next state
    sol_array.append(mp.Value('d',1.0))

    # dir_name = 'gNb_0.5_al_0.25_th_0.01_ss_10_sgN_0.2_v15-100_1000t_0.5m'
    # dir_name = 'my_alpha_3_0.2_off_0.3'
    # dir_name = '20_alpha_0.15'
    # dir_name = '1_3_p3_r1_al_0.4_v3' # noise 1,3 | final pos 3 | 
    # dir_name = '21_noise_0.2_det'
    # dir_name = '21_sample_10_noise_0.6_v2'
    # dir_name = '22_cvr_0.1_noise_p3.5'
    # dir_name = '22_noise_0.8_pred' ## pred or vel
    # dir_name = '22_no_noise_p3.5_det_vel' ## cvr or det
    # dir_name = '22_1_3_p3_r1_al_0.35_v2'
    # dir_name = '22_p3.5_5samples_1_noise_0.1_alpha'
    # dir_name = 'p3.5_noise_0.5_alpha_0.5_risk_0.05_r_0.01_v2'
    # dir_name = 'no_noise_alpha_0.1_r_0.1'
    # dir_name = 'sync_p_3.5_gN_0_soft_5e3_0.1_hhs_al_0.35_th_0.01'
    # dir_name = 'sync_soft_1e5_al_0.9_th_0.001'
    # dir_name = 'fin_alpha_0.9_theta_0.01'
    # dir_name = 'trial_T3_0.9'
    # dir_name = 'T30_p4_al_0.9_th_0.0001'
    # dir_name = 'sN_0.6_pN_1.8_al_0.05_th_0.01'
    # dir_name = 'traj'
    # dir_name = '31_p1_al_0.01_th_0.01'
    # dir_name = '1_p2_sN_0.6_pN_1.8_al_0.5_th_0.0001_T30'
    # dir_name = 'true_p1_al_0.01_th_0.0001_pos8_v2'
    # dir_name = 'true_p1_al_0.01_th_0.0001_pos8_v2'
    # dir_name = 'true_p2_al_0.05_th_0.001_pos8_sN_0.2_pN_0.6'
    # dir_name = 'true2_p1_al_0.01_th_0.01_pos8_4' ### 120 sims
    # dir_name = 'true2_p2_al_0.05_th_0.001_sN_0.8_pN_2.4_pos8_4' ### 180 sims
    dir_name = 'noise_const_vel_0'
    # dir_name = 'dummy2'
    ## 0.5 0.6 v2 has 5 cases
    ### 
    try: 
        os.mkdir(dir_name)
    except:
        pass

    for sim_id in range(n_sim):
        sim = mp.Process(target=start_sim, args=(is_collision,im_done,sim_id,sol_array))
        p1 = mp.Process(target=mpc, args=(0,n_agents,[4,4,5],is_collision,im_done,sim_id,dir_name,sol_array)) 
        p2 = mp.Process(target=mpc, args=(1,n_agents,[-4,-4,5],is_collision,im_done,sim_id,dir_name,sol_array)) 
        # p3 = mp.Process(target=mpc, args=(2,n_agents,[3.5,-4,5],is_collision,im_done,sim_id,dir_name)) 
        # p4 = mp.Process(target=mpc, args=(3,n_agents,[-3.5,4,5],is_collision,im_done,sim_id,dir_name)) 
        # starting processes
        p1.start()
        p2.start()
        # p3.start()
        # p4.start()
        sim.start()
        # wait until processes are finished
        p1.join()
        p2.join()
        # p3.join()
        # p4.join()
        sim.join()

        coll_array.append(is_collision.value)
        sol_array[-1].value = 1

        print("All processes finished execution!")
        np_coll_array = np.array(coll_array)
        np.save(dir_name+'/collision_array.npy', np_coll_array)

        im_done.value = 0.0
        is_collision.value = 0.0

    print('collision array: ', coll_array)
    np_coll_array = np.array(coll_array)
    np.save(dir_name+'/collision_array.npy', np_coll_array)

