#!/home/atharva/BTP/btp/bin/python

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc

from std_srvs.srv import Empty
from numpy.lib.function_base import vectorize
import rospy
from std_msgs.msg import String
import numpy as np
import std_msgs.msg
import math
#import tf
from geometry_msgs.msg import Transform, Quaternion, Point, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from nav_msgs.msg import Odometry
import message_filters
import settings

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator



def control_callback(*args):
    global Ctrl_cmd
    Ctrl_cmd = []
    n_agents = len(args)
    for i in range(n_agents):
        Ctrl_cmd.append(args[i])
    # print('got control callback')

def euler_to_quaternion(roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def model(_x, _u, _t):
    g = 9.81
    x_tilde = np.zeros((12,1))
    x_tilde[0] = _x[3]
    x_tilde[1] = _x[4]
    x_tilde[2] = _x[5]
    x_tilde[3]= g*sin(_x[7])
    x_tilde[4] = -g*sin(_x[6])
    x_tilde[5] = _u[0]*cos(_x[6])*cos(_x[7])
    x_tilde[6] = _x[9]
    x_tilde[7] = _x[10]
    x_tilde[8] = _x[11]
    x_tilde[9] = _u[1]
    x_tilde[10] = _u[2]
    x_tilde[11] = _u[3]
    x_next = _x + _t*x_tilde

    return x_next


def sim_dyn(x0,u0,sim_delta):
    sim_delta_n = round(0.1/sim_delta)
    for i in range(sim_delta_n):
        # if(i%50 == 0):
            # print('inside sim_dyn')
        # x1 = model(x0,u0,sim_delta)
        # x0 = x1
        x0 = model(x0,u0,sim_delta)
    return x0

def next_state(sol_array,n_agents):
    return_value = 1
    for i in range(n_agents):
        if sol_array[i].value == 0:
            return_value = 0
    return return_value

def all_started(sol_array,n_agents):
    return_value = 1
    for i in range(n_agents):
        if sol_array[i].value == 1:
            return_value = 0
    return return_value

def start_sim(is_collision, im_done, sim_id, sol_array):
    
    """add parameters"""
    n_agents = 2
    sim_delta = 0.001
    sim_delta_t = 0.1#0.15

    """declaring variables"""
    sim_steps = 180                                      #time steps
    state = np.zeros((n_agents,12,sim_steps+1))              #3d state array

    # Subscriber
    rospy.init_node('sim_node', anonymous=True)
    sub_list = []
    for i in  range(n_agents):
        topic = '/firefly' + str(i+1) + '/command/trajectory'
        sub = message_filters.Subscriber(topic, MultiDOFJointTrajectory)
        sub_list.append(sub)

    ts = message_filters.TimeSynchronizer(sub_list, 20)
    ts.registerCallback(control_callback)

    # Publisher
    pub_list = []
    for i in range(n_agents):
        topic = '/firefly'+str(i+1)+'/odometry_sensor1/odometry'
        pub = rospy.Publisher(topic, Odometry, queue_size=20)
        pub_list.append(pub)

    # Initial State

    # Bot 1
    state[0,0,0] = -8
    state[0,1,0] = -8

    # Bot 2
    state[1,0,0] = 8
    state[1,1,0] = 8

    # # Bot 3
    # state[2,0,0] = -4
    # state[2,1,0] = 4

    # # Bot 4
    # state[3,0,0] = 4
    # state[3,1,0] = -4

    # # Bot 5
    # state[4,0,0] = 0
    # state[4,1,0] = -5

    # # Bot 6
    # state[5,0,0] = -3
    # state[5,1,0] = -5



    #### hexagon small
    # # Bot 1
    # state[0,0,0] = -2.5
    # state[0,1,0] = 4.33

    # # Bot 2im_done
    # state[1,0,0] = 2.5
    # state[1,1,0] = 4.33

    # # Bot 3
    # state[2,0,0] = 5
    # state[2,1,0] = 0

    # # Bot 4
    # state[3,0,0] = 2.5
    # state[3,1,0] = -4.33

    # # Bot 5
    # state[4,0,0] = -2.5
    # state[4,1,0] = -4.33

    # # Bot 6
    # state[5,0,0] = -5
    # state[5,1,0] = 0

    odom = Odometry()

    t_now = 0

    # Initialise
    t = 0

    in_exception = 0

    

    # Dynamics
    while t_now<sim_steps-1:
        try :

            u0 = np.zeros((4,1))
            
            u0[0] = Ctrl_cmd[0].points[-1].accelerations[0].linear.z
            u0[1] = Ctrl_cmd[0].points[-1].accelerations[0].angular.x
            u0[2] = Ctrl_cmd[0].points[-1].accelerations[0].angular.y
            u0[3] = Ctrl_cmd[0].points[-1].accelerations[0].angular.z

            

            #### synch code
            while next_state(sol_array,n_agents) == 0:
                rospy.sleep(0.01)
                # print('waiting numsim')
            sol_array[-1].value = 0
            sol_array[-2].value = 0

            if is_collision.value == 1 or im_done.value==1:
                break

            

            if t < 500:
                print('sim_step: ', t)
                print('sim_ID: ',sim_id)
                t_now = t
                


                for i in range(n_agents):
                    x0 = np.transpose(state[i,:,[t]])
                    # print('1',i)
                    # print('shape: ', state[1,1,:].shape)
                    u0 = np.zeros((4,1))
                    
                    u0[0] = Ctrl_cmd[i].points[-1].accelerations[0].linear.z
                    u0[1] = Ctrl_cmd[i].points[-1].accelerations[0].angular.x
                    u0[2] = Ctrl_cmd[i].points[-1].accelerations[0].angular.y
                    u0[3] = Ctrl_cmd[i].points[-1].accelerations[0].angular.z
                

                    # print('2',i)

                    x1 = sim_dyn(x0,u0,sim_delta)

                    # print('3',i)
                    [X,Y,Z,W] = euler_to_quaternion(x1[6],x1[7],x1[8])

                    # print('4',i)
                    odom.pose.pose.position.x = x1[0]
                    odom.pose.pose.position.y = x1[1]
                    odom.pose.pose.position.z = x1[2]
                    odom.twist.twist.linear.x = x1[3]
                    odom.twist.twist.linear.y = x1[4]
                    odom.twist.twist.linear.z = x1[5]
                    odom.pose.pose.orientation.x = X
                    odom.pose.pose.orientation.y = Y
                    odom.pose.pose.orientation.z = Z
                    odom.pose.pose.orientation.w = W           
                    odom.twist.twist.angular.x = x1[9] 
                    odom.twist.twist.angular.y = x1[10] 
                    odom.twist.twist.angular.z = x1[11] 
                    # print('5',i)

                    pub_list[i].publish(odom)
                    # print('6',i)
                    state[i,:,t+1] = x1[:,0]
                
                
                t = t + 1 

                # rospy.sleep(0.5)

                #### sync code
                sol_array[-2].value = 1
                sol_array[-1].value = 1
                while all_started(sol_array,n_agents) == 0:
                    rospy.sleep(0.005)
                    # print('waiting inside')
                sol_array[-2].value = 0
                
                # print('exited num sim')
                # rospy.sleep(sim_delta_t) 
                
        except: 

            in_exception += 1

            try:
                print('no exception', Ctrl_cmd[1].points[-1].accelerations[0].linear.z)
            except: 
                print('exception: did not get control commands')

            for i in range(n_agents):
                x0 = np.transpose(state[i,:,[t_now]])
                # print('shape: ', state[1,1,:].shape)
                u0 = np.zeros((4,1))
                

                x1 = sim_dyn(x0,u0,sim_delta)
                [X,Y,Z,W] = euler_to_quaternion(x1[6],x1[7],x1[8])
                odom.pose.pose.position.x = x1[0]
                odom.pose.pose.position.y = x1[1]
                odom.pose.pose.position.z = x1[2]
                odom.twist.twist.linear.x = x1[3]
                odom.twist.twist.linear.y = x1[4]
                odom.twist.twist.linear.z = x1[5]
                odom.pose.pose.orientation.x = X
                odom.pose.pose.orientation.y = Y
                odom.pose.pose.orientation.z = Z
                odom.pose.pose.orientation.w = W           
                odom.twist.twist.angular.x = x1[9] 
                odom.twist.twist.angular.y = x1[10] 
                odom.twist.twist.angular.z = x1[11] 
                

                pub_list[i].publish(odom)
            print('waiting...')
            time.sleep(0.5)

            if(in_exception > 20):
                
                break

    
    im_done.value = 1
    # print('final state for agent 1: ')
    # print(state[0,:,-1])





if __name__ == '__main__':
    try:
        start_sim()
    except rospy.ROSInterruptException:
        pass