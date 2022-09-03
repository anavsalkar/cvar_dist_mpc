#!/home/atharva/BTP/btp/bin/python

from dis import dis
from re import L
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
import copy


from quadprog import solve_qp
# from quad import dist_from_poly
# import quad


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

global Pred
Pred = []

def odometry_callback(*args):
    global Odom
    Odom = []
    n_agents = len(args)
    for i in range(n_agents):
        Odom.append(args[i])
    # print('got_callback')

# def pred_callback(*args):
#     global Pred
#     Pred = []
#     n_agents = len(args)
#     for i in range(n_agents):
#         Pred.append(args[i])
#     print('pred callback')

def pred_callback(args):
    # print('pred callback')
    if len(Pred) < n_agents:
        Pred.append(args)
    else: 
        id = int(args.header.frame_id)
        # print('id: ', id, ' pred length: ', len(Pred))
        Pred[id] = args
    # print('pred callback')

def get_state(n_agents,my_ID):
    # global OdomStruct
    # OdomStruct = []
    
    for id in range(n_agents):
        OdomData = Odom[id]
        W = OdomData.pose.pose.orientation.w
        X = OdomData.pose.pose.orientation.x
        Y = OdomData.pose.pose.orientation.y
        Z = OdomData.pose.pose.orientation.z

        phi,theta,psi = quaternion_to_euler_angle_vectorized1(W,X,Y,Z)
        phi,theta,psi = euler_from_quaternion(W,X,Y,Z)

        x0 = np.zeros((12,1))
        # print(x0.shape)
        pos_noise_std_dev = 0
        vel_noise_std_dev = 0

        if id!=my_ID:
            x_dist = abs(OdomData.pose.pose.position.x-Odom[my_ID].pose.pose.position.x)
            y_dist = abs(OdomData.pose.pose.position.y-Odom[my_ID].pose.pose.position.y)
            z_dist = abs(OdomData.pose.pose.position.z-Odom[my_ID].pose.pose.position.z)
             
            # if (x_dist**2 + y_dist**2  < 1.01**2):
            #     print('**** Collision! ', math.sqrt(x_dist**2 + y_dist**2 ))

            # print('dist: ', math.sqrt(x_dist**2 + y_dist**2 ))

            x0[0] = OdomData.pose.pose.position.x + np.random.normal(0,pos_noise_std_dev)
            x0[1] = OdomData.pose.pose.position.y + np.random.normal(0,pos_noise_std_dev)
            x0[2] = OdomData.pose. pose.position.z + np.random.normal(0,pos_noise_std_dev)
            x0[3] = OdomData.twist.twist.linear.x + np.random.normal(0,vel_noise_std_dev)
            x0[4] = OdomData.twist.twist.linear.y + np.random.normal(0,vel_noise_std_dev)
            x0[5] = OdomData.twist.twist.linear.z + np.random.normal(0,vel_noise_std_dev)
        else: 
            x0[0] = OdomData.pose.pose.position.x
            x0[1] = OdomData.pose.pose.position.y
            x0[2] = OdomData.pose.pose.position.z
            x0[3] = OdomData.twist.twist.linear.x
            x0[4] = OdomData.twist.twist.linear.y
            x0[5] = OdomData.twist.twist.linear.z

        x0[6] = euler_from_quaternion(W,X,Y,Z)[0]
        x0[7] = euler_from_quaternion(W,X,Y,Z)[1]
        x0[8] = euler_from_quaternion(W,X,Y,Z)[2]
        x0[9] = OdomData.twist.twist.angular.x
        x0[10] = OdomData.twist.twist.angular.y
        x0[11] = OdomData.twist.twist.angular.z
        settings.OdomStruct[id] = x0

        # settings.OdomStruct[id][0] = OdomData.pose.pose.position.x
        # settings.OdomStruct[id][1] = OdomData.pose.pose.position.y
        # settings.OdomStruct[id][2] = OdomData.pose.pose.position.z
        # settings.OdomStruct[id][3] = OdomData.twist.twist.linear.x
        # settings.OdomStruct[id][4] = OdomData.twist.twist.linear.y
        # settings.OdomStruct[id][5] = OdomData.twist.twist.linear.z
        # settings.OdomStruct[id][6] = euler_from_quaternion(W,X,Y,Z)[0]
        # settings.OdomStruct[id][7] = euler_from_quaternion(W,X,Y,Z)[1]
        # settings.OdomStruct[id][8] = euler_from_quaternion(W,X,Y,Z)[2]
        # settings.OdomStruct[id][9] = OdomData.twist.twist.angular.x
        # settings.OdomStruct[id][10] = OdomData.twist.twist.angular.y
        # settings.OdomStruct[id][11] = OdomData.twist.twist.angular.z
        # settings.OdomStruct[id] = x0 
        # settings.OdomStruct.append(x0)
    # print(OdomStruct)
    # x0 = np.ones((12,1))

    return x0

# numpy variable for storing the predictions
# list(number of agents), numpy array -> (21,12)



def get_predictions(n_agents,my_ID):
    n_horizon = 20

    const_vel = False
    delta_t = 0.1

    ## Normal Distribution
    pos_noise_std_dev = 0
    vel_noise_std_dev = 0

    x_noise_offset = np.random.normal(0,pos_noise_std_dev) 
    y_noise_offset = np.random.normal(0,pos_noise_std_dev)
    z_noise_offset = np.random.normal(0,pos_noise_std_dev)
    vx_noise_offset = np.random.normal(0,vel_noise_std_dev)
    vy_noise_offset = np.random.normal(0,vel_noise_std_dev)
    vz_noise_offset = np.random.normal(0,vel_noise_std_dev)



    # ## Beta Distribution
    # alpha = 2
    # beta = 3
    # position_error_max = 2
    # velocity_error_max = 1

    # x_noise_offset = position_error_max*(np.random.beta(2,3) - 0.5)
    # y_noise_offset = position_error_max*(np.random.beta(2,3) - 0.5)
    # z_noise_offset = position_error_max*(np.random.beta(2,3) - 0.5)
    # vx_noise_offset = velocity_error_max*(np.random.beta(1.5,3) - 0.5)
    # vy_noise_offset = velocity_error_max*(np.random.beta(1.5,3) - 0.5)
    # vz_noise_offset = velocity_error_max*(np.random.beta(1.5,3) - 0.5)


    for id in range(n_agents):
        # print('len(Pred): ', len(Pred), ' id: ', id)
        PredData = Pred[id]
        x_noise_offset = np.random.normal(0,pos_noise_std_dev) 
        y_noise_offset = np.random.normal(0,pos_noise_std_dev)
        z_noise_offset = np.random.normal(0,pos_noise_std_dev)
        vx_noise_offset = np.random.normal(0,vel_noise_std_dev)
        vy_noise_offset = np.random.normal(0,vel_noise_std_dev)
        vz_noise_offset = np.random.normal(0,vel_noise_std_dev)
        # print(PredData)

        for k in range(n_horizon+1):

            
            # if my_ID!=id:
            #     settings.PredStruct[id][0,k] = PredData.points[k].transforms[0].translation.x + np.random.normal(0,pos_noise_std_dev)
            #     settings.PredStruct[id][1,k] = PredData.points[k].transforms[0].translation.y + np.random.normal(0,pos_noise_std_dev)
            #     settings.PredStruct[id][2,k] = PredData.points[k].transforms[0].translation.z + np.random.normal(0,pos_noise_std_dev)
            #     settings.PredStruct[id][3,k] = PredData.points[k].velocities[0].linear.x + np.random.normal(0,vel_noise_std_dev)
            #     settings.PredStruct[id][4,k] = PredData.points[k].velocities[0].linear.y + np.random.normal(0,vel_noise_std_dev)
            #     settings.PredStruct[id][5,k] = PredData.points[k].velocities[0].linear.z + np.random.normal(0,vel_noise_std_dev)

            if my_ID!=id and const_vel==False:
                settings.PredStruct[id][0,k] = PredData.points[k].transforms[0].translation.x + x_noise_offset
                settings.PredStruct[id][1,k] = PredData.points[k].transforms[0].translation.y + y_noise_offset
                settings.PredStruct[id][2,k] = PredData.points[k].transforms[0].translation.z + z_noise_offset
                settings.PredStruct[id][3,k] = PredData.points[k].velocities[0].linear.x + vx_noise_offset
                settings.PredStruct[id][4,k] = PredData.points[k].velocities[0].linear.y + vy_noise_offset
                settings.PredStruct[id][5,k] = PredData.points[k].velocities[0].linear.z + vz_noise_offset
            
            elif my_ID!=id and const_vel==True:
                settings.PredStruct[id][0,k] = settings.OdomStruct[id][0,0]+k*delta_t*settings.OdomStruct[id][3,0]
                settings.PredStruct[id][1,k] = settings.OdomStruct[id][1,0]+k*delta_t*settings.OdomStruct[id][4,0]
                settings.PredStruct[id][2,k] = settings.OdomStruct[id][2,0]+k*delta_t*settings.OdomStruct[id][5,0]
                settings.PredStruct[id][3,k] = settings.OdomStruct[id][3,0]
                settings.PredStruct[id][4,k] = settings.OdomStruct[id][4,0]
                settings.PredStruct[id][5,k] = settings.OdomStruct[id][5,0]

            else:
                settings.PredStruct[id][0,k] = PredData.points[k].transforms[0].translation.x
                settings.PredStruct[id][1,k] = PredData.points[k].transforms[0].translation.y
                settings.PredStruct[id][2,k] = PredData.points[k].transforms[0].translation.z
                settings.PredStruct[id][3,k] = PredData.points[k].velocities[0].linear.x
                settings.PredStruct[id][4,k] = PredData.points[k].velocities[0].linear.y
                settings.PredStruct[id][5,k] = PredData.points[k].velocities[0].linear.z
            # single_agent_list[k] = z0
        # if len(settings.PredStruct) != 6:
        #     settings.PredStruct.append(single_agent_list)
        # else: 
        #     settings.PredStruct[id] = single_agent_list
        #     settings.PredStruct[id][k]
        # single_agent_list.clear()
    # print('predStruct size in gedPred: ', predS)
    return 0

# def store_predictions():




def get_G_and_H(x):
    G = np.zeros((6,3))
    G[0,0] = 1.
    G[1,0] = -1.
    G[2,1] = 1.
    G[3,1] = -1.
    G[4,2] = 1.
    G[5,2] = -1.

    X = x[0]
    Y = x[1]
    Z = x[2]

    # x_offset = 1
    # y_offset = 1
    # z_offset = 0.25
    x_offset = 1
    y_offset = 1
    z_offset = 7
    ## 0.75,0.75,0.5 for gazebo

    H = np.ones((6,1))
    H[0,0] = X+x_offset
    H[1,0] = -(X-x_offset)
    H[2,0] = Y+y_offset
    H[3,0] = -(Y-y_offset)
    H[4,0] = Z+z_offset
    H[5,0] = -(Z-z_offset)

    return G,H


def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 


def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def euler_to_quaternion(roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def setInitLambda(mpc, n_agents, my_ID):
    lambda0 = 0.15*np.ones((6,1))
    lambda0[0] = 0.3
    lambda0[2] = 0.3
    lambda0[4] = 0.3
    for i in (j for j in range(n_agents) if j!=my_ID):
        lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(i+1)
        mpc.u0[lambda_var_name] = lambda0
    return mpc


def setInitCvarLambda(mpc, n_agents, my_ID, n_samples):
    lambda0 = 0.15*np.ones((6,1))
    lambda0[0] = 0.3
    lambda0[2] = 0.3
    lambda0[4] = 0.3
    
    for j in (j for j in range(n_agents) if j!= my_ID):
        for n in range(n_samples):
            lambda_var_name = 'lambda' + str(my_ID+1) + '_for_agent' + str(j+1) + '_for_sample' + str(n)
            mpc.u0[lambda_var_name] = lambda0
    return mpc

def predDataToMsg(pred_data,seq,my_ID):
    pred_msg = MultiDOFJointTrajectory()
    n_horizon = 20
    pred_msg.header.frame_id = str(my_ID)
    pred_msg.header.seq = seq
    pred_msg.header.stamp = rospy.Time.now()
    for k in range(n_horizon+1):
        transforms =Transform()
        velocities =Twist()
        accelerations=Twist()
        transforms.translation.x = pred_data[0,k,0]
        transforms.translation.y = pred_data[1,k,0]
        transforms.translation.z = pred_data[2,k,0]
        velocities.linear.x = pred_data[3,k,0]
        velocities.linear.y = pred_data[4,k,0]
        velocities.linear.z = pred_data[5,k,0]
        pred_point = MultiDOFJointTrajectoryPoint([transforms],[velocities],[accelerations],rospy.Time.now())
        pred_msg.points.append(pred_point)
    return pred_msg
    
def dist_from_poly(G,H,p):
    qp_G = np.eye(3)
    qp_a = p.reshape((3,))
    qp_C = -G.T
    qp_b = -H.reshape((6,))
    meq = 0
    sol = solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
    dist = np.linalg.norm(sol-p)
    return dist

def quad(i, my_ID):
    G = np.zeros((6,3))
    G[0,0] = 1.
    G[1,0] = -1.
    G[2,1] = 1.
    G[3,1] = -1.
    G[4,2] = 1.
    G[5,2] = -1.

    X = settings.OdomStruct[i][0,0]
    Y = settings.OdomStruct[i][1,0]
    Z = settings.OdomStruct[i][2,0]

    x_offset = 1.5
    y_offset = 1.5
    z_offset = 7

    H = np.ones((6,1))
    H[0,0] = X+x_offset
    H[1,0] = -(X-x_offset)
    H[2,0] = Y+y_offset
    H[3,0] = -(Y-y_offset)
    H[4,0] = Z+z_offset
    H[5,0] = -(Z-z_offset)
    # H = np.array([1.86314506,  0.13685494 , 1.5533926,   0.4466074 , 11.44289717,  2.55710283])
    # H = np.array([[ 1.74318264],
    #     [ 0.25681736],
    #     [ 1.42388943],
    #     [ 0.57611057],
    #     [11.55039395],
    #     [ 2.44960605]])
    # p = np.array([settings.OdomStruct[my_ID][0],settings.OdomStruct[my_ID][0],settings.OdomStruct[my_ID][0]])
    # print(np.array([settings.OdomStruct[my_ID][0],settings.OdomStruct[my_ID][1],settings.OdomStruct[my_ID][1]]))
    # print('my_ID: ', my_ID)
    # print('other_ID: ', i)
    
    # print('my_X: ', settings.OdomStruct[my_ID][0,0])
    # print('my_Y: ', settings.OdomStruct[my_ID][1,0])
    # print('my_Z: ', settings.OdomStruct[my_ID][2,0])

    my_X = settings.OdomStruct[my_ID][0,0]
    my_Y = settings.OdomStruct[my_ID][1,0]
    my_Z = settings.OdomStruct[my_ID][2,0]


    start = time.time()
    # print('dist: ',dist_from_poly(G,H,np.array([0., 0., 0.])))
    # print('dist: ',dist_from_poly(G,H,np.array([my_X, my_Y, my_Z])))
    # print(dist_from_poly(G,H,p))
    end = time.time()
    # print('time taken: ', end-start)
    
    # print('\n')


        


def mpc(my_ID, n_agents_ip, final_pos, is_collision, im_done, sim_id, dir_name, sol_array):
    # global OdomStruct
    # OdomStruct = []
    n_samples = 10
    global n_agents
    n_agents = n_agents_ip
    is_collision.value = 0
    n_horizon = 20
    d_min = 0.01

    settings.init()
    for id in range(n_agents):
        x0 = np.ones((12,1))
        x0 = (id+1)*x0
        settings.OdomStruct.append(x0)
        settings.PredStruct.append(np.zeros((6,n_horizon+1)))
        # p0 = np.ones((12,21,1))
        # p0 = (id+1)*p0
        # settings.PredStruct.append(p0)


    rospy.init_node('talker'+str(my_ID), anonymous=True)
    sub_list = []
    for i in  range(n_agents):
        topic = '/firefly'+str(i+1)+'/odometry_sensor1/odometry'
        sub = message_filters.Subscriber(topic, Odometry)
        sub_list.append(sub)

    ts = message_filters.TimeSynchronizer(sub_list, 20)
    ts.registerCallback(odometry_callback)

    # """ multiple subscribers for predictions """
    # pred_sub_list = []
    # for i in  range(n_agents):
    #     topic = '/firefly'+str(i+1)+'/prediction'
    #     pred_sub = message_filters.Subscriber(topic, MultiDOFJointTrajectory)
    #     pred_sub_list.append(pred_sub)

    # pred_ts = message_filters.TimeSynchronizer(pred_sub_list, 10)
    # pred_ts.registerCallback(pred_callback)

    for i in  range(n_agents):
        topic = "/firefly"+str(i+1)+"/prediction"
        rospy.Subscriber(topic, MultiDOFJointTrajectory, pred_callback)
    # rospy.Subscriber("/firefly5/prediction", MultiDOFJointTrajectory, pred_callback)

    # rospy.Subscriber('/firefly'+str(my_ID+1)+'/odometry_sensor1/odometry', Odometry, odometry_callback)
    
    """ User settings: """
    show_animation = False
    store_results = True

    """
    Get configured do-mpc modules:
    """
    time.sleep(2)
    x0 = settings.OdomStruct[my_ID]
    # print(x0)
    time.sleep(2)
    model = template_model(my_ID, n_agents)
    mpc = template_mpc(model, my_ID, n_agents, final_pos)
    # simulator = template_simulator(model,my_ID, n_agents, final_pos, OdomStruct)
    # estimator = do_mpc.estimator.StateFeedback(model)



    """
    Set initial state
    """
    np.random.seed(99)

    e = np.zeros([model.n_x,1])
    #e[2] = 5
    x0 = e # Values between +3 and +3 for all states

    # time.sleep(5)
    x0 = settings.OdomStruct[my_ID]
    mpc.x0 = x0
    # simulator.x0 = x0
    # estimator.x0 = x0
    u0 = 0*np.ones((4,1))
    mpc.u0['u'] = u0
    # simulator.u0['u'] = u0
    # estimator.u0['u'] = u0

    # initialise the lambda dual variable
    mpc = setInitLambda(mpc, n_agents, my_ID)
    mpc = setInitCvarLambda(mpc, n_agents, my_ID, n_samples)
    
    
    # Use initial state to set the initial guess.
    mpc.set_initial_guess()


    """
    ROS Subscriber and Publisher 
    """
    
    rate = rospy.Rate(50) # 10hz
    pub_topic = '/firefly' + str(my_ID+1) + '/command/trajectory'
    firefly_command_publisher = rospy.Publisher(pub_topic, MultiDOFJointTrajectory, queue_size=10)
    #rospy.Subscriber('/firefly/odometry_sensor1/odometry', Odometry, odometry_callback)


    pred_pub_topic = '/firefly' + str(my_ID+1) + '/prediction'
    prediction_publisher = rospy.Publisher(pred_pub_topic, MultiDOFJointTrajectory, queue_size=10)


    traj = MultiDOFJointTrajectory()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time()
    header.frame_id = 'frame'
    traj.joint_names.append('base_link')
    traj.header=header
    transforms =Transform()
    velocities =Twist()
    accelerations=Twist()
    
    logT = 150

    x_log = np.zeros((4,logT))


    """
    Log Prediction Data for each time step
    """
    common_pred_log = [] ## Array of n_agents (6) arrays
    for i in  range(n_agents):
        common_pred_log.append([])

    

    """
    Error Log Array
    """
    # x_pred_error_log = []
    # for i in range(n_horizon):
    #     x_pred_error_log.append([])
    
    # for i in range(n_agents):
    #     settings.ErrorLog.append(x_pred_error_log)

    rows, cols = (n_agents,n_horizon)
    settings.ErrorLog = [[[] for i in range(cols)] for j in range(rows)]

    time_log = []
    dist_log = []
    # dist_int = np.zeros(3)
    dist_int = 0
        
    """
    Run MPC ROS Loop
    """
    k = 0
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    while not rospy.is_shutdown():
        

        ### wait till I get the new state
        while sol_array[-1].value == 0:
            rospy.sleep(0.01) 
        sol_array[my_ID].value = 0
        
        # print(k)

        # unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # rospy.sleep(0.01)
        # pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        
        #print(OdomData)
        get_state(n_agents,my_ID)
        OdomData = settings.OdomStruct[my_ID]

        

        
        # print(str(my_ID),OdomData)
        start_time = time.time()

        x0[0] = OdomData[0]
        x0[1] = OdomData[1] 
        x0[2] = OdomData[2] 
        x0[3] = OdomData[3] 
        x0[4] = OdomData[4] 
        x0[5] = OdomData[5] 
        x0[6] = OdomData[6] 
        x0[7] = OdomData[7] 
        x0[8] = OdomData[8] 
        x0[9] = OdomData[9] 
        x0[10] = OdomData[10] 
        x0[11] = OdomData[11] 

        # print('current state 0 is:',x0[0])

        # x0[0] = OdomData[0] + np.random.normal(0,0.1)
        # x0[1] = OdomData[1] + np.random.normal(0,0.1)
        # x0[2] = OdomData[2] + np.random.normal(0,0.1)
        # x0[3] = OdomData[3] + np.random.normal(0,0.1)
        # x0[4] = OdomData[4] + np.random.normal(0,0.1)
        # x0[5] = OdomData[5] + np.random.normal(0,0.1)
        # x0[6] = OdomData[6] + np.random.normal(0,0.01)
        # x0[7] = OdomData[7] + np.random.normal(0,0.01)
        # x0[8] = OdomData[8] + np.random.normal(0,0.01)
        # x0[9] = OdomData[9] + np.random.normal(0,0.01)
        # x0[10] = OdomData[10] + np.random.normal(0,0.01)
        # x0[11] = OdomData[11] + np.random.normal(0,0.01)
        
        # W = OdomData.pose.pose.orientation.w
        # X = OdomData.pose.pose.orientation.x
        # Y = OdomData.pose.pose.orientation.y
        # Z = OdomData.pose.pose.orientation.z
        
        # phi,theta,psi = quaternion_to_euler_angle_vectorized1(W,X,Y,Z)
        # phi,theta,psi = euler_from_quaternion(W,X,Y,Z)
        
        
        # x0[0] = OdomData.pose.pose.position.x
        # x0[1] = OdomData.pose.pose.position.y
        # x0[2] = OdomData.pose.pose.position.z
        # x0[3] = OdomData.twist.twist.linear.x

        # x0[11] = OdomData.twist.twist.angular.z
        # print('x'+str(my_ID))
        


        ## prediction data storage ##
        if k > 10:
            for i in range(n_agents):
                common_pred_log[i].append(copy.deepcopy(settings.PredStruct[i]))
                # if my_ID==0 and i != my_ID:
                #     print('state of', i, ' is ',settings.OdomStruct[i].T)
                #     print(settings.PredStruct[i])

        ## prediction error log            
        if k > 10+n_horizon:
            for l in range(n_horizon):
                for i in range(n_agents):
                    error = settings.OdomStruct[i][0:6,0] - common_pred_log[i][-1-l][:,l+1] ####today

                    # error = settings.OdomStruct[i][0:6,0] - common_pred_log[i][-1-l][:,1+l]
                    # error = common_pred_log[i][-1-l][:,1+l]
                    # print('error: ', str(l),' ', str(i),' ', str(my_ID),' ', error)
                    # print('l = ', l, ' i = ', i, ' e =  ',  error[0:3])
                    # if my_ID==0 and i!=my_ID:
                    #     print('l = ', l, ' i = ', i, ' e =  ',  error[0:3].T)
                        # print('p = ',common_pred_log[i][-1-l][:,l+1])
                        # if l==4:
                        #     print(common_pred_log[i][-10])
                    settings.ErrorLog[i][l].append(copy.deepcopy(error[0:3]))
            #         if l == 5 and my_ID==0:
            #             print('agent is ', i, ' and error is ',settings.ErrorLog[i][5][-1][0],' ',settings.ErrorLog[0][5][-1][0])
            # if my_ID == 0:
            #     print('0: ',settings.ErrorLog[0][5][-1][0],' 1: ',settings.ErrorLog[1][5][-1][0])


        # if my_ID == 0:
        #     print(x0.T)
    
        
        # if k == 0:
        #     for k_ in range(21):
        #         settings.PredStruct[my_ID][:,k_,0] = x0[:,0]
        
        # rospy.sleep(0.7)

        u0 = mpc.make_step(x0)
        
        
        # print('u'+str(my_ID))
        # print(u0.T)
        
        mpc.x0 = x0
        # simulator.x0 = x0
        # estimator.x0 = x0
        mpc.u0 = u0
        # simulator.u0 = u0
        # estimator.u0 = u0
        mpc.set_initial_guess()
        pred_data = mpc.data.prediction(('_x','x'))
        # print('pred_data shape is: ',pred_data.shape)
        # print('pred_data 0,0,0 is: ',pred_data[0,0,0])
        
    
        # print('**********')
        # settings.PredStruct[my_ID] = pred_data

        pred_msg = predDataToMsg(pred_data, k, my_ID)
        prediction_publisher.publish(pred_msg)

        for i in (j for j in range(n_agents) if j!= my_ID):

            """ COLLISION CHECK """
            pos_i = np.array([Odom[i].pose.pose.position.x,Odom[i].pose.pose.position.y,Odom[i].pose.pose.position.z])
            (qp_G,qp_H) = get_G_and_H(pos_i)
            p = np.array([Odom[my_ID].pose.pose.position.x,Odom[my_ID].pose.pose.position.y,Odom[my_ID].pose.pose.position.z])
            # p = np.array([settings.OdomStruct[my_ID][0,0],settings.OdomStruct[my_ID][1,0],settings.OdomStruct[my_ID][2,0]])
            dist = dist_from_poly(qp_G,qp_H,p)
            transforms.translation.z = dist
            dist_log.append(dist)
            print('dist: ',dist)


            # quad(i,my_ID)
            # if(my_ID==0):
                # print('dist: ',dist)
                # print('pos: ', p)
                # print('qp_H: ', qp_H)
                # print('qp_G: ', qp_G)

            
            if dist < d_min:
                print(str(my_ID)+'***** COLLISION with '+str(i)+'dist: ',dist)
                is_collision.value = 1
                im_done.value = 1
                sol_array[my_ID].value = 1
                break
                
                

        




        # if my_ID == 4 and k>12:
        #     # print('predData', settings.PredStruct[my_ID][0,5,0])
        #     # print('predData@0: ', settings.PredStruct[my_ID][1,0,0])
        #     # print('actualData: ', settings.OdomStruct[my_ID][1,0])
        #     print('predData for id5: ', settings.PredStruct[5][0,15]) # agent_id, state(row), time(column) 
        #     # print('predStruct shape: ', len(settings.PredStruct[5]))
        # if my_ID == 5 and k>12:
        #     print('odomData for id5: ', settings.OdomStruct[my_ID][0,0]) # state
    
        # which simualation
        # y_next = simulator.make_step(u0)
        # x0 = estimator.make_step(y_next)
        
            
        # try:
        #     get_predictions(n_agents)
        # except: 
        #     print('did not get the predictions')
        # # point = MultiDOFJointTrajectoryPoint()

        if len(Pred) == n_agents:
            # print('Pred shape: ', len(Pred))
            get_predictions(n_agents,my_ID)
            # store_predictions()
        
        accelerations.linear.x = 0
        accelerations.linear.y = 0
        accelerations.linear.z = u0[0]
        
        accelerations.angular.x = u0[1]
        accelerations.angular.y = u0[2]
        accelerations.angular.z = u0[3]
        
        point = MultiDOFJointTrajectoryPoint([transforms],[velocities],[accelerations],rospy.Time.now())
        # print('id: ', my_ID, ' u0: ', u0)

        # print('my_ID @', k)

        traj.points.append(point)
        firefly_command_publisher.publish(traj) 

        


        x_log[0,k] = x0[0]
        x_log[1,k] = x0[1]
        x_log[2,k] = x0[2]
        #rate.sleep()

        # if k == 10:
        #     time.sleep(10)

        

        # if my_ID == 4:
        #     print(model.tvp['H25',1])
        
   
        end_time = time.time()
        time_taken = (end_time - start_time)
        time_log.append(time_taken)
        # print("time_taken by ", my_ID,": ",time_taken)
        x_log[3,k] = time_taken
        print("time_taken: ",time_taken," by my id: ", my_ID, " @sim id: ", sim_id)
        # print(x0)
        # if(my_ID==0):
        #     print("time_taken: ",time_taken," by my id: ", my_ID, " @sim id: ", sim_id)
        #rate.sleep()
        #rospy.sleep()
        # if time_taken < 0.2:
        #     time.sleep(0.2-time_taken)


        # dist_int += (abs(np.array(final_pos)-np.array([x0[0,0],x0[1,0],x0[2,0]])))*time_taken
        # dist_int += np.linalg.norm(np.array(final_pos)-np.array([x0[0,0],x0[1,0],x0[2,0]]))*time_taken
        dist_int += np.linalg.norm(np.array(final_pos)-np.array(copy.deepcopy([x0[0,0],x0[1,0],x0[2,0]])))*0.1  ## for sync case
        # print(dist_int)


        k = k+1
        if k == logT or is_collision.value == 1 or im_done.value == 1:
            # print('mean time: ', np.mean(x_log[3,:]))

            error_array = np.array(settings.ErrorLog)
            time_array = np.array(time_log)
            dist_array = np.array(dist_log)

            dict = {"error":error_array, "time":time_array, "dist":dist_array, "is_collision":is_collision.value, "dist_int":dist_int}
            np.save(dir_name+'/log_of_'+str(my_ID)+'_@'+str(sim_id)+'.npy',dict)
            np.save('rep_traj2'+str(my_ID)+'.npy',x_log)
            # np.save('errorlog_'+str(my_ID), error_array)

            # np.save('sim_id'+str(sim_id)+'_x_log'+str(my_ID)+'_al_0.9.npy',x_log)

            print('dist min:', dist_array.min(), ' std: ',dist_array.std())

            print('mean time: ', np.mean(time_array))
            print('max. time: ', np.max(time_array))
            print('std. time: ', np.std(time_array))

            # if my_ID==1:
            #     for l in range(n_horizon):
            #         print('mean of 1 is ',error_array[0,l,:,1].mean())
            #         # print('std of 1 is ',error_array[1,l,:,1].std())

            # np.save('rpi_pc'+str(my_ID), x_log)
            # if my_ID == 1:
            #     np.save('x_hist_3_from_1_dummy', np.array(settings.ErrorLog[3]))
            # sol_array[my_ID].value = 1
            # im_done.value == 1
            rospy.signal_shutdown('no reason')
            break
        


        ############ indicate that MPC has found solution 
        sol_array[my_ID].value = 1
        while sol_array[-2].value == 0 and k > 20:
            rospy.sleep(0.01) 
        
        

            
    
    # Store results:
    # if store_results:
    #     do_mpc.data.save_results([mpc, simulator], 'oscillating_masses')
    
    # np.save('tri'+str(my_ID), x_log)

if __name__ == '__main__':
    try:
        #rospy.Subscriber('/firefly/odometry_sensor1/odometry', Odometry, odometry_callback)
        mpc(0,6,[4,-6,3],0)
    except rospy.ROSInterruptException:
        pass
