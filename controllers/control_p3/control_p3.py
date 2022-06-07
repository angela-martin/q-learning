"""control_p3 controller."""
from controller import Robot
import time
import random

TIME_STEP = 32
CRUISE_SPEED = 16
GAMMA = 0.5
#ALPHA = 0.5

ultrasonic_sensors_names = ["left ultrasonic sensor", "front left ultrasonic sensor", "front ultrasonic sensor", "front right ultrasonic sensor",
  "right ultrasonic sensor"];
  
infrared_sensors_names = [
  # turret sensors
  "rear left infrared sensor", "left infrared sensor", "front left infrared sensor", "front infrared sensor",
  "front right infrared sensor", "right infrared sensor", "rear right infrared sensor", "rear infrared sensor",
  # ground sensors
  "ground left infrared sensor", "ground front left infrared sensor", "ground front right infrared sensor",
  "ground right infrared sensor"];

#################################################################################
# TESTS
#################################################################################

def test_actions():
    advance()
    turn_left()
    advance()
    advance()
    stop()

#################################################################################
# AUXILIARY FUNCTIONS 
#################################################################################

def avoid_wall():
    speed_offset = 0.2 * (CRUISE_SPEED - 0.03 * infrared_sensors[3].getValue());
    speed_delta = 0.03 * infrared_sensors[2].getValue() - 0.03 * infrared_sensors[4].getValue();
    
    leftMotor.setVelocity(speed_offset + speed_delta)
    rightMotor.setVelocity(speed_offset - speed_delta)  

#################################################################################
# MATRIX RELATED FUNCTIONS
#################################################################################

def print_matrix(matrix=[[]]):
    for i in range(len(matrix)):
        print(matrix[i])
        
"""
create rowsxcols matrix of zeros.
"""
def zeros_matrix(rows, cols):
    matrix = []

    while len(matrix) < rows:
        matrix.append([])
        while len(matrix[-1]) < cols:
            matrix[-1].append(0)
 
    return matrix
                
#################################################################################
# LIST OF POSSIBLE ACTIONS
# List of possible actions:
#    - Turn right: corresponds to position 0 in matrix Q
#    - Turn left: corresponds to position 1 in matrix Q
#    - Advance straight: corresponds to position 2 in matrix Q
#################################################################################

"""
Basic action.
"""
def do_action(leftSpeed=0, rightSpeed=0):
    aux = 0
    
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed) 
    
    while aux < 1:
        aux += 1
        robot.step(TIME_STEP) 

"""
Stop motors.
"""   
def stop():
    do_action(0, 0)    
    
"""
Turn to the right.
"""   
def turn_right():
    do_action(CRUISE_SPEED*2, 0) 
    return 0
    
"""
Turn to the left.
"""
def turn_left():   
    do_action(0, CRUISE_SPEED*2)     
    return 1
    
"""
Advance.
"""
def advance():
    do_action(CRUISE_SPEED, CRUISE_SPEED)     
    return 2
    
"""
Choose a random action to realize.
"""
def choose_action():
    action = 0
    aux = random.randint(0,2)
    
    if aux == 0:
        action = turn_right()
    elif aux == 1:
        action = turn_left()
    elif aux == 2:
        action = advance()
        
    return action

"""
Choose a specific action to realize.
"""    
def do_specific_action(action):
    if action == 0:
        turn_right()
    elif action == 1:
        turn_left()
    elif action == 2:
        advance()

#################################################################################
# REINFORCEMENT MANAGEMENT
#################################################################################

"""
Returns an array indicating the number of sensors that detect black.
"""  
def caculate_value_sensors():
    array = [0,0,0,0]
    
    for i in range(8,12): 
        if infrared_sensors[i].getValue() < 500:
            array[i-8] = 1
            
    return array 

"""
calculates the reinforcement value based on the values ​​detected by the sensors 
before and after performing an action.
""" 
def calculate_reinforcement(array_before, array_after):
    reinforcement = 0
    num_black_sensors = array_before.count(1) 
    num_black_sensors_after = array_after.count(1)
    
    diff = num_black_sensors_after - num_black_sensors

    # number of sensors detecting black is the same
    if diff == 0 and num_black_sensors_after > 2:
        reinforcement = 0.5
    elif diff == 0 and num_black_sensors_after < 3:
        reinforcement = 0.35
    # number of sensors detecting black is higher
    elif diff > 0 and num_black_sensors_after < 3:
        reinforcement = 0.5
    elif diff > 0 and num_black_sensors_after > 2:
        reinforcement = 0.75
    # number of sensors detecting black is lower
    elif diff < 0 and num_black_sensors_after > 3:
        reinforcement = -0.5
    elif diff < 0 and num_black_sensors_after < 2:
        reinforcement = -0.75
        
    if array_after[1] == [1,1,1,1]:
         reinforcement += 0.1
    
    return reinforcement
   
          
#################################################################################
# ROBOT STATE MANAGEMENT
# List of possible states:
#    - Robot leaves the line to the left: corresponds to position 0 in matrix Q
#    - Robot leaves the line to the right: corresponds to position 1 in matrix Q
#    - Other cases: corresponds to position 2 in matrix Q
#################################################################################

"""
Returns the current state of the robot.
"""
def get_state():
    # khepera leaves line to the left
    if infrared_sensors[9].getValue() > 750 and infrared_sensors[11].getValue() < 500:
        return 0
    # khepera leaves line to the right
    elif infrared_sensors[10].getValue() > 750 and infrared_sensors[8].getValue() < 500: 
        return 1
    # other cases 
    else: 
        return 2 

#################################################################################
# GET MOST APPROPIATE ACTION
#################################################################################        

"""
Returns the largest value in array Q that corresponds to a given state.
"""
def max_action_value(state, array):
    action_value = -10
    action = -10

    for i in range(len(array)):
        value = array[state][i]
        if value > action_value:
            action_value = value
            action = i
    
    return action_value

"""
Returns the action with the largest value in array Q that corresponds to a given state.
"""    
def max_action(state, array):
    action_value = -10
    action = -10

    for i in range(len(array)):
        value = array[state][i]
        if value > action_value:
            action_value = value
            action = i
    
    return action
    
#################################################################################
# Q-LEARNING
#################################################################################

def q_learning(qarray, qarrayv):    
    sensors_values = [0,0,0,0]
    sensors_values_after = [0,0,0,0]
    
    # save s and a values (state = s from Q(s,a), action = a from Q(s,a))
    state = get_state() 
    sensors_value = caculate_value_sensors()
    action = choose_action() 
    
    ALPHA = 1 / (1 + qarrayv[state][action])
    
    # save s' (state_after = s' from Q(s',a'))
    state_after = get_state() 
    sensors_value_after = caculate_value_sensors()
    
    # calculate reinforcement
    r = calculate_reinforcement(sensors_value, sensors_value_after) 
    
    QSA = qarray[state][action] # (Q(s,a)) 
    QMAXSA = max_action_value(state_after, qarray) # (max a' Q(s',a'))
    
    qarray[state][action] = (1 - ALPHA) * QSA + ALPHA * (r + (GAMMA * QMAXSA)) 
    qarrayv[state][action] += 1
    
    return qarray, qarrayv
    
def q_learning_choose_action(qarray, qarrayv):
    sensors_values = [0,0,0,0]
    sensors_values_after = [0,0,0,0]
    
    # save s and a values (state = s from Q(s,a), action = a from Q(s,a))
    state = get_state() #
    sensors_value = caculate_value_sensors()
    action = max_action(state, qarray) 
    do_specific_action(action)
    
    ALPHA = 1 / (1 + qarrayv[state][action])
    
    # save s' (state_after = s' from Q(s',a'))
    state_after = get_state()
    sensors_value_after = caculate_value_sensors()
    
    # calculate reinforcement
    r = calculate_reinforcement(sensors_value, sensors_value_after) 
    
    QSA = qarray[state][action] # (Q(s,a)) 
    QMAXSA = max_action_value(state_after, qarray) # (max a' Q(s',a'))
    
    qarray[state][action] = (1 - ALPHA) * QSA + ALPHA * (r + (GAMMA * QMAXSA)) 
    qarrayv[state][action] += 1
    
    return qarray, qarrayv   
 
#################################################################################
# INICIALIZATION AND ENABLE OF DEVICES
#################################################################################     

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
    
# get and enable infrared sensors
infrared_sensors = ['','','','','','','','','','','','']
for aux in range(len(infrared_sensors_names)):
    infrared_sensors[aux] = robot.getDevice(infrared_sensors_names[aux])
    infrared_sensors[aux].enable(TIME_STEP)

# get the motors and set target position to infinity (speed control)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

leftMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setPosition(float('inf'))
rightMotor.setVelocity(0.0)

# matrix Q
Q = zeros_matrix(3,3)
Q_visits = zeros_matrix(3,3)

# number of iteration
iterations = 0

#################################################################################
# MAIN PROGRAM
#################################################################################

while robot.step(timestep) != -1:

    wall = False
    aux_it = random.randint(0,10)

    # Check and store if there is a wall 
    for i in range(1,6): 
        if infrared_sensors[i].getValue() > 270:
            wall = True
        
    # If a wall is detected and the black line is not being followed
    if wall and caculate_value_sensors().count(1) < 2:
        avoid_wall() 
    # If there is no wall       
    else:
        # refining matrix Q
        if iterations < 600:           
            if iterations < 100:
                Q, Q_visits = q_learning(Q, Q_visits)
            # random action 90 percent of the time
            elif 100 <= iterations < 200:
                if aux_it > 1:
                    Q, Q_visits = q_learning(Q, Q_visits)
                else:
                    Q, Q_visits = q_learning_choose_action(Q, Q_visits)
            # random action 80 percent of the time
            elif 200 <= iterations < 300:
                if aux_it > 2:
                    Q, Q_visits = q_learning(Q, Q_visits)
                else:    
                    Q, Q_visits = q_learning_choose_action(Q, Q_visits)   
            # random action 60 percent of the time     
            elif 300 <= iterations < 400:
                if aux_it > 4:
                    Q, Q_visits = q_learning(Q, Q_visits)
                else:     
                    Q, Q_visits = q_learning_choose_action(Q, Q_visits) 
            # random action 40 percent of the time    
            elif 400 <= iterations < 500:
                if aux_it > 6:
                    Q, Q_visits = q_learning(Q, Q_visits)
                else:    
                    Q, Q_visits = q_learning_choose_action(Q, Q_visits)  
            # random action 20 percent of the time
            else:
                if aux_it > 8:
                    Q, Q_visits = q_learning(Q, Q_visits)
                else:    
                    Q, Q_visits = q_learning_choose_action(Q, Q_visits)       
                    
            iterations += 1
        # matrix Q already refined
        else:
            if iterations == 600:
                print('Matriz Q:')
                print_matrix(Q)
                print('Matriz de visitas a Q:')
                print_matrix(Q_visits)
                iterations += 1
                
            state = get_state()
            do_specific_action(max_action(state, Q))
                
