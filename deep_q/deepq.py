from __future__ import print_function

import argparse
import numpy.core.multiarray
import cv2
#import skimage as skimage
#from skimage import transform, color, exposure
#from skimage.transform import rotate
#from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializers
#from tensorflow.keras import initializers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras
from keras.layers import  Input
from keras.models import  Model

#initializer = initializers.TruncatedNormal(mean=0, stddev=0.01)

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 500000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-6
CONTINUE_TRAIN = True

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Building Model")
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    #model = Sequential()
    #keras.initializers.RandomNormal(mean=0, stddev=0.01)

    #S = Input(shape = (img_rows,img_cols,img_channels, ), name = 'Input')
    #h0 = Convolution2D(32, kernel_size = (8,8), strides = (4,4), activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'random_normal')(S)
    #h1 = Convolution2D(64, kernel_size = (4,4), strides = (2,2), activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'random_normal')(h0)
    #h11 = Convolution2D(64, kernel_size = (3,3), strides = (1,1), activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'random_normal')(h1)
    #h2 = Flatten()(h11)
    #h3 = Dense(512, activation = 'relu', kernel_initializer = 'random_normal', bias_initializer = 'random_normal') (h2)
    #O_p = Dense(2, name = 'o_P', activation = 'sigmoid', kernel_initializer = 'random_normal', bias_initializer = 'random_normal') (h3)
    

    #model = Model(inputs = S, outputs = [O_p])
    
   
    adam = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("Model is built")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState(30)

    # Make a double ended queue to store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    #x_t = skimage.color.rgb2gray(x_t)
    #x_t = skimage.transform.resize(x_t,(80,80))
    #x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    #x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    while (True):
        #We reduced the epsilon gradually
        if not args['mode'] == 'Run':
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        
        

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        #x_t1 = skimage.color.rgb2gray(x_t1_colored)
        #x_t1 = skimage.transform.resize(x_t1,(80,80))
        #x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)

        #x_t1 = x_t1 / 255.0


        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            minibatch = np.array(minibatch)
            state_t1 = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            state_t1 = np.vstack(minibatch[:,3])
            action_array_t = np.zeros((BATCH,ACTIONS))
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  
            Q_sa = model.predict(state_t1)


            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]

                action_array_t[i,action_t] = 1
                inputs[i:i + 1] = state_t
                
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa[i])

            #Now we do the experience replay
            #state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            #state_t = np.concatenate(state_t)
            #state_t1 = np.concatenate(state_t1)
            #targets = model.predict(state_t)
            #Q_sa = model.predict(state_t1)
            #targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
        f = open("rewards.txt","a")
        f.write("TIMESTEP: " + str(t) + ", Q_MAX " + str(np.max(Q_sa)) + ", EPSILON: " + str(epsilon) + ", REWARD: " + str(r_t) + ", Loss: " + str(loss)  + "\n")
        f.close()

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()