import keras
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
import sys
import cv2
import numpy as np
sys.path.append("game/")
import wrapped_flappy_bird as game
BETA = 0.01
const = 1e-5

#loss function for policy output
def logloss(y_true, y_pred):     #policy loss
    return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
    # BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image):
    image = cv2.cvtColor(cv2.resize(image, (84,84)), cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    #image = skimage.color.rgb2gray(image)
    #image = skimage.transform.resize(image, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')   
    #image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image
    

model = load_model("saved_models/model_updates7000", custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
game_state = game.GameState(30)

currentScore = 0
topScore = 0
a_t = [1,0]
FIRST_FRAME = True

terminal = False
r_t = 0
while True:
    if FIRST_FRAME:
        x_t = game_state.getCurrentFrame()
        x_t = preprocess(x_t)
        s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
        FIRST_FRAME = False     
    else:
        x_t, r_t, terminal = game_state.frame_step(a_t)
        x_t = preprocess(x_t)
        s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
    
    y = model.predict(s_t)
    no = np.random.random() 
    no = np.random.rand()
    a_t = [0,1] if no < y[0] else [1,0]    #stochastic policy
    #a_t = [0,1] if 0.5 <y[0] else [1,0]   #deterministic policy
    
    if(r_t == 1):
        currentScore += 1
        topScore = max(topScore, currentScore)
        print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))
    if terminal == True:
        FIRST_FRAME = True
        terminal = False
        currentScore = 0