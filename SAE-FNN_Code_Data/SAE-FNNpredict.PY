from keras.models import Model, Sequential, save_model, load_model
import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import gzip
import pandas as pd
import matplotlib.image as mpimg
import glob,os
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from keras.layers.advanced_activations import LeakyReLU, PReLU
def rmse(predictions, targets): #RMSE function for model evaluation
    return np.sqrt(((predictions - targets) ** 2).mean())

##########################################################################################################
#Parameters
##########################################################################################################
FNNmodelpath="logs/trained_SAE-FNNmodels/SAE-FNN10/FNN_model"                          #SAE-FNN model path. (Please change SAE-FNN10 to get different model)
FNNmodelweightpath="logs/trained_SAE-FNNmodels/SAE-FNN10/FNN_model.weights.hdf5"       #SAE-FNN model weights path. (Please change SAE-FNN10 to get different model)
meanspectrumdataset="data/Meanspectra.csv"                                             #192 Mean spectra in both calibration and testing set


###########################################################################################################
#Load Dataset.
###########################################################################################################
Meanspectrums = pd.read_csv(meanspectrumdataset,header=0)
x=np.array(Meanspectrums)[:,3:515]   # 512 bands of mean spectra
y=np.array(Meanspectrums)[:,2:3]     # N content label

#Divided into calibration and testing set
calibration_x=x[np.where(np.array(Meanspectrums)[:,1] == 1)] # Get 128 mean spectra,calibration_x, dataset value=1
calibration_y=y[np.where(np.array(Meanspectrums)[:,1] == 1)] # Get 128 N content, calibration_y 
testing_x=x[np.where(np.array(Meanspectrums)[:,1] == 0)]     # Get 64 mean spectra, testing_x, dataset value=0
testing_y=y[np.where(np.array(Meanspectrums)[:,1] == 0)]     # Get 64 N content,testing_y


###########################################################################################################
#Restore learned SAE-FNN model
###########################################################################################################
SAEs_FNNregression = load_model(FNNmodelpath)
SAEs_FNNregression.load_weights(FNNmodelweightpath)
SAEs_FNNregression.compile(optimizer='adam', loss='mean_squared_error')
SAEs_FNNregression.summary()



###########################################################################################################
#Predict in both calibration and tesing set
###########################################################################################################
#calibration set
Predictclass=SAEs_FNNregression.predict(calibration_x)
#save results
np.savetxt("results/SAE-FNN_calibration_results.csv", [calibration_y,Predictclass], delimiter=",")

rmse_val = rmse(calibration_y, Predictclass)      #RMSEC
r2_val = r2_score(calibration_y, Predictclass)    #R2C
RPDc=np.std(calibration_y)/rmse_val*1.0           #RPDC
print("calibration r2 is: " + str(r2_val))
print("calibration rms error is: " + str(rmse_val))
print("calibration RPDc is: " + str(RPDc))

#testing set
Predictclass=SAEs_FNNregression.predict(testing_x)
#save results
np.savetxt("results/SAE-FNN_testing_results.csv", [testing_y,Predictclass], delimiter=",")

rmse_val = rmse(testing_y, Predictclass)       #RMSEP
r2_val = r2_score(testing_y, Predictclass)     #R2P
RPDp=np.std(testing_y)/rmse_val*1.0            #RPDp

print(" prediction r2 is: " + str(r2_val))
print(" prediction rms error is: " + str(rmse_val))
print(" calibration RPDp is: " + str(RPDp))