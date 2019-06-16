from keras.layers import Input, Dense
from keras.models import Model, Sequential, save_model, load_model
from keras.callbacks import CSVLogger,ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import gzip
import pandas as pd
import matplotlib.ticker as ticker
from keras.layers.advanced_activations import LeakyReLU

###########################################################################################################
#Parameters
###########################################################################################################
encoding_dim = 10                                                 #deep spectral feature dimension.  5,10,20,30,40,50,60,70,80,90 were used in paper
input_dim = 512                                                   #input orginal pixel spectral band number
batch_size = 100                                                  #learning batch of model
nb_epoch = 100                                                    #learning epoch of model
SAEmodelpath="logs/SAEs_model"                                    #save SAE model file path
SAEmodelweightpath="logs/SAEs_model.weights.hdf5"                 #save SAE weight file path
SAEmodeltrainlogpath="logs/SAEs_model.train.log"                  #train log file path
FNNmodelpath="logs/trained_SAE-FNNmodels/SAE-FNN10/FNN_model"     #save SAE-FNN model file path
FNNmodelweightpath="logs/trained_SAE-FNNmodels/SAE-FNN10/FNN_model.weights.hdf5" #save SAE-FNN model weights file path
FNNmodeltrainlogpath="logs/FNNmodel.train.log"                    #train log file path
train_dataset="data/train_pixels/ramdonpixel_train.pkl.gz"        #train dataset 
validation_dataset="data/train_pixels/ramdonpixel_val.pkl.gz"     #validation dataset
meanspectrumdataset="data/Meanspectra.csv"                        #Mean spectra for each sample in both calibration and testing set

###########################################################################################################
#Load data.
###########################################################################################################
with gzip.open(train_dataset, 'rb') as f:
            train_set1, valid_set1 = pickle.load(f)
with gzip.open(validation_dataset, 'rb') as f:
            train_set2, valid_set2= pickle.load(f)
x_train=train_set1[0]  #orginal 51,264 pixel spectra with 512 bands
x_valid=train_set2[0]  #orginal 51,264 pixel spectra with 512 bands
y_train=train_set1[1]  # N content label of spectra
y_valid=train_set2[1]  # N content label of spectra


###########################################################################################################
#Build a SAE model with 512-220-100-encoding_dim-100-220-512
###########################################################################################################
SAEsmodel = Sequential()
SAEsmodel.add(Dense(220, input_dim=input_dim, kernel_initializer='normal', name="dense_1")) #encodeing layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function
SAEsmodel.add(Dense(100, kernel_initializer='normal', name="dense_2"))                      #encodeing layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function
SAEsmodel.add(Dense(encoding_dim, kernel_initializer='normal', name="dense_3"))             #this is the deep feature layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function
SAEsmodel.add(Dense(100, kernel_initializer='normal', name="dense_4"))                      #deconding layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function
SAEsmodel.add(Dense(220, kernel_initializer='normal', name="dense_5"))                      #deconding layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function
SAEsmodel.add(Dense(input_dim, kernel_initializer='normal',name="dense_6"))                 #deconding layer
SAEsmodel.add(LeakyReLU())                                                                      #add active function

#Pre-train SAE model
save_model(SAEsmodel,SAEmodelpath)                                       #Save model
checkpoint = ModelCheckpoint(filepath=SAEmodelweightpath, verbose=0)     #Save weights
csv_logger = CSVLogger(SAEmodeltrainlogpath,separator=',', append=False) #Save train history
SAEsmodel.compile(optimizer='Adam', loss='mean_squared_error')           #loss function MSE, optimizer adam
SAEsmodel.summary() #Show SAEs model struction

#Training
history = SAEsmodel.fit(x_train, x_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_valid, x_valid),
                callbacks=[checkpoint,csv_logger])

###########################################################################################################
#After pre-training SAE, we build SAE-FNN model:512-220-100-encoding_dim-1
###########################################################################################################
###########################################################################################################
#Rebuild SAE model without decoding layers. 
SAEs_FNNmodel = Sequential()
SAEs_FNNmodel.add(Dense(220, input_dim=input_dim, kernel_initializer='normal', name="dense_1"))   
SAEs_FNNmodel.add(LeakyReLU())
SAEs_FNNmodel.add(Dense(100, kernel_initializer='normal', name="dense_2"))
SAEs_FNNmodel.add(LeakyReLU())
SAEs_FNNmodel.add(Dense(encoding_dim, kernel_initializer='normal', name="dense_3")) #this is the deep feature layer
SAEs_FNNmodel.add(LeakyReLU())

#Add a 1 output fully-connected layer to the encoding layer of SAE without active function
SAEs_FNNmodel.add(Dense(1, kernel_initializer='normal', name="FNNlayer"))

#Restore the trained SAE model weights.The learned weights of SAE model will be used as initial weights for SAE-FNN
SAEs_FNNmodel.load_weights(SAEmodelweightpath,by_name=True) 

#build SAE-FNN model 
save_model(SAEs_FNNmodel,FNNmodelpath)
SAEs_FNNmodel.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

#save SAE-FNN model weights                                                        
checkpoint = ModelCheckpoint(filepath=FNNmodelweightpath, verbose=0)
csv_logger = CSVLogger(FNNmodeltrainlogpath,separator=',', append=False) #Save train history                     
SAEs_FNNmodel.summary()

#Load Dataset.
###########################################################################################################
Meanspectrums = pd.read_csv(meanspectrumdataset,header=0)
x=np.array(Meanspectrums)[:,3:515]   #512 bands of mean spectra
y=np.array(Meanspectrums)[:,2:3]     # N content label

#Divided into calibration and testing set
calibration_x=x[np.where(np.array(Meanspectrums)[:,1] == 1)] # Get 128 calibration_x, dataset value=1
calibration_y=y[np.where(np.array(Meanspectrums)[:,1] == 1)] # Get calibration_y , dataset value=1

#fine-tune the SAE-FNN model
history = SAEs_FNNmodel.fit(calibration_x, calibration_y,
                nb_epoch=11000,
                batch_size=100,
                shuffle=True,
                validation_data=(calibration_x, calibration_y),
                callbacks=[csv_logger])
loss, accuracy = SAEs_FNNmodel.evaluate(calibration_x, calibration_y)
print("finished training of SAE-FNN model, please use SAE-FNNpredict file to get prediction results")