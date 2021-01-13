#%%
import tensorflow as tf
import pandas as pd 
import numpy as np

# 調整weights相對大小
#%%
weights = dict(
    input_to_v1 = 3,
    v1_to_spat1 = 0.6,
    spat1_to_v1 = 0.4,
    spat1_to_spat2 = 1,
    spat2_to_spat1 = 0.4,
    v1_to_obj1 = 1,
    obj1_to_v1 = 0.25,
    obj1_to_obj2 = 1,
    obj2_to_obj1 = 0.25,
    spat1_to_obj1 = 2,
    obj1_to_spat1 = 0.5,
    spat2_to_obj2 = 2,
    obj2_to_spat2 = 0.5,
    obj2_to_output = 1,
    output_to_obj2 = 0.25,
    spat1_lateral_inhibit = 1,
    spat2_lateral_inhibit = 1
)

# 調整各layer之node數目、形狀(?x?)、和各layer間的連結方式 
#%%
class Connections():
    def __init__(self, wt):
        self.wt = wt
        
        self.weights = {k:v*self.wt for k, v in weights.items()}
        self.weights

#%%
        self.input_to_v1 = np.identity(7)*self.weights["input_to_v1"]
        self.input_to_v1

#%%
        self.v1_to_spat1 = np.array([[1,0,0,0,0],
                                [1,1,0,0,0],
                                [1,1,1,0,0],
                                [0,1,1,1,0],
                                [0,0,1,1,1],
                                [0,0,0,1,1],
                                [0,0,0,0,1]])*self.weights["v1_to_spat1"]
        self.v1_to_spat1

        self.spat1_to_v1 = np.transpose(self.v1_to_spat1)*self.weights["spat1_to_v1"]

#%%
        self.spat1_to_spat2 = np.array([[1,0,0],
                                [1,1,0,],
                                [1,1,1],
                                [0,1,1],
                                [0,0,1]])*self.weights["spat1_to_spat2"]
        self.spat1_to_spat2

        self.spat2_to_spat1 = np.transpose(self.spat1_to_spat2)*self.weights["spat2_to_spat1"]

#%%
        self.v1_to_obj1 = np.array([[1,0,0,0,0],
                                [1,1,0,0,0],
                                [1,1,1,0,0],
                                [0,1,1,1,0],
                                [0,0,1,1,1],
                                [0,0,0,1,1],
                                [0,0,0,0,1]])*self.weights["v1_to_obj1"]
        self.v1_to_obj1

        self.obj1_to_v1 = np.transpose(self.v1_to_obj1)*self.weights["obj1_to_v1"]

#%%
        self.obj1_to_obj2 = np.array([[1,0,0],
                                [1,1,0],
                                [1,1,1],
                                [0,1,1],
                                [0,0,1]])*self.weights["obj1_to_obj2"]
        self.obj1_to_obj2

        self.obj2_to_obj1 = np.transpose(self.obj1_to_obj2)*self.weights["obj2_to_obj1"]

#%%
        self.spat1_to_obj1 = np.identity(5)*self.weights["spat1_to_obj1"]
        self.spat1_to_obj1

        self.obj1_to_spat1 = np.transpose(self.spat1_to_obj1)*self.weights["obj1_to_spat1"]

#%%
        self.spat2_to_obj2 = np.identity(3)*self.weights["spat2_to_obj2"]
        self.spat2_to_obj2

        self.obj2_to_spat2 = np.transpose(self.spat2_to_obj2)*self.weights["obj2_to_spat2"]

#%%
        self.obj2_to_output = np.ones([3,1])*self.weights["obj2_to_output"]
        self.obj2_to_output

        self.output_to_obj2 = np.transpose(self.obj2_to_output)*self.weights["output_to_obj2"]

#Lateral Inhibition
 #%%
        self.spat1_lateral_inhibit = (np.identity(5)*2-np.ones(5))*self.weights["spat1_lateral_inhibit"]
        self.spat1_lateral_inhibit

#%%
        self.spat2_lateral_inhibit = (np.identity(3)*2-np.ones(3))*self.weights["spat1_lateral_inhibit"]
        self.spat2_lateral_inhibit