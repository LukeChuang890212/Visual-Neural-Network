#%%
import numpy as np

#%%
weights = dict(
    input_to_v1 = 3,
    v1_to_input = 0,
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

wt = 0.1
weights = {k:v*wt for k, v in weights.items()}
weights

#%%
input_to_v1 = np.identity(7)
input_to_v1

v1_to_input = np.transpose(input_to_v1)

#%%
v1_to_spat1 = np.array([[1,0,0,0,0],
                        [1,1,0,0,0],
                        [1,1,1,0,0],
                        [0,1,1,1,0],
                        [0,0,1,1,1],
                        [0,0,0,1,1],
                        [0,0,0,0,1]])*weights["v1_to_spat1"]
v1_to_spat1

spat1_to_v1 = np.transpose(v1_to_spat1)*weights["spat1_to_v1"]

#%%
spat1_to_spat2 = np.array([[1,0,0],
                        [1,1,0,],
                        [1,1,1],
                        [0,1,1],
                        [0,0,1]])*weights["spat1_to_spat2"]
spat1_to_spat2

spat2_to_spat1 = np.transpose(spat1_to_spat2)*weights["spat2_to_spat1"]

#%%
v1_to_obj1 = np.array([[1,0,0,0,0],
                        [1,1,0,0,0],
                        [1,1,1,0,0],
                        [0,1,1,1,0],
                        [0,0,1,1,1],
                        [0,0,0,1,1],
                        [0,0,0,0,1]])*weights["v1_to_obj1"]
v1_to_obj1

obj1_to_v1 = np.transpose(v1_to_obj1)*weights["obj1_to_v1"]

#%%
obj1_to_obj2 = np.array([[1,0,0],
                        [1,1,0],
                        [1,1,1],
                        [0,1,1],
                        [0,0,1]])*weights["obj1_to_obj2"]
obj1_to_obj2

obj2_to_obj1 = np.transpose(obj1_to_obj2)*weights["obj2_to_obj1"]

#%%
spat1_to_obj1 = np.identity(5)*weights["spat1_to_obj1"]
spat1_to_obj1

obj1_to_spat1 = np.transpose(spat1_to_obj1)*weights["obj1_to_spat1"]

#%%
spat2_to_obj2 = np.identity(3)*weights["spat2_to_obj2"]
spat2_to_obj2

obj2_to_spat2 = np.transpose(spat2_to_obj2)*weights["obj2_to_spat2"]

#%%
obj2_to_output = np.ones([3,1])*weights["obj2_to_output"]
obj2_to_output

output_to_obj2 = np.transpose(obj2_to_output)*weights["output_to_obj2"]

#Lateral Inhibition
#%%
spat1_lateral_inhibit = (np.identity(5)*2-np.ones(5))*weights["spat1_lateral_inhibit"]
spat1_lateral_inhibit

#%%
spat2_lateral_inhibit = (np.identity(3)*2-np.ones(3))*weights["spat1_lateral_inhibit"]
spat2_lateral_inhibit