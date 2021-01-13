#%%
import matplotlib.pyplot as plt 
import seaborn as sns

#%%
class Plotter():
    def __init__(self):
        pass
    def set_arrs(self,input_arr,v1_arr,spat1_arr,spat2_arr,obj1_arr,obj2_arr,output_arr):
        self.input_arr = input_arr
        self.v1_arr = v1_arr
        self.spat1_arr = spat1_arr
        self.spat2_arr = spat2_arr
        self.obj1_arr = obj1_arr
        self.obj2_arr = obj2_arr
        self.output_arr = output_arr
    def plot(self):
        # plt.subplot
        layer_dict = {
            "Input":self.input_arr,
            "V1":self.v1_arr,
            "Spat1":self.spat1_arr,
            "Spat2":self.spat2_arr,
            "Obj1":self.obj1_arr,
            "Obj2":self.obj2_arr,
            "Output":self.output_arr,
        }
        
        fig, axs = plt.subplots(4, 3)
        axes = [axs[3,1],axs[2,1],axs[1,0],axs[0,0],axs[1,2],axs[0,2],axs[0,1]]
        for axis, layer_item in zip(axes,layer_dict.items()):
            sns.heatmap(layer_item[1], xticklabels=[],yticklabels=[],ax=axis)
            axis.set_title(layer_item[0])
        plt.show()
