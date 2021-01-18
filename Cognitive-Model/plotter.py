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
            "Spat2":self.spat2_arr,
            "Output":self.output_arr,
            "Obj2":self.obj2_arr,
            "Spat1":self.spat1_arr,
            "Obj1":self.obj1_arr,
            "V1":self.v1_arr,
            "Input":self.input_arr,
        }
        
        fig, axs = plt.subplots(4, 3)

        fig.set_size_inches(10, 8)

        axes = [axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,2],axs[2,1],axs[3,1]]
        for axis, layer_item in zip(axes,layer_dict.items()):
            sns.heatmap(layer_item[1], xticklabels=[], yticklabels=[], ax=axis)
            axis.set_title(layer_item[0])
        axs[1,1].axis("off")
        axs[2,0].axis("off")
        axs[2,2].axis("off")
        axs[3,0].axis("off")
        axs[3,2].axis("off")
        plt.show()

        print("input_arr")
        print(self.input_arr,'\n')

        print("v1_arr")
        print(self.v1_arr,'\n')

        print("spat1_arr")
        print(self.spat1_arr,'\n')

        print("obj1_arr")
        print(self.obj1_arr,'\n')

        print("output_arr")
        print(self.output_arr,'\n')