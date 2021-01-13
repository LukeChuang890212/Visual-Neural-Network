#%%
from connections import tf, np

class Cascade():
#%%
    def __init__(self, bias):
        # 調整bias的地方
        Cascade.bias = bias # currently, used only in obj2_to_output

# 調整每一種層和層之間訊息傳遞的運算方式
#%%
    def two_d_softmax(self,tensor):
        two_d_arr = tensor.numpy() 
        shape = two_d_arr.shape
        one_d_arr = two_d_arr.flatten()
        arr = tf.nn.softmax(one_d_arr.astype(float)).numpy().reshape(list(shape))
        return tf.convert_to_tensor(arr,dtype=tf.dtypes.float32)

    def two_d_sigmoid(self,tensor):
        two_d_arr = tensor.numpy() 
        shape = two_d_arr.shape
        one_d_arr = two_d_arr.flatten()
        arr = tf.nn.sigmoid(one_d_arr.astype(float)).numpy().reshape(list(shape))
        return tf.convert_to_tensor(arr,dtype=tf.dtypes.float32)

# 調整每一種層和層之間訊息傳遞的運算方式
#%%
    def input_to_v1(self,classname):
        classname.v1_arr = ((self.two_d_softmax(classname.input_layer(classname.input_tensor))+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate)
        # classname.v1_arr = (classname.input_layer(classname.input_tensor)+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate

    def v1_to_spat1(self,classname):
        # classname.spat1_arr = ((self.two_d_softmax(classname.v1(classname.v1_arr, "Spat1"))+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate)
        classname.spat1_arr = (classname.v1(classname.v1_arr, "Spat1")+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate
    def spat1_to_v1(self,classname):
        classname.v1_arr = (self.two_d_softmax(classname.spat1(classname.spat1_arr, "V1"))+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate
        # classname.v1_arr = (classname.spat1(classname.spat1_arr, "V1")+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate

    def spat1_to_spat2(self,classname):  
        # classname.spat2_arr = ((self.two_d_softmax(classname.spat1(classname.spat1_arr, "Spat2"))+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate)
        classname.spat2_arr = (classname.spat1(classname.spat1_arr, "Spat2")+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate
    def spat2_to_spat1(self,classname):
        classname.spat1_arr = (self.two_d_softmax(classname.spat2(classname.spat2_arr, "Spat1"))+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate
        # classname.spat1_arr = (classname.spat2(classname.spat2_arr, "Spat1")+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate

    def v1_to_obj1(self,classname):
        classname.obj1_arr = ((self.two_d_softmax(classname.v1(classname.v1_arr, "Obj1"))+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate)
        # classname.obj1_arr = (classname.v1(classname.v1_arr, "Obj1")+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate
    def obj1_to_v1(self,classname):
        # classname.v1_arr = (self.two_d_softmax(classname.obj1(classname.obj1_arr, "V1"))+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate
        classname.v1_arr = (classname.obj1(classname.obj1_arr, "V1")+classname.v1_arr)*(1-classname.cascade_rate)+classname.v1_arr*classname.cascade_rate

    def obj1_to_obj2(self,classname):   
        classname.obj2_arr = ((self.two_d_softmax(classname.obj1(classname.obj1_arr, "Obj2"))+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate)
        # classname.obj2_arr = (classname.obj1(classname.obj1_arr, "Obj2")+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate
    def obj2_to_obj1(self,classname):
        # classname.obj1_arr = (self.two_d_softmax(classname.obj2(classname.obj2_arr, "Obj1"))+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate
        classname.obj1_arr = (classname.obj2(classname.obj2_arr, "Obj1")+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate

    def spat1_to_obj1(self,classname):
        classname.obj1_arr = ((self.two_d_softmax(classname.spat1(classname.spat1_arr, "Obj1"))+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate)
        # classname.obj1_arr = (classname.spat1(classname.spat1_arr, "Obj1")+classname.obj1_arr)*(1-classname.cascade_rate)+classname.obj1_arr*classname.cascade_rate
    def obj1_to_spat1(self,classname):
        # classname.spat1_arr = (self.two_d_softmax(classname.obj1(classname.obj1_arr, "Spat1"))+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate
        classname.spat1_arr = (classname.obj1(classname.obj1_arr, "Spat1")+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate

    def spat2_to_obj2(self,classname):
        classname.obj2_arr = ((self.two_d_softmax(classname.spat2(classname.spat2_arr, "Obj2"))+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate)
        # classname.obj2_arr = (classname.spat2(classname.spat2_arr, "Obj2")+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate
    def obj2_to_spat2(self,classname):
        # classname.spat2_arr = (self.two_d_softmax(classname.obj2(classname.obj2_arr, "Spat2"))+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate
        classname.spat2_arr = (classname.obj2(classname.obj2_arr, "Spat2")+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate

    def obj2_to_output(self,classname):   
        # classname.output_arr = ((self.two_d_softmax(classname.obj2(classname.obj2_arr, "Output"))+classname.output_arr)*(1-classname.cascade_rate)+classname.output_arr*classname.cascade_rate)-0.05
        classname.output_arr = ((classname.obj2(classname.obj2_arr, "Output")+classname.output_arr)*(1-classname.cascade_rate)+classname.output_arr*classname.cascade_rate)+Cascade.bias
    def output_to_obj2(self,classname):      
        # classname.obj2_arr = (self.two_d_softmax(classname.output_layer(classname.output_arr))+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate
        classname.obj2_arr = (classname.output_layer(classname.output_arr)+classname.obj2_arr)*(1-classname.cascade_rate)+classname.obj2_arr*classname.cascade_rate 

    def spat1_lateral_inhibit(self,classname):
        # classname.spat1_arr = (self.two_d_softmax(classname.spat1(classname.spat1_arr, "classname"))+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate
        classname.spat1_arr = ((classname.spat1(classname.spat1_arr, "self")+classname.spat1_arr)*(1-classname.cascade_rate)+classname.spat1_arr*classname.cascade_rate)
    def spat2_lateral_inhibit(self,classname):
        # classname.spat2_arr = (self.two_d_softmax(classname.spat2(classname.spat2_arr, "classname"))+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate
        classname.spat2_arr = ((classname.spat2(classname.spat2_arr, "self")+classname.spat2_arr)*(1-classname.cascade_rate)+classname.spat2_arr*classname.cascade_rate)
    def set_zero(self,classname):
        classname.obj1_arr = tf.Variable(classname.obj1_arr)[0].assign([0,0,0,0,0])
        classname.obj2_arr = tf.Variable(classname.obj2_arr)[0].assign([0,0,0])
        classname.output_arr = tf.Variable(classname.output_arr)[0].assign([0])
        # classname.obj1_arr = tf.Variable(classname.obj1_arr)[0].assign([0,0,0,0,0])
        # classname.obj2_arr = tf.Variable(classname.obj2_arr)[0].assign([0,0,0])
        # classname.output_arr = tf.Variable(classname.output_arr)[0].assign([0])