#%%
from connections import tf, np
from random import shuffle

import layers

# 調整每一種層和層之間訊息傳遞的運算方式
#%%
def two_d_softmax(tensor):
  two_d_arr = tensor.numpy() 
  shape = two_d_arr.shape
  one_d_arr = two_d_arr.flatten()
  arr = tf.nn.softmax(one_d_arr.astype(float)).numpy().reshape(list(shape))
  return tf.convert_to_tensor(arr,dtype=tf.dtypes.float32)

def two_d_sigmoid(tensor):
  two_d_arr = tensor.numpy() 
  shape = two_d_arr.shape
  one_d_arr = two_d_arr.flatten()
  arr = tf.nn.sigmoid(one_d_arr.astype(float)).numpy().reshape(list(shape))
  return tf.convert_to_tensor(arr,dtype=tf.dtypes.float32)

# 調整每一種層和層之間訊息傳遞的運算方式
#%%
# 調整bias的地方
bias = -0.5 # currently, used only in obj2_to_output

def input_to_v1(self):
  self.v1_arr = ((two_d_softmax(self.input_layer(self.input_tensor))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate)
  # self.v1_arr = (self.input_layer(self.input_tensor)+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate

def v1_to_spat1(self):
  # self.spat1_arr = ((two_d_softmax(self.v1(self.v1_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)
  self.spat1_arr = (self.v1(self.v1_arr, "Spat1")+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate
def spat1_to_v1(self):
  self.v1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "V1"))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate
  # self.v1_arr = (self.spat1(self.spat1_arr, "V1")+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate

def spat1_to_spat2(self):  
  # self.spat2_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "Spat2"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate)
  self.spat2_arr = (self.spat1(self.spat1_arr, "Spat2")+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate
def spat2_to_spat1(self):
  self.spat1_arr = (two_d_softmax(self.spat2(self.spat2_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate
  # self.spat1_arr = (self.spat2(self.spat2_arr, "Spat1")+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate

def v1_to_obj1(self):
  self.obj1_arr = ((two_d_softmax(self.v1(self.v1_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate)
  # self.obj1_arr = (self.v1(self.v1_arr, "Obj1")+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate
def obj1_to_v1(self):
  # self.v1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "V1"))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate
  self.v1_arr = (self.obj1(self.obj1_arr, "V1")+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate

def obj1_to_obj2(self):   
  self.obj2_arr = ((two_d_softmax(self.obj1(self.obj1_arr, "Obj2"))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate)
  # self.obj2_arr = (self.obj1(self.obj1_arr, "Obj2")+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate
def obj2_to_obj1(self):
  # self.obj1_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate
  self.obj1_arr = (self.obj2(self.obj2_arr, "Obj1")+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate

def spat1_to_obj1(self):
  self.obj1_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate)
  # self.obj1_arr = (self.spat1(self.spat1_arr, "Obj1")+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate
def obj1_to_spat1(self):
  # self.spat1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate
  self.spat1_arr = (self.obj1(self.obj1_arr, "Spat1")+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate

def spat2_to_obj2(self):
  self.obj2_arr = ((two_d_softmax(self.spat2(self.spat2_arr, "Obj2"))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate)
  # self.obj2_arr = (self.spat2(self.spat2_arr, "Obj2")+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate
def obj2_to_spat2(self):
  # self.spat2_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Spat2"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate
  self.spat2_arr = (self.obj2(self.obj2_arr, "Spat2")+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate

def obj2_to_output(self):   
  # self.output_arr = ((two_d_softmax(self.obj2(self.obj2_arr, "Output"))+self.output_arr)*(1-self.cascade_rate)+self.output_arr*self.cascade_rate)-0.05
  self.output_arr = ((self.obj2(self.obj2_arr, "Output")+self.output_arr)*(1-self.cascade_rate)+self.output_arr*self.cascade_rate)+bias
def output_to_obj2(self):      
  # self.obj2_arr = (two_d_softmax(self.output_layer(self.output_arr))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate
  self.obj2_arr = (self.output_layer(self.output_arr)+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate 

def spat1_lateral_inhibit(self):
  # self.spat1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "self"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate
  self.spat1_arr = ((self.spat1(self.spat1_arr, "self")+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)
def spat2_lateral_inhibit(self):
  # self.spat2_arr = (two_d_softmax(self.spat2(self.spat2_arr, "self"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate
  self.spat2_arr = ((self.spat2(self.spat2_arr, "self")+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate)

#%%
class VisualNetWork(tf.keras.Model):
  def __init__(self, wt, cascade_rate): 
    super(VisualNetWork, self).__init__(name='')
    
    layers.Layers(wt)

    self.cascade_rate = cascade_rate

    self.input_layer = layers.InputLayer()
    self.v1 = layers.V1()
    self.spat1 = layers.Spat1()
    self.spat2 = layers.Spat2()
    self.obj1 = layers.Obj1()
    self.obj2 = layers.Obj2()
    self.output_layer = layers.OutputLayer()

    self.input_arr = tf.zeros([2,7],dtype=tf.dtypes.float32)
    self.v1_arr = tf.zeros([2,7],dtype=tf.dtypes.float32)
    self.spat1_arr = tf.zeros([2,5],dtype=tf.dtypes.float32)
    self.spat2_arr = tf.zeros([2,3],dtype=tf.dtypes.float32)
    self.obj1_arr = tf.zeros([2,5],dtype=tf.dtypes.float32)
    self.obj2_arr = tf.zeros([2,3],dtype=tf.dtypes.float32)
    self.output_arr = tf.zeros([2,1],dtype=tf.dtypes.float32)
  
  def call(self, input_tensor, iscue):
    self.input_tensor = input_tensor

    # 調整哪些layer之間有連結，可以直接刪掉以切斷連結
    path_functions = [input_to_v1, #0
                          v1_to_spat1, #1
                          spat1_to_v1, #2
                          spat1_to_spat2, #3
                          spat2_to_spat1, #4
                          v1_to_obj1, #5
                          obj1_to_v1, #6
                          obj1_to_obj2, #7
                          obj2_to_obj1, #8
                          spat1_to_obj1, #9
                          obj1_to_spat1, #10
                          spat2_to_obj2, #11
                          obj2_to_spat2, #12
                          obj2_to_output, #13
                          output_to_obj2, #14
                          spat1_lateral_inhibit, #15
                          spat2_lateral_inhibit #16
                          ] 

    if iscue:
      cycle = 0
      while(cycle < 100):
        # Ignore below (for testing) 
        # wanted_paths = [i for i in range(17) if i not in [7,9,11,14]]
        # wanted_path_functions = list(np.array(path_functions)[wanted_paths])

        shuffle(path_functions)

        for path_f in path_functions:
          path_f(self)
          # Ignore below (for testing) 
          # print(path_f,'\n')
          # print("obj2_arr")
          # print(self.obj2_arr,'\n')
          # print("output_arr")
          # print(self.output_arr,'\n') 

        cue_node = self.output_arr[1][0]
        cycle += 1
        
      print("Cue appear...")
      print("output_arr")
      print(self.output_arr,'\n') 
      print("->cue_node:{}".format(cue_node))
      print("->cycle:{}".format(cycle),'\n')
    else:  
      cycle = 0
      target_node = self.output_arr[0][0]
      while(float(target_node) < 0.6):
        shuffle(path_functions)

        for path_f in path_functions:
          path_f(self)  

        
        target_node = self.output_arr[0][0]
        cycle += 1

      print("Target appear...")
      print("output_arr")
      print(self.output_arr,'\n') 
      print("->target_node:{}".format(target_node))
      print("->cycle:{}".format(cycle))

      return [float(target_node), cycle]
