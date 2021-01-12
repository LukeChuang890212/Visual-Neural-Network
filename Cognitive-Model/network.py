#%%
import layers
from layers import conn

from layers import tf
from connections import np

from random import shuffle

#%%
def two_d_softmax(tensor):
  two_d_arr = tensor.numpy() 
  shape = two_d_arr.shape
  one_d_arr = two_d_arr.flatten()
  arr = tf.nn.softmax(one_d_arr.astype(float)).numpy().reshape(list(shape))
  return tf.convert_to_tensor(arr,dtype=tf.dtypes.double)

def input_to_v1(self):
  self.v1_arr = ((two_d_softmax(self.input_layer(self.input_tensor))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate)

def v1_to_spat1(self):
  self.spat1_arr = ((two_d_softmax(self.v1(self.v1_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)
def spat1_to_v1(self):
  self.v1_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "V1"))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate)

def spat1_to_spat2(self):  
  self.spat2_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "Spat2"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate)
def spat2_to_spat1(self):
  self.spat1_arr = ((two_d_softmax(self.spat2(self.spat2_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)

def v1_to_obj1(self):
  self.obj1_arr = ((two_d_softmax(self.v1(self.v1_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate)
def obj1_to_v1(self):
  self.v1_arr = ((two_d_softmax(self.obj1(self.obj1_arr, "V1"))+self.v1_arr)*(1-self.cascade_rate)+self.v1_arr*self.cascade_rate)

def obj1_to_obj2(self):   
  self.obj2_arr = ((two_d_softmax(self.obj1(self.obj1_arr, "Obj2"))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate)
def obj2_to_obj1(self):
  self.obj1_arr = ((two_d_softmax(self.obj2(self.obj2_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate)

def spat1_to_obj1(self):
  self.obj1_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "Obj1"))+self.obj1_arr)*(1-self.cascade_rate)+self.obj1_arr*self.cascade_rate)
def obj1_to_spat1(self):
  self.spat1_arr = ((two_d_softmax(self.obj1(self.obj1_arr, "Spat1"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)

def spat2_to_obj2(self):
  self.obj2_arr = ((two_d_softmax(self.spat2(self.spat2_arr, "Obj2"))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate)
def obj2_to_spat2(self):
  self.spat2_arr = ((two_d_softmax(self.obj2(self.obj2_arr, "Spat2"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate)

def obj2_to_output(self):       
  self.output_arr = ((two_d_softmax(self.obj2(self.obj2_arr, "Output"))+self.output_arr)*(1-self.cascade_rate)+self.output_arr*self.cascade_rate)
def output_to_obj2(self):      
  self.obj2_arr = ((two_d_softmax(self.output_layer(self.output_arr))+self.obj2_arr)*(1-self.cascade_rate)+self.obj2_arr*self.cascade_rate)

def spat1_lateral_inhibit(self):
  self.spat1_arr = ((two_d_softmax(self.spat1(self.spat1_arr, "self"))+self.spat1_arr)*(1-self.cascade_rate)+self.spat1_arr*self.cascade_rate)
def spat2_lateral_inhibit(self):
  self.spat2_arr = ((two_d_softmax(self.spat2(self.spat2_arr, "self"))+self.spat2_arr)*(1-self.cascade_rate)+self.spat2_arr*self.cascade_rate)

#%%
class VisualNetWork(tf.keras.Model):
  def __init__(self, wt, cascade_rate):
    super(VisualNetWork, self).__init__(name='')

    conn.wt = wt

    self.cascade_rate = cascade_rate

    self.input_layer = layers.InputLayer()
    self.v1 = layers.V1()
    self.spat1 = layers.Spat1()
    self.spat2 = layers.Spat2()
    self.obj1 = layers.Obj1()
    self.obj2 = layers.Obj2()
    self.output_layer = layers.OutputLayer()

    self.input_arr = tf.zeros([2,7],dtype=tf.dtypes.double)
    self.v1_arr = tf.zeros([2,7],dtype=tf.dtypes.double)
    self.spat1_arr = tf.zeros([2,5],dtype=tf.dtypes.double)
    self.spat2_arr = tf.zeros([2,3],dtype=tf.dtypes.double)
    self.obj1_arr = tf.zeros([2,5],dtype=tf.dtypes.double)
    self.obj2_arr = tf.zeros([2,3],dtype=tf.dtypes.double)
    self.output_arr = tf.zeros([2,1],dtype=tf.dtypes.double)
  
  def call(self, input_tensor, iscue):
    self.input_tensor = input_tensor

    path_functions = [input_to_v1,
                          v1_to_spat1,
                          spat1_to_v1,
                          spat1_to_spat2,
                          spat2_to_spat1,
                          v1_to_obj1,
                          obj1_to_v1,
                          obj1_to_obj2,
                          obj2_to_obj1,
                          spat1_to_obj1,
                          obj1_to_spat1,
                          spat2_to_obj2,
                          obj2_to_spat2,
                          obj2_to_output,
                          output_to_obj2,
                          spat1_lateral_inhibit,
                          spat2_lateral_inhibit]
    if iscue:
      cycle = 0
      while(cycle < 50):
        shuffle(path_functions)

        for path_f in path_functions:
          path_f(self)  

        target_node = self.output_arr[0][0]
        cycle += 1

      print("Cue appear...")
      print("->target_node:{}".format(target_node))
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
      print("->target_node:{}".format(target_node))
      print("->cycle:{}".format(cycle))

      return [target_node, cycle]

