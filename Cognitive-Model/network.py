#%%
import layers
from layers import tf
from connections import np

#%%
def two_d_softmax(tensor):
  two_d_arr = tensor.numpy() 
  shape = two_d_arr.shape
  one_d_arr = two_d_arr.flatten()
  arr = tf.nn.softmax(one_d_arr.astype(float)).numpy().reshape(list(shape))
  return tf.convert_to_tensor(arr)

#%%
class VisualNetWork(tf.keras.Model):
  def __init__(self):
    super(VisualNetWork, self).__init__(name='')

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
    if iscue:
      cycle = 0
      while(cycle < 100):
        self.v1_arr = (two_d_softmax(self.input_layer(input_tensor))+self.v1_arr)/2

        self.spat1_arr = (two_d_softmax(self.v1(self.v1_arr, "Spat1"))+self.spat1_arr)/2
        self.v1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "V1"))+self.v1_arr)/2

        self.spat2_arr = (two_d_softmax(self.spat1(self.spat1_arr, "Spat2"))+self.spat2_arr)/2
        self.spat1_arr = (two_d_softmax(self.spat2(self.spat2_arr, "Spat1"))+self.spat1_arr)/2

        self.obj1_arr = (two_d_softmax(self.v1(self.v1_arr, "Obj1"))+self.obj1_arr)/2
        self.v1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "V1"))+self.v1_arr)/2
        
        self.obj2_arr = (two_d_softmax(self.obj1(self.obj1_arr, "Obj2"))+self.obj2_arr)/2
        self.obj1_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Obj1"))+self.obj1_arr)/2

        self.obj1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "Obj1"))+self.obj1_arr)/2
        self.spat1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "Spat1"))+self.spat1_arr)/2

        self.obj2_arr = (two_d_softmax(self.spat2(self.spat2_arr, "Obj2"))+self.obj2_arr)/2
        self.spat2_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Spat2"))+self.spat2_arr)/2
        
        self.output_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Output"))+self.output_arr)/2
        self.obj2_arr = (two_d_softmax(self.output_layer(self.output_arr))+self.obj2_arr)/2

        self.spat1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "self"))+self.spat1_arr)/2
        self.spat2_arr = (two_d_softmax(self.spat2(self.spat2_arr, "self"))+self.spat2_arr)/2    

        cycle += 1
    else:  
      cycle = 0
      target_node = self.output_arr[0][0]
      while(float(target_node) < 0.6):
        self.v1_arr = (two_d_softmax(self.input_layer(input_tensor))+self.v1_arr)/2

        print(self.spat1_arr)
        print(two_d_softmax(self.v1(self.v1_arr, "Spat1")))
        self.spat1_arr = (two_d_softmax(self.v1(self.v1_arr, "Spat1"))+self.spat1_arr)/2
        self.v1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "V1"))+self.v1_arr)/2
        print(self.spat1_arr)
        input("continue")

        self.spat2_arr = (two_d_softmax(self.spat1(self.spat1_arr, "Spat2"))+self.spat2_arr)/2
        self.spat1_arr = (two_d_softmax(self.spat2(self.spat2_arr, "Spat1"))+self.spat1_arr)/2

        self.obj1_arr = (two_d_softmax(self.v1(self.v1_arr, "Obj1"))+self.obj1_arr)/2
        self.v1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "V1"))+self.v1_arr)/2
        
        self.obj2_arr = (two_d_softmax(self.obj1(self.obj1_arr, "Obj2"))+self.obj2_arr)/2
        self.obj1_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Obj1"))+self.obj1_arr)/2

        self.obj1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "Obj1"))+self.obj1_arr)/2
        self.spat1_arr = (two_d_softmax(self.obj1(self.obj1_arr, "Spat1"))+self.spat1_arr)/2

        self.obj2_arr = (two_d_softmax(self.spat2(self.spat2_arr, "Obj2"))+self.obj2_arr)/2
        self.spat2_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Spat2"))+self.spat2_arr)/2
        
        self.output_arr = (two_d_softmax(self.obj2(self.obj2_arr, "Output"))+self.output_arr)/2
        self.obj2_arr = (two_d_softmax(self.output_layer(self.output_arr))+self.obj2_arr)/2

        self.spat1_arr = (two_d_softmax(self.spat1(self.spat1_arr, "self"))+self.spat1_arr)/2
        self.spat2_arr = (two_d_softmax(self.spat2(self.spat2_arr, "self"))+self.spat2_arr)/2

        target_node = two_d_softmax(self.output_arr)[0][0]
        cycle += 1
        # print(target_node)

      return [target_node, cycle]

