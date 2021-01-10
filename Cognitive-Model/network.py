#%%
import layers
from layers import tf
from connections import np

#%%
def two_d_softmax(tensor):
  two_d_arr = tensor.numpy() 
  shape = two_d_arr.shape
  one_d_arr = two_d_arr.flatten()
  tensor = tf.nn.softmax(one_d_arr.astype(float)).numpy().reshape(list(shape))
  return tensor

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

    self.input_arr = np.zeros([2,7])
    self.v1_arr = np.zeros([2,7])
    self.spat1_arr = np.zeros([2,5])
    self.spat2_arr = np.zeros([2,3])
    self.obj1_arr = np.zeros([2,5])
    self.obj2_arr = np.zeros([2,3])
    self.output_arr = np.zeros([2,1])
  
  def call(self, input_tensor, iscue):
    if iscue:
      cycle = 0
      while(cycle < 100):
        self.v1_arr += self.input_layer(input_tensor)

        self.spat1_arr += self.v1(self.v1_arr, "Spat1")
        self.v1_arr += self.spat1(self.spat1_arr, "V1")

        self.spat2_arr += self.spat1(self.spat1_arr, "Spat2")
        self.spat1_arr += self.spat2(self.spat2_arr, "Spat1")

        self.obj1_arr += self.v1(self.v1_arr, "Obj1")
        self.v1_arr += self.obj1(self.obj1_arr, "V1")
        
        self.obj2_arr += self.obj1(self.obj1_arr, "Obj2")
        self.obj1_arr += self.obj2(self.obj2_arr, "Obj1")

        self.obj1_arr += self.spat1(self.spat1_arr, "Obj1")
        self.spat1_arr += self.obj1(self.obj1_arr, "Spat1")

        self.obj2_arr += self.spat2(self.spat2_arr, "Obj2")
        self.spat2_arr += self.obj2(self.obj2_arr, "Spat2")
        
        self.output_arr += self.obj2(self.obj2_arr, "Output")
        self.obj2_arr += self.output_layer(self.output_arr)

        self.spat1_arr += self.spat1(self.spat1_arr, "self")
        self.spat2_arr += self.spat2(self.spat2_arr, "self")

        cycle += 1
    else:  
      cycle = 0
      target_node = self.output_arr.numpy()[0][0]
      while(float(target_node) < 0.6):
        self.v1_arr = (two_d_softmax(self.input_layer(input_tensor))+self.v1_arr)/2

        self.spat1_arr += self.v1(self.v1_arr, "Spat1")
        self.v1_arr += self.spat1(self.spat1_arr, "V1")

        self.spat2_arr += self.spat1(self.spat1_arr, "Spat2")
        self.spat1_arr += self.spat2(self.spat2_arr, "Spat1")

        self.obj1_arr += self.v1(self.v1_arr, "Obj1")
        self.v1_arr += self.obj1(self.obj1_arr, "V1")
        
        self.obj2_arr += self.obj1(self.obj1_arr, "Obj2")
        self.obj1_arr += self.obj2(self.obj2_arr, "Obj1")

        self.obj1_arr += self.spat1(self.spat1_arr, "Obj1")
        self.spat1_arr += self.obj1(self.obj1_arr, "Spat1")

        self.obj2_arr += self.spat2(self.spat2_arr, "Obj2")
        self.spat2_arr += self.obj2(self.obj2_arr, "Spat2")
        
        self.output_arr += self.obj2(self.obj2_arr, "Output")
        self.obj2_arr += self.output_layer(self.output_arr)

        self.spat1_arr += self.spat1(self.spat1_arr, "self")
        self.spat2_arr += self.spat2(self.spat2_arr, "self")

        target_node = two_d_softmax(self.output_arr)[0][0]
        cycle += 1
        print(target_node)

      return [target_node, cycle]
