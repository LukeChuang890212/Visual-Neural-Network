#%%
import tensorflow as tf 
import connections as conn 
from connections import np

#%%
def matmul_with_rowswap(input, kernel):
  input = np.array(input)
  cp1 = tf.matmul(input, kernel)

  input[[0,1]] = input[[1,0]] # swap the rows of input
  cp2 = tf.matmul(input, kernel)

  return cp1 + cp2

#%%
class InputLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(InputLayer, self).__init__()
    
  def call(self, input):
    self.kernel = conn.input_to_v1
    return tf.matmul(input, self.kernel)

#%%
class V1(tf.keras.layers.Layer):
  def __init__(self):
    super(V1, self).__init__()

  def call(self, input, next_layer):
    if next_layer == "Spat1":
      self.kernel = conn.v1_to_spat1
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "Obj1":
      self.kernel = conn.v1_to_obj1
      return tf.matmul(input, self.kernel)

#%%
class Spat1(tf.keras.layers.Layer):
  def __init__(self):
    super(Spat1, self).__init__()

  def call(self, input,next_layer):    
    if next_layer == "Spat2":
      self.kernel = conn.spat1_to_spat2
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "V1":
      self.kernel = conn.spat1_to_v1
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "Obj1":
      self.kernel = conn.spat1_to_obj1
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "self":
      self.kernel = conn.spat1_lateral_inhibit
      return tf.matmul(input, self.kernel)

#%%
class Spat2(tf.keras.layers.Layer):
  def __init__(self):
    super(Spat2, self).__init__()

  def call(self, input, next_layer):
    if next_layer == "Obj2":
      self.kernel = conn.spat2_to_obj2
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "Spat1":
      self.kernel = conn.spat2_to_spat1
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "self":
      self.kernel = conn.spat2_lateral_inhibit
      return tf.matmul(input, self.kernel)

#%%
class Obj1(tf.keras.layers.Layer):
  def __init__(self):
    super(Obj1, self).__init__()

  def call(self, input, next_layer):
    if next_layer == "Obj2":
      self.kernel = conn.obj1_to_obj2
      return tf.matmul(input, self.kernel)
    elif next_layer == "V1":
      self.kernel = conn.obj1_to_v1
      return tf.matmul(input, self.kernel)
    elif next_layer == "Spat1":
      self.kernel = conn.obj1_to_spat1
      return matmul_with_rowswap(input,self.kernel)

#%%
class Obj2(tf.keras.layers.Layer):
  def __init__(self):
    super(Obj2, self).__init__()

  def call(self, input, next_layer):
    if next_layer == "Output":
      self.kernel = conn.obj2_to_output
      return tf.matmul(input, self.kernel)
    elif next_layer == "Spat2":
      self.kernel = conn.obj2_to_spat2
      return matmul_with_rowswap(input,self.kernel)
    elif next_layer == "Obj1":
      self.kernel = conn.obj2_to_obj1
      return tf.matmul(input, self.kernel)

#%%
class OutputLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(OutputLayer, self).__init__()

  def call(self, input):
    self.kernel = conn.output_to_obj2
    return tf.matmul(input, self.kernel)
