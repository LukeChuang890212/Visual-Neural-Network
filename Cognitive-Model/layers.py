#%%
import tensorflow as tf 
import connections as conn 

#%%
class InputLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(InputLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.kernel = conn.input_to_v1
    print(self.kernel)
  def call(self, input):
    return tf.matmul(input, self.kernel)

layer = InputLayer(10)
layer(tf.zeros([10, 5]))
np.array([[1,2,3],[4,5,6]]).flatten().reshape([2,3])

#%%
class V1(tf.keras.layers.Layer):
  def __init__(self, num_outputs, next_layer):
    super(V1, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    if next_layer == "Spat1":
      self.kernel = conn.v1_to_spat1
    elif next_layer == "Obj1":
      self.kernel == conn.v1_to_obj1
  def call(self, input):
    return tf.matmul(input, self.kernel)

#%%
class Spat1(tf.keras.layers.Layer):
  def __init__(self, num_outputs, next_layer):
    super(Spat1, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    if next_layer == "Spat2":
      self.kernel = conn.spat1_to_spat2
    elif next_layer == "V1":
      self.kernel = conn.v1_to_spat1
  def call(self, input):
    return tf.matmul(input, self.kernel)

#%%
class Spat2(tf.keras.layers.Layer):
  def __init__(self, num_outputs, next_layer):
    super(Spat2, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    if next_layer == ""
  def call(self, input):
    return tf.matmul(input, self.kernel)

#%%
class Obj1(tf.keras.layers.Layer):
  def __init__(self, num_outputs, next_layer):
    super(Obj1, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
  if next_layer == "Obj2":
    self.kernel = conn.obj1_to_obj2
  elif next_layer == "V1":
    self.kernel = conn.v1_to_obj1
  def call(self, input):
    return tf.matmul(input, self.kernel)

#%%
class obj2(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(obj2, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

#%%
class output_layer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(output_layer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)
