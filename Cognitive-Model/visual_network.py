import layers
from cascade import Cascade
from plotter import Plotter

#%%
class VisualNetWork(tf.keras.Model, Cascade):
  def __init__(self, wt, cascade_rate, bias): 
    super(VisualNetWork, self).__init__(name='')
    
    layers.Layers(wt) # set weight
    self.cascade_rate = cascade_rate # set cascade_rate
    self.cascade = Cascade(bias) # set bias

    self.plotter = Plotter()

    self.input_layer = layers.InputLayer()
    self.v1 = layers.V1()
    self.spat1 = layers.Spat1()
    self.spat2 = layers.Spat2()
    self.obj1 = layers.Obj1()
    self.obj2 = layers.Obj2()
    self.output_layer = layers.OutputLayer()

    self.input_arr = tf.Variable(tf.zeros([2,7],dtype=tf.dtypes.float32))
    self.v1_arr = tf.Variable(tf.zeros([2,7],dtype=tf.dtypes.float32))
    self.spat1_arr = tf.Variable(tf.zeros([2,5],dtype=tf.dtypes.float32))
    self.spat2_arr = tf.Variable(tf.zeros([2,3],dtype=tf.dtypes.float32))
    self.obj1_arr = tf.Variable(tf.zeros([2,5],dtype=tf.dtypes.float32))
    self.obj2_arr = tf.Variable(tf.zeros([2,3],dtype=tf.dtypes.float32))
    self.output_arr = tf.Variable(tf.zeros([2,1],dtype=tf.dtypes.float32))
  
  def call(self, input_tensor, iscue):
    self.input_arr = input_tensor
    self.input_tensor = input_tensor

    # 調整哪些layer之間有連結，可以直接刪掉以切斷連結
    path_functions = [self.cascade.input_to_v1, #0
                          self.cascade.v1_to_spat1, #1
                          self.cascade.spat1_to_v1, #2
                          self.cascade.spat1_to_spat2, #3
                          self.cascade.spat2_to_spat1, #4
                          self.cascade.v1_to_obj1, #5
                          self.cascade.obj1_to_v1, #6
                          self.cascade.obj1_to_obj2, #7
                          self.cascade.obj2_to_obj1, #8
                          self.cascade.spat1_to_obj1, #9
                          self.cascade.obj1_to_spat1, #10
                          self.cascade.spat2_to_obj2, #11
                          self.cascade.obj2_to_spat2, #12
                          self.cascade.obj2_to_output, #13
                          self.cascade.output_to_obj2, #14
                          self.cascade.spat1_lateral_inhibit, #15
                          self.cascade.spat2_lateral_inhibit, #16
                          self.cascade.reset_target_zero, #17
                          self.cascade.reset_cue_zero #18
                          ] 

    if iscue:
      print("Cue appear...",'\n')

       # 挑掉不要的paths
      unwanted = [18] # 將不要的路徑編號填入此 (編號請查詢上方)
      wanted_paths = [i for i in range(len(path_functions)) if i not in unwanted]
      path_functions = list(np.array(path_functions)[wanted_paths])

      cycle = 0
      # 調整cue的cycle數
      cue_cycle_num = 100
      while(cycle < cue_cycle_num):
        shuffle(path_functions)

        for path_f in path_functions:
          path_f(self)

        cue_node = self.output_arr[1][0]
        cycle += 1

        if cycle % 20 == 0:
          print("cycle:{}".format(cycle),'\n')
          self.plotter.set_arrs(self.input_arr,self.v1_arr,self.spat1_arr,self.spat2_arr,self.obj1_arr,self.obj2_arr,self.output_arr)
          self.plotter.plot()
        
      print("output")
      print(self.output_arr,'\n') 
      print("->cue_node:{}".format(cue_node))
      print("->cycle:{}".format(cycle),'\n')
    else: 
      print("Target appear...",'\n')

      # 挑掉不要的paths
      unwanted = [17] # 將不要的路徑編號填入此 (編號請查詢上方)
      wanted_paths = [i for i in range(len(path_functions)) if i not in unwanted]
      path_functions = list(np.array(path_functions)[wanted_paths])

      cycle = 0
      target_node = self.output_arr[0][0]
      threshold = 0.6 
      while(float(target_node) < threshold):
        shuffle(path_functions)

        for path_f in path_functions:
          path_f(self)  

        target_node = self.output_arr[0][0]
        cycle += 1

        if cycle % 20 == 0:
          print("cycle:{}".format(cycle),'\n')
          self.plotter.set_arrs(self.input_arr,self.v1_arr,self.spat1_arr,self.spat2_arr,self.obj1_arr,self.obj2_arr,self.output_arr)
          self.plotter.plot()

      print("output")
      print(self.output_arr,'\n') 
      print("->target_node:{}".format(target_node))
      print("->cycle:{}".format(cycle))

      return [float(target_node), cycle]