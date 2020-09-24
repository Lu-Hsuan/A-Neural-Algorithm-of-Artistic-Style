import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from model_loss import *
import argparse

print(tf.__version__)
tf.keras.backend.set_image_data_format(
     'channels_last'
)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data_img", help="path to dataset")
parser.add_argument("--style_path", type=str, default="./style_img", help="path to style")
parser.add_argument("--out_path", type=str, default='./outputs',help="out_path")
opt_ = parser.parse_args()
print(opt_)
os.makeedirs(opt_.out_path,exist_ok=True)
#@title train loop {form-width : "20%"}
s_w = 3e5
c_w = 1e3
v_w = 1e3
img_c = cv2.imread(opt_.data_path)
img_s = cv2.imread(opt_.style_path)

img_c = cv2.resize(img_c,(256,256))
img_s = cv2.resize(img_s,(256,256))

target_width = img_c.shape[1]
target_height = img_c.shape[0]

target_width_s = img_s.shape[1]
target_height_s = img_s.shape[0]
load_model = True #@param {type:"boolean"}
if(load_model == True):
  c_model , s_model , g_model = creat_model(target_height,target_width,target_height_s,target_width_s)

print(img_c.shape)
print(img_s.shape)
img_c = preprocess_array(img_c)
img_s = preprocess_array(img_s)
img_c = img_c[np.newaxis,...]
img_s = img_s[np.newaxis,...]
print(img_c.shape)
Px = c_model(img_c)
Ax = s_model(img_s)
ws = np.ones(len(style_layer_names),dtype=np.float32)/(len(style_layer_names))
#print(Px,Ax)

opt = tf.optimizers.Adam(learning_rate=15, beta_1=0.99, epsilon=1e-1)
generate_image = np.random.randint(256, size=(1,target_height, target_width, 3)).astype('float32')
#'''
img_g = img_c*0.2+img_s*0.01 + generate_image*0.7
plt.imshow(postprocess_array(img_g[0]))
plt.show()
#'''

x = tf.Variable(img_g,name='input',dtype=tf.float32)
@tf.function
def train_step(x,Px,Ax): #X PX AX
  with tf.GradientTape() as tape:
    #tape.watch(supervised_nn.trainable_variables)
    Y = g_model(x, training=True)
    loss 	= get_total_loss(Y[0],Y[1],Px,Ax,ws,alpha=c_w, beta=s_w)
    loss += v_w*total_variation_loss(x)
  g = tape.gradient(loss, [x]) 
  opt.apply_gradients(zip(g,[x]))
  return loss

def train():
  #generate_image = preprocess_input(np.expand_dims(generate_image, 0))
  for i in range(1,1000):
    
    loss_ = train_step(x,Px,Ax)	           # trainin
    
    if i % 250 == 0:
      #print(opt.iterations)
      print(loss_)
      plt.imshow(postprocess_array(np.array(x[0])))
      plt.show()
      """
      print(lr_schedule(i))
      loss = loss / 100
      print(f'{i} {loss}',end='\n')
      losses.append(loss)
      loss = 0
      """
train()
print(x.shape)
fig, axes = plt.subplots(1, 3,figsize=(8,8))  
[axes[i].axis('off') for i in range(3)]
axes[0].imshow(postprocess_array(img_c[0].copy()))
axes[1].imshow(postprocess_array(img_s[0].copy()))
axes[2].imshow(postprocess_array(np.array(x[0])))
fig.savefig(f"{opt_.out_path}/mix.png", bbox_inches="tight", pad_inches=0.0,dpi=400)
fig.show()
input()