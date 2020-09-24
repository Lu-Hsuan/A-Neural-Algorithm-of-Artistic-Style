import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
content_layer_name = ['block5_conv2']

style_layer_names = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']
def creat_model(target_height,target_width,target_height_s,target_width_s):
    cModel = tf.keras.applications.VGG19(include_top=False,input_shape=(target_height,target_width,3))
    sModel = tf.keras.applications.VGG19(include_top=False,input_shape=(target_height_s,target_width_s,3))
    gModel = tf.keras.applications.VGG19(include_top=False,input_shape=(target_height,target_width,3))
    #cModel.summary(200)



    #x = np.random.randint(256, size=(1,target_width, target_height, 3)).astype('float32')
    #xx = np.random.randint(256, size=(1,target_width, target_height, 3)).astype('float32')
    P = get_feature_represent(content_layer_name,cModel)
    A = get_feature_represent(style_layer_names,sModel)
    F = get_feature_represent(content_layer_name,gModel)
    G = get_feature_represent(style_layer_names,gModel)
    c_model = tf.keras.models.Model(inputs = cModel.inputs,outputs = P)
    #c_model.summary()
    #Px = c_model(xx)
    s_model = tf.keras.models.Model(inputs = sModel.inputs,outputs = A)
    #s_model.summary()
    #Ax = s_model(xx)
    g_model = tf.keras.models.Model(inputs = gModel.inputs,outputs = [F,G])
    #g_model.summary()
    #Y = g_model(x)
    return c_model , s_model , g_model

def preprocess_array(x):
    x= x.astype(np.float64)
    #if x.shape != (target_width, target_height, 3):
    #    x = x.reshape((target_width, target_height, 3))
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    
    x = x[..., ::-1]  # BGR-->RGB
    #x = np.clip(x, 0, 255)
    #x = x.astype('uint8')
    return x#/255.
def postprocess_array(x):

    #if x.shape != (target_width, target_height, 3):
    #    x = x.reshape((target_width, target_height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68

    #x = x[..., ::-1]  # BGR-->RGB
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def get_feature_represent(layer_names, model):
    feature_matrices = []
    for ln in layer_names:
        select_layer = model.get_layer(ln)
        feature_raw = select_layer.output
        feature_raw_shape = feature_raw.shape
        N_l = feature_raw_shape[-1]
        M_l = feature_raw_shape[1]*feature_raw_shape[2]
        feature_matrix = tf.reshape(feature_raw, (M_l, N_l))
        feature_matrix = tf.transpose(feature_matrix)
        feature_matrices.append(feature_matrix)
    return feature_matrices
@tf.function
def get_content_loss(F, P):
    content_loss = 0.5*tf.math.reduce_sum(tf.math.square(F-P))
    return content_loss
@tf.function
def get_gram_matrix(F):
    G = tf.matmul(F, tf.transpose(F))
    return G
@tf.function
def get_style_loss(ws, Gs, As):
  style_loss = 0
  #for w , g , a in zip(ws,Gs,As):
  for i in range(4):
    g = Gs[i]
    w = ws[i]
    a = As[i]
    M_l = g.get_shape()[-1]
    #print(M_l)
    N_l = g.get_shape()[-2]
    #print(M_l , N_l)
    G_gram = get_gram_matrix(g)
    #print(G_gram.shape)
    A_gram = get_gram_matrix(a)
    #print(A_gram.shape)
    style_loss += w*0.25*tf.math.reduce_mean(tf.math.square(G_gram-A_gram))/(N_l**2*M_l**2)
  return style_loss
@tf.function
def get_total_loss(F,Gs,P,As,ws,alpha=1.0, beta=10000.0):
    content_loss = get_content_loss(F, P)
    style_loss = get_style_loss(ws, Gs, As)
    total_loss = alpha*content_loss + beta*style_loss
    return total_loss
    
@tf.function
def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]
  return x_var, y_var
@tf.function
def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


