from keras_facenet import FaceNet
import cv2
import numpy as np
import tensorflow as tf
from keras_facenet import embedding_model, metadata, utils
from keras.models import Model, Sequential, load_model

from google.protobuf.text_format import Parse

front_face ='E:/facenet/date_jpeg/front face/201/201_01_01_010_08_cropped.jpg'
side_face ='E:/facenet/date_jpeg/side face/201/201_01_01_010_08_cropped_test.jpg'
#model_path = 'E:/DLtpgan_with_facenet/facenet_model/20180408-102900.pb'
model_path = 'E:/facenet/models/facenet_keras.h5'

def reshape_4d(img):
    size = img.shape
    img_4d = np.reshape(img, (1,size[0] ,size[1] ,size[2]))
    
    return img_4d

if __name__ == "__main__":
        

    '''
        graph_def = tf.GraphDef()
    
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        Parse(f.read(), graph_def)
        
    tf.import_graph_def(graph_def, name='facenet')
    '''
    facenet_model = load_model('E:/facenet/models/facenet_keras.h5')
    
    #facenet_model.summary()
'''
    embedder = FaceNet()
    
    sess=tf.Session()

    sess.run(tf.global_variables_initializer())
    
    print("IMG :",img)
    
    #img=img.eval(session=sess)
    #img= tf.convert_to_tensor(img)
    t = tf.convert_to_tensor(img, tf.float32, name='t')
    

    print("TENSOR :")
    with tf.Session() as sess:
        t = sess.run(t)
        print(t)
'''
    
    #img = reshape_4d(img)
    
    #img_emb = embedder.embeddings(img)
    
    #print(" result : ",img_emb)