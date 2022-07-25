from tpgan import TPGAN, multipie_gen
from keras.optimizers import SGD 
from keras.optimizers import Adam

import time
import multiprocessing  
from numba import prange
time_total =[]
time1=time.time()


if __name__ == '__main__':
    
    #K.clear_session()
    op = 'Adam'

    gan = TPGAN(base_filters=64, gpus=1,
                #facenet_weights='E:/DLtpgan_with_facenet/facenet_model/20180408-102900.pb',
                facenet_weights='D:/DLtpgan_with_facenet/facenet_keras_weights.h5',
                generator_weights='',
                classifier_weights='',   
                discriminator_weights='')
    
    datagen = multipie_gen.Datagen(dataset_dir='D:/DLtpgan_with_facenet/dataset', landmarks_dict_file='D:/DLtpgan_with_facenet/landmarks_file.pkl', 
                                   datalist_dir='D:/DLtpgan_with_facenet/datalist/datalist_side_face.pkl', min_angle=-90, max_angle=90, valid_count=360)

    if op == 'Adam':
        optimizer = Adam(lr=0.0001, beta_1=0.9)#, beta_2=0.999 # n=4
    elif op == 'SGD':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True) # n=2
                  
    #print('gan.discriminator():',gan.discriminator().output)
    gan.train_gan(gen_datagen_creator=datagen.get_generator, 
                  gen_train_batch_size=4,
                  gen_valid_batch_size=4,  
                  disc_datagen_creator=datagen.get_discriminator_generator, 
                  disc_batch_size=10, 
                  disc_gt_shape=gan.discriminator().output_shape[1:3],
                  optimizer=optimizer,
                  gen_steps_per_epoch=100, disc_steps_per_epoch=100,  
                  epochs=3200, out_dir='D:/DLtpgan_with_facenet/out_dir/', out_period=100, is_output_img=True,
                  lr=0.0001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1,
                  lambda_sym=1e-1, lambda_ip=1e-3, lambda_adv=5e-3, lambda_tv=1e-5,
                  lambda_class=1, lambda_parts=3)

    print ('Time')
    time111 = time.time()-time1
    print (('Time', time111))
    time_total.append (time111)

#    if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool.map(1, range(0,10000))
    pool.close()
