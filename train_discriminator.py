#from keras_tpgan.tpgan import TPGAN, multipie_gen
from tpgan import TPGAN, multipie_gen

#from tensorflow.contrib.training.python.training import hparam
 

if __name__ == '__main__':
    #train_discriminator
    gan = TPGAN(base_filters=64, gpus=1,
                lcnn_extractor_weights='',#''
                generator_weights='',#''
                classifier_weights='',#''
                discriminator_weights='')#''

    datagen = multipie_gen.Datagen(dataset_dir='', landmarks_dict_file='', 
                                   datalist_dir='', valid_count=4)
    train_gen = datagen.get_discriminator_generator(gan.generator(), batch_size=64,
                                                    gt_shape=gan.discriminator().output_shape[1:3],
                                                    setting = 'train')
    
    gan.train_discriminator(train_gen=train_gen, valid_gen=train_gen, steps_per_epoch=300,
                                      epochs=100, out_dir='', out_period=5)
    '''
    out_dir   :out dir for model, images, log
    out_period:interval epoch count for model and images
    job-dir   :GCS location to write checkpoints and export models
    '''
    
