from tpgan import TPGAN, multipie_gen
from keras.optimizers import Adam
from keras.optimizers import SGD

if __name__ == '__main__':
    
    #op = 'SGD'
    op = 'Adam'
    
    gan = TPGAN(base_filters=64, gpus=1,
                lcnn_extractor_weights='',
                generator_weights='',
                classifier_weights='',
                discriminator_weights='')
    
    datagen = multipie_gen.Datagen(dataset_dir='', landmarks_dict_file='', 
                                   datalist_dir='', min_angle=-30, max_angle=30, valid_count=4)
    
    train_gen = datagen.get_generator(setting='train', batch_size=8)
    valid_gen = datagen.get_generator(setting='valid', batch_size=4)

    if op == 'Adam':
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # n=4
    elif op == 'SGD':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True) # n=2
    
    gan.train_gan(gen_datagen_creator=datagen.get_generator, 
                  gen_train_batch_size=4, #5 - 8
                  gen_valid_batch_size=4,
                  disc_datagen_creator=datagen.get_discriminator_generator, 
                  disc_batch_size=10, #8 - 11
                  disc_gt_shape=gan.discriminator().output_shape[1:3],
                  optimizer=optimizer,
                  gen_steps_per_epoch=10, disc_steps_per_epoch=10,  
                  epochs=300, out_dir='path to output files', out_period=1, is_output_img=True,
                  lr=0.0001, decay=0, lambda_128=1, lambda_Consistency128=0.011, lambda_ip=1e-3, lambda_adv=5e-3, lambda_tv=1e-5, lambda_content=0.5, lambda_parts=3)