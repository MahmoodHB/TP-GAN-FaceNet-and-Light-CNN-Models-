import pickle

#src_dataset_dir = 'E:/tpgan-chinese/tpgan_keras(for chinese)/crop_img(jpg)' 
#src_dataset_dir = 'D:/desktop/tpgan_keras/dataset_png' 
#datalist_dir = 'E:/tpgan-chinese/tpgan_keras(for chinese)/datalist_test.pkl'
datalist_dir = 'E:\DLtpgan_with_facenet'

with open(datalist_dir, 'rb') as f:
    datalist = pickle.load(f)
    
