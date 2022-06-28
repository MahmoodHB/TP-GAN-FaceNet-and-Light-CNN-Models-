import os
import pickle

src_dataset_dir = 'E:/DLtpgan_with_facenet/dataset/dataset' 
#src_dataset_dir = 'D:/desktop/tpgan_keras/dataset_png' 
dest_dataset_dir = 'E:/DLtpgan_with_facenet'
#dest_dataset_dir = 'D:/desktop/tpgan_keras/out_data'

os.chdir(src_dataset_dir) #change pathway

subjects = os.listdir('.')
#   session, subject, number, image
  
out_dict = []

for subject in subjects:
    os.chdir(src_dataset_dir)
    
    images = os.listdir(subject)

    for image in images:
        src_number_dir = os.path.join(src_dataset_dir, subject)
        os.chdir(src_number_dir)
        
        
        print(subject + " " + image)
        data_name = os.path.basename(image)[:-4]
        data_path = os.path.join(subject, data_name)
        out_dict.append(data_path)
            
    
os.chdir(dest_dataset_dir)

with open('datalist_all.pkl', 'wb') as f:
    pickle.dump(out_dict, f)  #save pkl