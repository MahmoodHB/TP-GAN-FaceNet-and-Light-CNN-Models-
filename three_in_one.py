import os
import cv2
import face_alignment #exract
import pickle  #save data
import numpy as np

src_dataset_dir = 'D:/DLtpgan_with_facenet/dataset/dataset/'   # dataset
dest_dataset_dir = 'D:/DLtpgan_with_facenet/dataset/landmark'     # exract landmark
#jpg_dataset_dir = 'E:/tpgan_keras(for FEI)/FEI dataset(gray_img)'     # png2jpeg
out_dir = 'D:/DLtpgan_with_facenet/'                       # landmark in one

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


#MOOD = 'GRAY' 
MOOD = 'RGB'

if __name__ == '__main__':
    #comput
    Not_img = 0
    success = 0
          
    img_size = 160
    #curdir = os.getcwd()  #print working pathway    
    count={} #count    
    out_dict = {} #landmark in one    
    os.chdir(src_dataset_dir)  #change working pathway        
    subjects = os.listdir('.')    
    k=0    
    #subjects = subjects[0:90]
    
    for subject in subjects:  #讀取身分
        #change pathway
        os.chdir(src_dataset_dir)
        print(subject)  
        
        #count
        i=0
        # landmark in one
        out_dict[subject] = {}
        
        # exract landmark
        images = os.listdir(subject)
        out_subject_dir = os.path.join(dest_dataset_dir,subject)
        os.makedirs(out_subject_dir, exist_ok=True) #create folder
        src_subject_dir = os.path.join(src_dataset_dir,subject)
       
        # png2jpeg
       # out_jpg_subject_dir = os.path.join(jpg_dataset_dir,subject)
       # os.makedirs(out_jpg_subject_dir, exist_ok=True)
           
        for image in images: #讀取圖片
            i=i+1
            #change pathway
            os.chdir(src_subject_dir)
            
            
            out_image = os.path.join(out_subject_dir, os.path.splitext(image)[0] + '.pkl')
            
            # read_img
            src_image = image[:-4]
            src_jpg_path = os.path.join(src_subject_dir, src_image + '.Jpg')
            img =cv2.imread(src_jpg_path)
            
            if img is not None:
            #face deteced
                    
                    print(subject + " " + image + " : Ok")
                    success = success + 1
                                       
                    face_im = cv2.resize(img, (128, 128))
                
                    # exract landmark
                    img_path = os.path.join(src_subject_dir,image)
                          
                    landmarks = fa.get_landmarks(cv2.cvtColor(face_im,cv2.COLOR_BGR2RGB))
                    
                    #png2jpg
                    #dest_jpg_path = os.path.join(out_jpg_subject_dir, src_image + '.jpg')
                    '''
                    if MOOD =='GRAY':
                        gray_face_im = cv2.cvtColor(face_im,cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(dest_jpg_path, gray_face_im)
                        
                    elif MOOD == 'RGB':
                        cv2.imwrite(dest_jpg_path, face_im)
                    '''
                    if landmarks is None:
                        print("No face detected: {}".format(img_path))
                        continue
                    
                    #for i in range(landmarks):
                        #print('landmarks(i)',landmarks[0])
                        
                    with open(out_image, 'wb') as f:        
                        pickle.dump(landmarks[0], f)
                    
                    # landmark in one
                    with open(out_image, 'rb') as f:
                        landmark_mat = pickle.load(f)
                   
                    #landmark_mat[i][0] [i][1]
                    out_dict[subject][src_image] = landmark_mat.astype(np.uint16)

            else:
                print(subject + " " + image + " : No Image detected")
                Not_img = Not_img + 1
        count[subject] = i
        
    # landmark in one                
    os.chdir(out_dir)
    
    with open('landmarks_file.pkl', 'wb') as f:
        pickle.dump(out_dict, f)  #save pkl
        
    
    print("count: ",count)
    print("File is not image : ", Not_img)
    print("Success get landmark : ", success)

    