import os 
import random
import shutil

data_dir = "C://Users/mak44/Desktop/real_and_fake_face"
train_real_dir = "C://Users/mak44/Desktop/splitted_dataset/train/real"
train_fake_dir = "C://Users/mak44/Desktop/splitted_dataset/train/fake"

test_real_dir = "C://Users/mak44/Desktop/splitted_dataset/test/real"
test_fake_dir = "C://Users/mak44/Desktop/splitted_dataset/test/fake"

real_faces_dir = os.path.join(data_dir,"training_real")
fake_faces_dir = os.path.join(data_dir,"training_fake")

train_ratio = 0.8

os.makedirs(train_real_dir,exist_ok=True)
os.makedirs(train_fake_dir,exist_ok=True)

os.makedirs(test_real_dir,exist_ok=True)
os.makedirs(test_fake_dir,exist_ok=True)


for filename in os.listdir(real_faces_dir):
    file_path = os.path.join(real_faces_dir,filename)
    
    if random.random() < train_ratio :
        shutil.copy(file_path,train_real_dir)
    
    else :
        shutil.copy(file_path,test_real_dir)
        
for filename in os.listdir(fake_faces_dir):
    file_path = os.path.join(fake_faces_dir, filename)
    
    if random.random() < train_ratio:
        shutil.copy(file_path, train_fake_dir)
        
    else :
        shutil.copy(file_path, test_fake_dir)
        