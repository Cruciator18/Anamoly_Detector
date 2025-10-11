import torch
import os
import torch.nn as nn
from PIL import Image
from torchvision import datasets , transforms
from torch.utils.data import DataLoader, Dataset



root_dir = r"mvtec_anomaly_detection"
category = r"metal_nut"
BATCH_SIZE = 32





#Loading the transformed data

class MVTecDataset(Dataset):
    def __init__(self , root_dir ,classname, split ='train', transform =None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.split = split  
        self.classname = classname
        
        self.data_path = os.path.join(root_dir , classname ,split)
        self.image_paths = []
        self.labels = []
        
        
        if self.split == 'train':
            good_folder = os.path.join(self.data_path , 'good')
            image_files = os.listdir(good_folder)
            self.image_paths = [os.path.join(good_folder , f) for f in image_files]
            self.labels = [0] * len(self.image_paths)
        
        
        else:
            subfolders = os.listdir(self.data_path)
            for subfolder in subfolders:
                
                folder_path = os.path.join(self.data_path , subfolder)
                if os.path.isdir(folder_path):
                    image_files = os.listdir(folder_path)
                    for f in image_files :
                        self.image_paths.append (os.path.join(folder_path , f))
                        self.labels.append(0 if subfolder == 'good' else 1) 
                    
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        
        return image,label
    
    
#Define the transforms that your data will undergo , RandomFlipping, Resizing, Conversion to tensors, etc

transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalise according to the 
    ]
)  


train_dataset = MVTecDataset(
    root_dir=root_dir,
    classname=category,
    split='train',
    transform=transform
)

test_dataset = MVTecDataset(
    root_dir=root_dir,
    classname=category,
    split='test',
    transform=transform
)  
   
print(f"Successfully loaded the '{category}' dataset using the custom loader.")
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images: {len(test_dataset)}") 


train_loader = DataLoader(train_dataset , batch_size=BATCH_SIZE ,shuffle=True)
test_loader = DataLoader(test_dataset , batch_size=BATCH_SIZE , shuffle=False)

images , labels = next(iter(train_loader))
print(f"Shape of one batch:{images.shape}")
print(f"Labels in the batch :{labels}")