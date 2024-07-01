# import numpy as np
# import cv2
# import os
# from tqdm import tqdm as tqdm
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras.losses import binary_crossentropy
# from keras import backend as K
# import segmentation_models as sm
# # from keras.utils import to_categorical
# from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.utils import generic_utils
# from segmentation_models.losses import CategoricalFocalLoss
# from tensorflow.keras.optimizers import Adam,SGD
# from tensorflow.keras.utils import to_categorical
# from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
# # from keras.optimizers import Adam,SGD
# from keras.losses import BinaryCrossentropy
# from keras.metrics import MeanIoU
# from glob import glob


# #image_path = '/home/agrograde/Desktop/summer_onion/image_path'
# image_path = "/home/agrograde/Desktop/4th_mar/test_image_path"

# def preprocessMasks(mask,height,width):
#     mask_resized = cv2.threshold(cv2.resize(mask, (height,width)), 50, 1, cv2.THRESH_BINARY)[1]
#     mask_data = np.zeros((height,width,2))
    
#     for i in range(height):
#         for j in range(width):


#             #for segmentation mask
#             if mask_resized[i,j]> 0:
#                 mask_data[i,j,1] = 1
#             else:
#                 mask_data[i,j,0] = 1
                
#     return mask_data       #output from the function(height, width, 2)

# def load_data(path):
#     images = sorted(glob(os.path.join(image_path, 'images/*')))
#     sprout = sorted(glob(os.path.join(image_path, 'Sprout/*')))
#     peeled = sorted(glob(os.path.join(image_path, 'Peeled/*')))
#     rotten = sorted(glob(os.path.join(image_path, 'Rotten/*')))
#     double = sorted(glob(os.path.join(image_path, 'Double/*')))
#     background = sorted(glob(os.path.join(image_path, 'Background/*')))
#     return images,sprout,peeled,rotten,double,background
    

# images,sprout,peeled,rotten,double,background = load_data(image_path)
# def read_image(path):
    
#     x = cv2.imread(path)
#     x = cv2.resize(x, (224, 224))
#     x = x.astype(np.float32)
#     return x
# def read_mask(path):
#     x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     x = preprocessMasks(x, 224, 224)
#     x = x.astype(np.float32)
#     return x
    
    
# def preprocess(x,y1,y2,y3,y4,y5):
#     def f(x,y1,y2,y3,y4,y5):
#         x = x.decode()
#         y1 = y1.decode()
#         y2 = y2.decode()
#         y3 = y3.decode()
#         y4 = y4.decode()
#         y5 = y5.decode()
        
#         x = read_image(x)
#         y1 = read_mask(y1)
#         y2 = read_mask(y2)
#         y3 = read_mask(y3)
#         y4 = read_mask(y4)
#         y5 = read_mask(y5)
        
#         return x,y1,y2,y3,y4,y5
    
#     images, sprout, peeled, rotten,double,background = tf.numpy_function(f, [x,y1,y2,y3,y4,y5], [tf.float32, tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
#     images.set_shape([224, 224, 3])
#     sprout.set_shape([224, 224, 2])
#     peeled.set_shape([224, 224, 2])
#     rotten.set_shape([224, 224, 2])
#     double.set_shape([224, 224, 2])
#     background.set_shape([224, 224, 2])

#     return images, sprout, peeled, rotten,double, background





# def tf_dataset(x, y1, y2, y3, y4, y5, batch_size, train_split=0.8, val_split=0.1,test_split=0.1):
#     # Combine inputs and targets into a single dataset
#     dataset = tf.data.Dataset.from_tensor_slices((x, y1, y2, y3, y4, y5))
    
#     # Shuffle the dataset with a buffer size of 1000
#     dataset = dataset.shuffle(buffer_size=1000)
    
#     # Apply any necessary preprocessing to each sample (e.g., normalization, resizing)
#     dataset = dataset.map(preprocess)
    
#     # Calculate the number of samples for each split
#     total_samples = len(x)
#     train_samples = int(total_samples * train_split)
#     val_samples = int(total_samples * val_split)
#     test_samples = int(total_samples * test_split)
#     # Split dataset into training and validation sets
#     train_dataset = dataset.take(train_samples)
#     val_dataset = dataset.skip(train_samples).take(val_samples)
#     test_dataset = dataset.skip(train_samples + val_samples).take(test_samples)

#     print("Total samples:", total_samples)
#     print("Train dataset size:", train_samples)
#     print("Validation dataset size:", val_samples)
#     print("Test dataset size:", test_samples)

#     # Split inputs and targets into separate datasets
#     train_images = train_dataset.map(lambda x, *y: x)   
#     train_sprout = train_dataset.map(lambda x, *y: y[0])
#     train_peeled = train_dataset.map(lambda x, *y: y[1])
#     train_rotten = train_dataset.map(lambda x, *y: y[2])
#     train_double = train_dataset.map(lambda x, *y: y[3])
#     train_background = train_dataset.map(lambda x, *y: y[4])

#     # Test
#     test_images = test_dataset.map(lambda x, *y: x)
#     test_sprout = test_dataset.map(lambda x, *y: y[0])
#     test_peeled = test_dataset.map(lambda x, *y: y[1])
#     test_rotten = test_dataset.map(lambda x, *y: y[2])
#     test_double = test_dataset.map(lambda x, *y: y[3])
#     test_background = test_dataset.map(lambda x, *y: y[4])
    
#     # For validation dataset
#     val_images = val_dataset.map(lambda x, *y: x)
#     val_sprout = val_dataset.map(lambda x, *y: y[0])
#     val_peeled = val_dataset.map(lambda x, *y: y[1])
#     val_rotten = val_dataset.map(lambda x, *y: y[2])
#     val_double = val_dataset.map(lambda x, *y: y[3])
#     val_background = val_dataset.map(lambda x, *y: y[4])
    
#     # Batch the datasets
#     train_images = train_images.batch(batch_size)
#     train_sprout = train_sprout.batch(batch_size)
#     train_peeled = train_peeled.batch(batch_size)
#     train_rotten = train_rotten.batch(batch_size)
#     train_double = train_double.batch(batch_size)
#     train_background = train_background.batch(batch_size)

#     test_images = test_images.batch(batch_size)
#     test_sprout = test_sprout.batch(batch_size)
#     test_peeled = test_peeled.batch(batch_size)
#     test_rotten = test_rotten.batch(batch_size)
#     test_double = test_double
#     test_background = test_background.batch(batch_size)
    
#     val_images = val_images.batch(batch_size)
#     val_sprout = val_sprout.batch(batch_size)
#     val_peeled = val_peeled.batch(batch_size)
#     val_rotten = val_rotten.batch(batch_size)
#     val_double = val_double.batch(batch_size)
#     val_background = val_background.batch(batch_size)
    
#     # Prefetch data for improved performance
#     train_images = train_images.prefetch(buffer_size=tf.data.AUTOTUNE)
#     train_sprout = train_sprout.prefetch(buffer_size=tf.data.AUTOTUNE)
#     train_peeled = train_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
#     train_rotten = train_rotten.prefetch(buffer_size=tf.data.AUTOTUNE)
#     train_double = train_double.prefetch(buffer_size=tf.data.AUTOTUNE)
#     train_background = train_background.prefetch(buffer_size=tf.data.AUTOTUNE)

#     test_images = test_images.prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_sprout = test_sprout.prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_peeled = test_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_rotten = test_rotten.prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_double = test_double.prefetch(buffer_size=tf.data.AUTOTUNE)
#     test_background = test_background.prefetch(buffer_size=tf.data.AUTOTUNE)
        
#     val_images = val_images.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_sprout = val_sprout.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_peeled = val_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_rotten = val_rotten.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_double = val_double.prefetch(buffer_size=tf.data.AUTOTUNE)
#     val_background = val_background.prefetch(buffer_size=tf.data.AUTOTUNE)
    
#     # return (train_images, train_sprout, train_peeled, train_rotten, train_double, train_background), \
#     #        (val_images, val_sprout, val_peeled, val_rotten, val_double, val_background)

#     return (train_images, train_sprout, train_peeled, train_rotten, train_double, train_background), \
#          (val_images, val_sprout, val_peeled, val_rotten, val_double, val_background), \
#          (test_images, test_sprout, test_peeled, test_rotten, test_double, test_background)


# #images,sprout,peeled,rotten,double,background = load_data(image_path)

# def get_data():
#     images,sprout,peeled,rotten,double,background = load_data(image_path)
#     (train_images, train_sprout, train_peeled, train_rotten, train_double, train_background),(val_images, val_sprout, val_peeled, val_rotten, val_double, val_background),(test_images, test_sprout, test_peeled, test_rotten, test_double, test_background) = tf_dataset(images, sprout, peeled, rotten, double, background, batch_size=16, 
#                 train_split=0.8, val_split=0.1,test_split=0.1)

#     labels_dict = {
#         "sprout": train_sprout,
#         "peeled": train_peeled,
#         "rotten": train_rotten,
#         "double": train_double,
#         "background": train_background
#     }
#     val_labels_dict = {
#         "sprout": val_sprout,
#         "peeled": val_peeled,
#         "rotten": val_rotten,
#         "double": val_double,
#         "background": val_background
#     }  
#     test_labels_dict = {
#         "sprout": test_sprout,
#         "peeled": test_peeled,
#         "rotten": test_rotten,
#         "double": test_double,
#         "background": test_background
#     }

#     test_data = tf.data.Dataset.zip((test_images, test_labels_dict)) 
#     val_data = tf.data.Dataset.zip((val_images, val_labels_dict)) 
#     # Zip images and label dictionary into a single dataset
#     train_data = tf.data.Dataset.zip((train_images, labels_dict))
#    # print("succesfully loaded the train and valid data")
#     return train_data,val_data,test_data


## fpr 2 classes


import numpy as np
import cv2
import os
from tqdm import tqdm as tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy
from keras import backend as K
import segmentation_models as sm
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils import generic_utils
from segmentation_models.losses import CategoricalFocalLoss
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
# from keras.optimizers import Adam,SGD
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from glob import glob
#from utils.common import read_config
import argparse

#image_path = '/home/agrograde/Desktop/summer_onion/image_path'
image_path = "/home/agrograde/Desktop/12th_june/old_summer_onion/image_path"

def preprocessMasks(mask,height,width):
    mask_resized = cv2.threshold(cv2.resize(mask, (height,width)), 50, 1, cv2.THRESH_BINARY)[1]
    mask_data = np.zeros((height,width,2))
    
    for i in range(height):
        for j in range(width):


            #for segmentation mask
            if mask_resized[i,j]> 0:
                mask_data[i,j,1] = 1
            else:
                mask_data[i,j,0] = 1
                
    return mask_data       #output from the function(height, width, 2)

def load_data(path):
    images = sorted(glob(os.path.join(image_path, 'images/*')))
    #sprout = sorted(glob(os.path.join(image_path, 'Sprout/*')))
    peeled = sorted(glob(os.path.join(image_path, 'Peeled/*')))
   # rotten = sorted(glob(os.path.join(image_path, 'Rotten/*')))
    black_smut = sorted(glob(os.path.join(image_path, 'Black_smut/*')))
    background = sorted(glob(os.path.join(image_path, 'Background/*')))
    return images,peeled,black_smut,background
    

images,peeled,black_smut,background = load_data(image_path)
def read_image(path):
    
    x = cv2.imread(path)
    x = cv2.resize(x, (224, 224))
    x = x.astype(np.float32)
    return x
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = preprocessMasks(x, 224, 224)
    x = x.astype(np.float32)
    return x
    
    
def preprocess(x,y1,y2,y3):
    def f(x,y1,y2,y3):
        x = x.decode()
        y1 = y1.decode()
        y2 = y2.decode()
        y3 = y3.decode()
        # y4 = y4.decode()
        # y5 = y5.decode()
        
        x = read_image(x)
        y1 = read_mask(y1)
        y2 = read_mask(y2)
        y3 = read_mask(y3)
        # y4 = read_mask(y4)
        # y5 = read_mask(y5)
        
        return x,y1,y2,y3
    
    images,peeled, black_smut,background = tf.numpy_function(f, [x,y1,y2,y3], [tf.float32, tf.float32,tf.float32,tf.float32])
    images.set_shape([224, 224, 3])
    #sprout.set_shape([224, 224, 2])
    peeled.set_shape([224, 224, 2])
    #rotten.set_shape([224, 224, 2])
    black_smut.set_shape([224, 224, 2])
    background.set_shape([224, 224, 2])

    return images, peeled,black_smut, background



# parser = argparse.ArgumentParser()
# parser.add_argument("--config", "-c", default="config.yaml", type=str, required=True, help="/home/agrograde/Desktop/19th_march/onion_segmentation_AIops/config.yaml")
# parsed_args = parser.parse_args()
# config_path=parsed_args.config
# config = read_config(config_path)
# BATCH_SIZE = config["parameters"]["batch_size"]    



def tf_dataset(x, y1, y2, y3, batch_size, train_split=0.8, val_split=0.1,test_split=0.1):
    # Combine inputs and targets into a single dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y1, y2, y3))
    
    # Shuffle the dataset with a buffer size of 1000
    dataset = dataset.shuffle(buffer_size=1000)
    
    # Apply any necessary preprocessing to each sample (e.g., normalization, resizing)
    dataset = dataset.map(preprocess)
    
    # Calculate the number of samples for each split
    total_samples = len(x)
    train_samples = int(total_samples * train_split)
    val_samples = int(total_samples * val_split)
    test_samples = int(total_samples * test_split)
    # Split dataset into training and validation sets
    train_dataset = dataset.take(train_samples)
    val_dataset = dataset.skip(train_samples).take(val_samples)
    test_dataset = dataset.skip(train_samples + val_samples).take(test_samples)

    print("Total samples:", total_samples)
    print("Train dataset size:", train_samples)
    print("Validation dataset size:", val_samples)
    print("Test dataset size:", test_samples)

    # Split inputs and targets into separate datasets
    train_images = train_dataset.map(lambda x, *y: x)   
    train_peeled = train_dataset.map(lambda x, *y: y[0])
    train_black_smut = train_dataset.map(lambda x, *y: y[1])
    train_background = train_dataset.map(lambda x, *y: y[2])
   

    # Test
    test_images = test_dataset.map(lambda x, *y: x)
    test_peeled = test_dataset.map(lambda x, *y: y[0])
    test_black_smut = test_dataset.map(lambda x, *y: y[1])
    test_background = test_dataset.map(lambda x, *y: y[2])
  
    # For validation dataset
    val_images = val_dataset.map(lambda x, *y: x)
    val_peeled = val_dataset.map(lambda x, *y: y[0])
    val_black_smut = val_dataset.map(lambda x, *y: y[1])
    val_background = val_dataset.map(lambda x, *y: y[2])
    
    # Batch the datasets
    train_images = train_images.batch(batch_size)
    train_peeled = train_peeled.batch(batch_size)
    train_black_smut = train_black_smut.batch(batch_size)
    train_background = train_background.batch(batch_size)


    test_images = test_images.batch(batch_size)
    test_peeled = test_peeled.batch(batch_size)
    test_black_smut = test_black_smut.batch(batch_size)
    test_background = test_background.batch(batch_size)
    
    val_images = val_images.batch(batch_size)
    val_peeled = val_peeled.batch(batch_size)
    val_black_smut = val_black_smut.batch(batch_size)
    val_background = val_background.batch(batch_size)
   
    # Prefetch data for improved performance
    train_images = train_images.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_peeled = train_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_black_smut = train_black_smut.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_background = train_background.prefetch(buffer_size=tf.data.AUTOTUNE)


    test_images = test_images.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_peeled = test_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_black_smut = test_black_smut.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_background = test_background.prefetch(buffer_size=tf.data.AUTOTUNE)
    
        
    val_images = val_images.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_peeled = val_peeled.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_black_smut = val_black_smut.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_background = val_background.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    # return (train_images, train_sprout, train_peeled, train_rotten, train_double, train_background), \
    #        (val_images, val_sprout, val_peeled, val_rotten, val_double, val_background)

    return (train_images, train_peeled, train_black_smut, train_background), \
         (val_images, val_peeled, val_black_smut, val_background), \
         (test_images, test_peeled, test_black_smut, test_background)


#images,sprout,peeled,rotten,double,background = load_data(image_path)

def get_data():
    images,peeled,black_smut,background = load_data(image_path)
    (train_images, train_peeled, train_black_smut, train_background),(val_images, val_peeled, val_black_smut, val_background),(test_images, test_peeled, test_black_smut, test_background) = tf_dataset(images, peeled, black_smut, background, batch_size=8, 
                train_split=0.8, val_split=0.1,test_split=0.1)

    labels_dict = {
        "peeled": train_peeled,
        "black_smut": train_black_smut,
        "background": train_background
    }
    val_labels_dict = {
        "peeled": val_peeled,
        "black_smut": val_black_smut,
        "background": val_background
    }  
    test_labels_dict = {
        "peeled": test_peeled,
        "black_smut": test_black_smut,
        "background": test_background
    }

    test_data = tf.data.Dataset.zip((test_images, test_labels_dict)) 
    val_data = tf.data.Dataset.zip((val_images, val_labels_dict)) 
    # Zip images and label dictionary into a single dataset
    train_data = tf.data.Dataset.zip((train_images, labels_dict))
   # print("succesfully loaded the train and valid data")
    return train_data,val_data,test_data



## Data Visualization
import matplotlib.pyplot as plt

def visualize_batch(dataset, num_images=4):
    """
    Visualize a batch of images and their corresponding masks.

    Args:
    - dataset: The dataset to visualize data from.
    - num_images: The number of images to display.
    """
    # Take one batch from the dataset
    for images, masks in dataset.take(1):
        # Get the batch size (number of images in the batch)
        batch_size = images.shape[0]

        # Set the number of images to display
        num_images = min(num_images, batch_size)

        # Create a figure to display the images and masks
        fig, axes = plt.subplots(num_images, 4, figsize=(15, 5 * num_images))

        for i in range(num_images):
            # Display the image
            axes[i, 0].imshow(images[i].numpy().astype("uint8"))
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            # Display the peeled mask
            axes[i, 1].imshow(masks["peeled"][i].numpy()[:, :, 1], cmap="gray")
            axes[i, 1].set_title("Peeled Mask")
            axes[i, 1].axis("off")

            # Display the black smut mask
            axes[i, 2].imshow(masks["black_smut"][i].numpy()[:, :, 1], cmap="gray")
            axes[i, 2].set_title("Black Smut Mask")
            axes[i, 2].axis("off")

            # Display the background mask
            axes[i, 3].imshow(masks["background"][i].numpy()[:, :, 1], cmap="gray")
            axes[i, 3].set_title("Background Mask")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.show()

# Get the datasets
train_data, val_data, test_data = get_data()

# Visualize a batch of data from the training dataset
visualize_batch(train_data)
