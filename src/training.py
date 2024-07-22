# from utils.common import read_config
# import argparse
# import os
# from utils.dataloader import tf_dataset
# from utils.dataloader import get_data
# from utils.model import my_model
# from utils.model import save_model
# from utils.callbacks import get_callbacks
# import logging

# #model_dir = r"/home/agrograde/Desktop/19th_march/onion_segmentation_AIops/models"

# logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
# log_dir = "training_logs"
# os.makedirs(log_dir,exist_ok=True)
# logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format = logging_str,filemode="a")



# def training(config_path):
#     config = read_config(config_path)
#     model_new = my_model()
#     train_data,val_data,test_data = get_data()

#     CALLBACKS_LIST = get_callbacks(config)
#     EPOCHS = config["parameters"]["epochs"]
#     BATCH_SIZE = config["parameters"]["batch_size"]    
#     try:
#         logging.info(">>>>>>>>>>>>>> TRAINING STARTING <<<<<<<<<<<<<<<")
#         H = model_new.fit(train_data,
#                     epochs=EPOCHS,
#                     validation_data=val_data,
#                     callbacks = CALLBACKS_LIST,
#                     batch_size=BATCH_SIZE)
#         logging.info(">>>>>>>>>>>>> TRAINING DONE SUCCESFULLY <<<<<<<<<<<<<<")
#         for epoch in range(EPOCHS):
#             logging.info(f"Epoch {epoch + 1} - Training Metrics:")
#             for metric_name, metric_values in H.history.items():
#                 logging.info(f"{metric_name}: {metric_values[epoch]}")
#     except Exception as e:
#         logging.exception(e)
    
        
    
    
#     artifacts_dir = config["artifacts"]["artifacts_dir"]
#     model_dir = config["artifacts"]["model_dir"]

    
#     model_dir_path = os.path.join(artifacts_dir,model_dir)
#     os.makedirs(model_dir_path,exist_ok=True)

#     model_name = config["artifacts"]["model_name"]

#     save_model(model_new,model_name,model_dir_path)

#     # EVALUATION
#     logging.info(">>>>>>>>>>> EVALUATION AND TEST MODEL <<<<<<<<<<<")
#     evaluation_metrics= model_new.evaluate(test_data,batch_size = 4)
#     # Log evaluation metrics
#     logging.info("Evaluation Metrics:")
#     for metric_name, metric_value in zip(model_new.metrics_names, evaluation_metrics):
#         logging.info(f"{metric_name}: {metric_value}")



#     #print(new_model1.summary())

#    # print(config)
#     #print("training data collected successfully")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", "-c", default="config.yaml", type=str, required=True, help="/home/agrograde/Desktop/19th_march/onion_segmentation_AIops/config.yaml")
#     parsed_args = parser.parse_args()

#     training(config_path=parsed_args.config)


### training command    "python src/training.py --config /home/agrograde/Desktop/19th_march/onion_segmentation_AIops/config.yaml"
from utils.common import read_config
import argparse
import os
from utils.dataloader import tf_dataset
from utils.dataloader import get_data
from utils.model import my_model
from utils.model import save_model
from utils.callbacks import get_callbacks
import logging

#model_dir = r"/home/agrograde/Desktop/19th_march/onion_segmentation_AIops/models"
"""Here we are printing the logs at athe time of training

"""

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "training_logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO,format = logging_str,filemode="a")

"""Defining the training function"""

def training(config_path):
    config = read_config(config_path)
    model_new = my_model()
    train_data,val_data = get_data()

    CALLBACKS_LIST = get_callbacks(config)
    EPOCHS = config["parameters"]["epochs"]
    BATCH_SIZE = config["parameters"]["batch_size"]    
    try:
        logging.info(">>>>>>>>>>>>>> TRAINING STARTING <<<<<<<<<<<<<<<")
        H = model_new.fit(train_data,
                    epochs=EPOCHS,
                    validation_data=val_data,
                    callbacks = CALLBACKS_LIST,
                    batch_size=BATCH_SIZE)
        logging.info(">>>>>>>>>>>>> TRAINING DONE SUCCESFULLY <<<<<<<<<<<<<<")

        for epoch in range(EPOCHS):
            logging.info(f"Epoch {epoch + 1} - Training Metrics:")
            for metric_name, metric_values in H.history.items():
                logging.info(f"{metric_name}: {metric_values[epoch]}")
         # EVALUATION
       
    except Exception as e:
        logging.exception(e)
    
        
    
    
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    
    model_dir_path = os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path,exist_ok=True)

    model_name = config["artifacts"]["model_name"]

    save_model(model_new,model_name,model_dir_path)

   

    ## Quantization
    




    # logging.info(">>>>>>>>>>> EVALUATION AND TEST MODEL <<<<<<<<<<<")
    # evaluation_metrics= model_new.evaluate(test_data,batch_size = BATCH_SIZE)
    # #Log evaluation metrics
    # logging.info("Evaluation Metrics:")
    # for metric_name, metric_value in zip(model_new.metrics_names, evaluation_metrics):
    #     logging.info(f"{metric_name}: {metric_value}")



    #print(new_model1.summary())

   # print(config)
    #print("training data collected successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", type=str, required=True, help="/home/agrograde/Desktop/19th_march/onion_segmentation_AIops/config.yaml")
    parsed_args = parser.parse_args()
    """ Calling the training function completely to execute training """
    training(config_path=parsed_args.config)



