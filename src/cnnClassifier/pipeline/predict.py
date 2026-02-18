import numpy as np 
import tensorflow
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image  
import os 

print("Libraries imported ")
class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("artifacts","training","model.keras"))

        imagename = self.filename
        test_image = image.load_img(imagename,target_size=(224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = np.argmax(model.predict(test_image),axis=1)

        print(f"Result : {result}")

        
        # Alphabetical order assigned by Keras flow_from_directory
        class_map = {
            0: "Coccidiosis",
            1: "Healthy",
            2: "New Castle Disease",
            3: "Salmonella"
        }

        prediction = class_map.get(result[0], "Unknown")
        return [{"image": prediction}]