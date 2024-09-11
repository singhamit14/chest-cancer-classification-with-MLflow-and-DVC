# import numpy as np
# import keras
# import os

# class PredictionPipeline:
#     def __init__(self, filename):
#         self.filename = filename

#     def predict(self):
#         # Load the model
#         model_path = os.path.join("model", "model.h5")
#         model = keras.models.load_model(model_path, compile=False)

#         # Check if model is loaded correctly
#         if model is None:
#             print("Error: Model not loaded.")
#             return [{"image": "Error loading model"}]
        
#         print("Model loaded:", model)
#         print("Type of model:", type(model))

#         # Load and preprocess the image
#         imagename = self.filename
#         try:
#             test_image = keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
#             test_image = keras.preprocessing.image.img_to_array(test_image)
#             test_image = np.expand_dims(test_image, axis=0)
#         except Exception as e:
#             print("Error loading or processing image:", e)
#             return [{"image": "Error processing image"}]
        
#         # Predict the result
#         try:
#             result = np.argmax(model.predict(test_image), axis=1)
#             print("Prediction result:", result)
#             if result[0] == 1:
#                 prediction = 'Normal'
#             else:
#                 prediction = 'Adenocarcinoma Cancer'
#             return [{"image": prediction}]
#         except Exception as e:
#             print("Error during prediction:", e)
#             return [{"image": "Error during prediction"}]




import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]
