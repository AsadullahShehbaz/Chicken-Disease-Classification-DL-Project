from cnnClassifier.config.configuration import EvaluationConfig
from cnnClassifier.utils.common import *
from pathlib import Path 
import tensorflow as tf 

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        # Store evaluation config (model path, image size, batch size, data path, etc.)
        self.config = config
    
    def _valid_generator(self):
        # Normalize pixel values [0,1] and reserve 30% of data for validation
        datagenerator_kwargs = dict(rescale=1./255, validation_split=0.30)

        # Resize images to model's expected input (drops channel dim) and set batch size
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # e.g. (224,224,3) -> (224,224)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"  # smooth resizing via bilinear interpolation
        )

        valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Flow images from directory using only the validation subset; shuffle=False for consistent eval
        self.valid_generator = valid_generator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,  # important: keep order fixed during evaluation
            **dataflow_kwargs
        )

    @staticmethod 
    def load_model(path: Path) -> tf.keras.Model:
        # Load saved Keras model (supports SavedModel and .keras formats)
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        # Persist loss and accuracy to scores.json for experiment tracking
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)