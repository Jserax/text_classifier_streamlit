import bentoml
from bentoml.io import Text, NumpyNdarray
import numpy as np
import keras_nlp

runner = bentoml.keras.get("my_keras_model:latest").to_runner()
svc = bentoml.Service(name="my_keras_model", runners=[runner])


@svc.api(input=Text(), output=NumpyNdarray(), route="text_classifier/predict")
def predict(input: str) -> np.ndarray:
    result = runner.run([input])
    return result