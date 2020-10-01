"""ImageModel class."""

import numpy as np
from typing import Tuple

from models.project_model import ProjectModel
from util.util import normalize_images


class ImageModel(ProjectModel):
    """Represents an ML model used on images."""

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Returns the prediction and confidence on a single image.
        :param image: a single raw (non-normalized) image.
        :return: the predicted class name and the confidence.
        """
        norm_image = normalize_images(image)
        pred_raw = self.network.predict(np.expand_dims(norm_image, 0),
                                        batch_size=1).flatten()
        i = np.argmax(pred_raw)
        confidence = pred_raw[i]
        pred_class = self.dataset.label_to_classname[i]
        return pred_class, confidence
