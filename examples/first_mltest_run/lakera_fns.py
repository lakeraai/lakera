""" Copyright 2023 Lakera AI. All Rights Reserved.

Lakera integration classes: DataGenerator and Predictor. 
"""

import glob
import os
from typing import Generator, Tuple

import numpy as np
from PIL import Image
import subprocess

from lakera import (
    Batch,
    DataGenerator,
    Predictor,
    RunnerOptions,
    Sample,
    TargetType,
)


class DummyGenerator(DataGenerator):
    def __init__(self, options: RunnerOptions):
        self.options = options

        # path_to_dataset is a field we set above in RunnerOptions
        data_path = self.options.dataset.path_to_dataset
        self._image_paths = glob.glob(os.path.join(data_path, "*.jpg"))
        self._image_paths.sort()
        # Use just a few images for this example
        self._image_paths = self._image_paths[:12]

    def generate(self) -> Generator[Batch, None, None]:
        batch_size = self.options.dataset.batch_size

        # Use a reproducible RNG
        rng = np.random.default_rng(seed=4)

        for i in range(0, len(self._image_paths), batch_size):
            # A Batch is a list-like container of Samples.
            batch = Batch()

            for image_path in self._image_paths[i : i + batch_size]:
                sample = Sample(
                    target_type=TargetType.BOUNDING_BOX, identifier=image_path
                )

                # Each image has shape [H, W, 3].
                sample.add_input(
                    np.array(Image.open(image_path).convert("RGB"))
                )

                # Add target (ground truth) bounding boxes
                for i in range(3):
                    x_min, y_min, x_max, y_max = get_random_box(sample, rng)
                    sample.add_pascal_target_bbox(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        # Labels can be strings or integers.
                        label=rng.choice(["foo", "bar"]),
                    )

                batch.add_sample(sample)
            yield batch


class DummyPredictor(Predictor):
    def __init__(self, options: RunnerOptions):
        # Use a reproducible RNG
        self.rng = np.random.default_rng(seed=3)

        # You would typically load your model here, but in this example
        # we're just making random predictions - no loading needed.

    def predict(self, batch: Batch) -> Batch:
        """Add predictions to the samples in `batch`, modifying them in-place."""
        for sample in batch:
            # Clear previous predictions in case there are any
            sample.clear_predictions()

            for i in range(2):
                x_min, y_min, x_max, y_max = get_random_box(sample, self.rng)

                sample.add_pascal_prediction_bbox(
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    label=self.rng.choice(["foo", "bar"]),
                    confidence=self.rng.uniform(0.5, 1.0),  # optional
                )

        return batch


def get_random_box(sample: Sample, rng: np.random.Generator) -> Tuple:
    """Returns a random bounding box."""
    height, width = sample.input.shape[:2]

    x1, x2 = rng.choice(
        width, size=2, replace=False
    )  # Two random unique x-coords
    y1, y2 = rng.choice(
        height, size=2, replace=False
    )  # Two random unique y-coords

    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def download_dataset(base_dir: str, imgs_dir: str, download_url: str) -> None:
    """Downoads a few example images if the dataset does not exist."""
    if not imgs_dir.is_dir():
        # The data hasn't been downloaded yet, do it now.

        # Create the directory into which we'll download COCO
        bashCommand = f"mkdir -p {base_dir}/data"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        process.wait()
        # Run the download
        bashCommand = (
            f"curl {download_url} --output {base_dir}/data/coco_small.zip"
        )
        process = subprocess.Popen(
            str(bashCommand).split(), stdout=subprocess.PIPE
        )
        process.wait()
        # Unzip the results
        bashCommand = (
            f"unzip {base_dir}/data/coco_small.zip -d {base_dir}/data/"
        )
        process = subprocess.Popen(
            str(bashCommand).split(), stdout=subprocess.PIPE
        )
        process.wait()

        assert imgs_dir.is_dir(), (
            f"Automatic download failed, please download the data manually"
            f" from {download_url} and extract to {base_dir}."
            f" The directory {imgs_dir} should exist when you're done."
        )
