""" Copyright 2023 Lakera AI. All Rights Reserved.

Simple test run example with a random predictor. 
"""

from pathlib import Path

import tempfile

from lakera import DefaultRunnerOptions, Runner

from lakera_fns import download_dataset, DummyGenerator, DummyPredictor


base_dir = Path(f"{tempfile.gettempdir()}/my_mltest_run/")
imgs_dir = base_dir / "data" / "coco_small" / "imgs"
download_url = (
    "https://lakera-public.s3.eu-west-1.amazonaws.com/coco_small.zip"
)

# Download a few example images.
download_dataset(base_dir, imgs_dir, download_url)
print("Number of images:", len(list(imgs_dir.glob("*.jpg"))))

# Initialize the test configuration.
options = DefaultRunnerOptions()

options.set_user_id("your_user_id")
options.set_project_id("quickstart")

# Add an example dataset.
options.add_dataset(
    "dummy_dataset",
    # These two parameters have no specific meaning to MLTest,
    # but can be accessed by your code - see below
    path_to_dataset=f"{base_dir}/data/coco_small/imgs/",
    batch_size=4,
)

# Run your pipeline!
runner = Runner(
    DummyGenerator,
    DummyPredictor,
    options,
    str(base_dir / "mltest_results"),
)
runner_results = runner.run()
