<h1 align="center">Lakera</h1>

<p align="center"><b>A platform to build computer vision that doesn't fail.</b></p>

<p align="center">
  <a href="https://lakera.ai" target="_blank">Website</a>
  |
  <a href="https://docs.lakera.ai" target="_blank">Docs</a>
  |
  <a href="https://www.lakera.ai/updates" target="_blank">Newsletter</a>
  | 
  <a href="https://www.lakera.ai/insights" target="_blank">Blog</a>
  | 
  <a href="https://www.linkedin.com/company/lakeraai" target="_blank">LinkedIn</a>
  |
  <a href="https://twitter.com/lakeraai" target="_blank">Twitter</a>
</p>


# :pencil2: What is Lakera's MLTest?

Lakera's MLTest **finds critical performance vulnerabilities** in computer vision systems and allows you to easily fix them before they enter operation.

You can think of MLTest as your trusted **safety expert** that knows how to ensure that your models work as expected once deployed. 

MLTest **completely automates testing** of computer vision systems - something that has previously required expert knowledge and significant resources.

So what are the **main benefits** of using MLTest?

* You automate testing and understand where your models are likely to fail at close to zero overhead.
* You know exactly which issues to focus on, which to fix next, thus significantly reducing iteration time.
* You don't have to set up complex testing infrastructure as MLTest provides that out of the box, meaning you can focus on what really matters and develop faster.
* You can showcase progress to team members, management, and customers easily.

MLTest is **designed privacy-first**. This means that while it's super easy to set up and configure, it completely runs on-premise – your data and models never leave your infrastructure.

## 1. Model testing

MLTest finds systematic performance drops and robustness issues in your models. It can run the following tests:

* **Stress tests**: Automatically profile your models against real-world variations in the input, detect (un)stable input regions of your model and highlights them to you.

* **Failure clustering**: MLTest automatically identifies and clusters failure cases of your models to highlight systematic performance drops.

* **Performance unit tests**: These tests detects and highlight slices of your dataset where the model exhibits lower performance than expected.

* **Metadata unit tests**: Ensure that models never drop below a certain performance threshold for dataset slices specified through metadata.

* **Robustness unit tests**: These tests generate additional synthetic data to test your model against real-world variations in the input.

:bulb: We are constantly extending MLTest's model capabilities. Have a particular need? Get in touch with us! :grinning:

## 2. Data testing

No model evaluation is complete without taking a look at the data as well. MLTest can run the following data tests:

* **Representativity tests**: MLTest automatically identifies and clusters underrepresentative data slices and biases to guide your data collection efforts.

* **Metadata representativity tests**: Identify underrepresentative data slices and biases based on metadata.

* **Shortcut tests**: Extract signals from your datasets that may lead to shortcut learnings.

:bulb: We are constantly extending MLTest's data capabilities. Have a particular need? Get in touch with us! :grinning:

## 3. MLTest Dashboard

MLTest has a convenient dashboard to **see test results at a glance** and **debug model behavior** efficiently. It also has **model comparison** features to understand how models differ between each other.

You can see a sample dashboard [here](#).

# :microscope: Model Selection

MLTest is often used as part of a **training-evaluation loop**. It comes with a simple programmatic Python API which makes integration easy. 

The following provides an example:

```python

from lakera.mltest import Evaluator

...

evaluator = Evaluator(PATH_TO_OUTPUT)

evaluator.load_dataset("coco")
evaluator.load_model("my_yolo_model")
evaluator.load_default_config()

# You're all setup, simply execute the evaluation. 
evaluator.run()
```

# :factory: CI/CD

MLTest **completely automates testing as part of existing CI/CD pipeliens**. You can think of MLTest as a quality gate that absorbs the complexity of testing while you focus on building the best-possible models.

To achieve that, MLTest can easily be packaged as a **Docker container** and can run as part of CI/CD pipelines (GitLab CI/CD, GitHub Actions, etc.).

For example:

```yaml

stages:
  - test

test_object_detector:
  stage: test
  image: registry.gitlab.com/lakeraai/lakera/mltest:latest
  script:
    - python run_mltest.py --base_path $CI_PROJECT_DIR --test_output_path $CI_PROJECT_DIR/test_output
  rules:
    ...
```

# :gear: Integrations

| Any infrastructure | Every platform   | All ML frameworks |
|--------------------|------------------|-------------------|
| Google Cloud       | Apache Airflow   | TensorFlow        |
| AWS                | MLFlow           | Keras             |
| Microsoft Azure    | Docker           | Keras             |
| Bitbucket          | DVC              | Scikit Learn      |
| GitLab             | Weights & Biases |                   |
| GitHub             | Kubeflow         |                   |

# :mega: Stay in touch

* To receive updates from us, connect on <a href="https://www.linkedin.com/company/lakeraai" target="_blank">LinkedIn</a>, <a href="https://twitter.com/lakeraai" target="_blank">Twitter</a>, or sign up for our <a href="https://www.lakera.ai/updates" target="_blank">updates</a>.

* We regularly share insights around computer vision development on our <a href="https://www.lakera.ai/insights" target="_blank">blog</a>.



