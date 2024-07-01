# Intro

This directory contains auxiliary libraries, scripts, and IPython notebook files used to align the Python and C++ implementations.

The Python version used for testing is 3.11.9.

## Model Conversion

First, we need to prepare the OpenPose model. Run python convert.py in the root directory. This will generate the converted OpenPose model in the ../models/ directory, which will be used by our C++ implementation.


## Compilation

Compile the detectors in the ../ directory by running the following commands:

```
mkdir build && cd build
cmake ..
make
```

This will compile the detectors library and the `detectors_bridge` executable, which we will use to help align the results.

## Data Preparation

- Download the [test data](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/media) to the `media` directory.
- Download the [body_pose_model.pth](https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/body_pose_model.pth) to the root of the scripts directory.

## Verify Results

In the directory, there is a `verify.ipynb` file. Simply run all the cells in the notebook.

In summary, it will:

- Use test cases in the `./media` directory (from the COCO test dataset), call detectors_bridge to generate results, and serialize these results as text files in the media directory.
- Iterate over the results in the media directory to parse the text data back into ndarray and visualize the results using utility functions.
- We compare the generated results with the results produced by Python using visual inspection and confirm they are identical, indicating the implementation is correct.