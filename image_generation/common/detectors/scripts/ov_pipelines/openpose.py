from collections import namedtuple
import torch


class OpenPoseOVModel:
    """Helper wrapper for OpenPose model inference"""

    def __init__(self, core, model_path, device="AUTO"):
        self.core = core
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)

    def __call__(self, input_tensor: torch.Tensor):
        """
        inference step

        Parameters:
          input_tensor (torch.Tensor): tensor with prerpcessed input image
        Returns:
           predicted keypoints heatmaps
        """
        h, w = input_tensor.shape[2:]
        input_shape = self.model.input(0).shape
        if h != input_shape[2] or w != input_shape[3]:
            self.reshape_model(h, w)
        results = self.compiled_model(input_tensor)
        return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])

    def reshape_model(self, height: int, width: int):
        """
        helper method for reshaping model to fit input data

        Parameters:
          height (int): input tensor height
          width (int): input tensor width
        Returns:
          None
        """
        self.model.reshape({0: [1, 3, height, width]})
        self.compiled_model = self.core.compile_model(self.model)

    def parameters(self):
        Device = namedtuple("Device", ["device"])
        return [Device(torch.device("cpu"))]

