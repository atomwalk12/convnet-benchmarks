import time
from typing import Dict, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models


class CUDNNBenchmark:
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 640,
        warmup_steps: int = 1000,
        benchmark_steps: int = 50,
    ):
        # Check CUDA availability first
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This benchmark requires CUDA.")

        # Initialize CUDA context
        torch.cuda.init()

        # Move model to GPU and set to train mode
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps

        self.steps = 10
        self.dry_runs = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.model.train()

        self._configure_cudnn()

        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"Running on device: {device.name}")

    def run_benchmark(self) -> Dict[str, float]:
        """Run the benchmark and return timing results"""
        input_tensor = self._makeInput(self._get_input_size())

        # Perform dry runs to warm up
        self._dry_run(input_tensor)

        # Time forward pass and get output for backward passes
        forward_time, output = self._time_forward(input_tensor)
        grad_output = torch.ones_like(output)

        # Time backward passes separately
        backward_input_time = self._time_backward_input(
            input_tensor, output, grad_output
        )
        backward_params_time = self._time_backward_params(
            input_tensor, output, grad_output
        )

        return {
            "forward_ms": forward_time * 1000,
            "backward_input_ms": backward_input_time * 1000,
            "backward_params_ms": backward_params_time * 1000,
            "total_ms": (forward_time + backward_input_time + backward_params_time)
            * 1000,
        }

    def _makeInput(self, input_size: Tuple) -> torch.Tensor:
        """Similar to makeInput function in original"""
        input_tensor = torch.randn(input_size).cuda()
        input_tensor.requires_grad = True
        return input_tensor

    def _get_input_size(self) -> Tuple[int, int, int, int]:
        return (self.batch_size, 3, 224, 224)

    def _configure_cudnn(
        self, enabled: bool = True, benchmark: bool = True, deterministic: bool = False
    ):
        """Configure cuDNN settings"""
        # cudnn.verbose = False -- no flag in original benchmark file
        cudnn.enabled = enabled
        cudnn.benchmark = benchmark
        cudnn.deterministic = deterministic
        torch.cuda.empty_cache()

    def _dry_run(self, input_tensor: torch.Tensor):
        """Perform dry runs to warm up the GPU and allow cuDNN to optimize"""
        for _ in range(self.dry_runs):
            # Zero gradients (equivalent to model:zeroGradParameters())
            # Not using an optimizer in order to match original benchmark style
            self.model.zero_grad(set_to_none=True)

            # Forward pass (equivalent to model:updateOutput())
            output = self.model(input_tensor)

            # Backward pass (combines updateGradInput and accGradParameters)
            # loss = output.sum()
            # loss.backward()

            # Create a dummy gradient for the output
            grad_output = torch.ones_like(output)

            # Compute input gradients (equivalent to updateGradInput)
            torch.autograd.grad(
                outputs=output,
                inputs=input_tensor,
                grad_outputs=grad_output,
                retain_graph=True,
            )

            # Compute parameter gradients (equivalent to accGradParameters)
            torch.autograd.grad(
                outputs=output, inputs=self.model.parameters(), grad_outputs=grad_output
            )

            # Synchronize GPU (equivalent to cutorch.synchronize())
            torch.cuda.synchronize()

            # Clean memory (equivalent to collectgarbage())
            torch.cuda.empty_cache()

    def _time_forward(self, input_tensor: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Time forward pass and return timing and output"""
        torch.cuda.synchronize()

        # Required for backward pass
        final_output = None
        start_time = time.perf_counter()

        for _ in range(self.steps):
            # NOTE Is necessary?
            # self.model.zero_grad(set_to_none=True)

            final_output = self.model(input_tensor)
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        forward_time = (end_time - start_time) / self.steps

        return forward_time, final_output

    def _time_backward_input(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> float:
        """Time backward pass for input gradients"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(self.steps):
            # Equivalent to updateGradInput
            torch.autograd.grad(
                outputs=output,
                inputs=input_tensor,
                grad_outputs=grad_output,
                retain_graph=True,
            )
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        backward_input_time = (end_time - start_time) / self.steps

        return backward_input_time

    def _time_backward_params(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> float:
        """Time backward pass for parameter gradients"""
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for i in range(self.steps):
            # Retain graph for all but the last iteration
            retain_graph = i < self.steps - 1

            torch.autograd.grad(
                outputs=output,
                inputs=list(self.model.parameters()),
                grad_outputs=grad_output,
                retain_graph=retain_graph,
            )
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        backward_params_time = (end_time - start_time) / self.steps

        return backward_params_time


if __name__ == "__main__":
    models = {
        "alexnet": models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        "vgg_a": models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT),
        "googlenet": models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
    }

    benchmark = CUDNNBenchmark(model=models["alexnet"])
    results = benchmark.run_benchmark()
    print(results)
