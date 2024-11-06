import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
from typing import Dict, Tuple
import torchvision.models as models

class CUDNNBenchmark:
    def __init__(self, 
                 model: nn.Module,
                 batch_size: int = 64,
                 warmup_steps: int = 10,
                 benchmark_steps: int = 50):
        
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps
        
        self.steps = 10
        self.dry_runs = 10
        
        self.model = model.cuda()
        self.model.train()
        self.configure_cudnn()
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f'Running on device: {device.name}')
        
    def _makeInput(self, input_size: Tuple) -> torch.Tensor:
        """Similar to makeInput function in original"""
        input_tensor = torch.randn(input_size).cuda()
        input_tensor.requires_grad = True
        return input_tensor
        
    def _get_input_size(self) -> Tuple[int, int, int, int]:
        return (self.batch_size, 3, 224, 224)
    
    def configure_cudnn(self, 
                       enabled: bool = True,
                       benchmark: bool = True,
                       deterministic: bool = False):
        """Configure cuDNN settings"""
        # cudnn.verbose = False -- no flag in original benchmark file
        cudnn.enabled = enabled
        cudnn.benchmark = benchmark
        cudnn.deterministic = deterministic
        
        
    def run_benchmark(self) -> Dict[str, float]:
        """Run the benchmark and return timing results"""
        input_tensor = self._makeInput(self._get_input_size())
        
        self._dry_run(input_tensor)
        
        # Benchmark
        forward_time = self._time_forward(input_tensor)
        backward_time = self._time_backward(input_tensor)
        
        return {
            'forward_ms': forward_time * 1000,
            'backward_ms': backward_time * 1000,
            'total_ms': (forward_time + backward_time) * 1000
        }

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
                retain_graph=True  # Need this to reuse the graph for parameter gradients
            )
            
            # Compute parameter gradients (equivalent to accGradParameters)
            torch.autograd.grad(
                outputs=output,
                inputs=self.model.parameters(),
                grad_outputs=grad_output
            )
            
            # Synchronize GPU (equivalent to cutorch.synchronize())
            torch.cuda.synchronize()
            
            # Clean memory (equivalent to collectgarbage())
            torch.cuda.empty_cache()


if __name__ == '__main__':
    models = {
        'alexnet': models.alexnet(pretrained=True),
        'vgg_a': models.vgg11_bn(pretrained=True),
        'googlenet': models.googlenet(pretrained=True)
    }
    
    benchmark = CUDNNBenchmark(model=models['alexnet'])
    results = benchmark.run_benchmark()
    print(results)