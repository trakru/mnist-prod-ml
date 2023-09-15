import torch

# Your function to test
def is_cuda_available():
    return torch.cuda.is_available()

# Your test case
def test_is_cuda_available():
    assert is_cuda_available() is True
