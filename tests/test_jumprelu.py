import torch
from models.jump_relu import JumpReLU


def test_jump_relu():
    jump_relu = JumpReLU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    act = jump_relu(x)
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(act, expected)


def test_jump_relu_with_jump():
    jump_relu = JumpReLU(jump=1.0)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    act = jump_relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
    assert torch.allclose(act, expected)
