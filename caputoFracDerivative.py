import torch

def caputo_fractional_derivative(output: torch.Tensor, input_var: torch.Tensor, alpha: float, order: int = 1) -> torch.Tensor:
    """Compute Caputo fractional derivative of a neural network output with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input_var,
            grad_outputs=torch.ones_like(input_var),
            create_graph=True,
            retain_graph=True,
        )[0]
    
    # Apply fractional order difference
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input_var,
            grad_outputs=torch.ones_like(input_var),
            create_graph=True,
            retain_graph=True,
        )[0] / ((input_var + 1e-8) ** alpha)  # Add small constant to avoid division by zero
    
    return df_value
