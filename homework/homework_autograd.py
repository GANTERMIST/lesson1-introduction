import torch
import math


# 2.1 градиенты f(x,y,z) = x^2 + y^2 + z^2 + 2xyz
# df/dx = 2x + 2yz, df/dy = 2y + 2xz, df/dz = 2z + 2xy

def task_2_1():
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    print("f =", f.item())
    print("df/dx autograd:", x.grad.item(), " аналит.:", 2*1 + 2*2*3)
    print("df/dy autograd:", y.grad.item(), " аналит.:", 2*2 + 2*1*3)
    print("df/dz autograd:", z.grad.item(), " аналит.:", 2*3 + 2*1*2)


# 2.2 градиент MSE для y = w*x + b

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def task_2_2():
    torch.manual_seed(42)
    x_data = torch.rand(10)
    y_true = 3.0 * x_data + 1.5

    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    loss = mse_loss(w * x_data + b, y_true)
    loss.backward()

    print(f"MSE: {loss.item():.4f}")
    print(f"dMSE/dw: {w.grad.item():.4f}")
    print(f"dMSE/db: {b.grad.item():.4f}")


# 2.3 цепное правило: f(x) = sin(x^2 + 1), df/dx = cos(x^2+1)*2x

def task_2_3():
    x_val = 1.0

    x = torch.tensor(x_val, requires_grad=True)
    torch.sin(x**2 + 1).backward()
    grad1 = x.grad.item()

    x2 = torch.tensor(x_val, requires_grad=True)
    grad2 = torch.autograd.grad(torch.sin(x2**2 + 1), x2)[0].item()

    grad_an = math.cos(x_val**2 + 1) * 2 * x_val

    print(f"backward:      {grad1:.6f}")
    print(f"autograd.grad: {grad2:.6f}")
    print(f"аналитически:  {grad_an:.6f}")


if __name__ == "__main__":
    print("--- 2.1 градиенты f(x,y,z) ---")
    task_2_1()
    print("\n--- 2.2 градиент MSE ---")
    task_2_2()
    print("\n--- 2.3 цепное правило ---")
    task_2_3()
