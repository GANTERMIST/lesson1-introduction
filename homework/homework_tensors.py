import torch


# 1.1 создание тензоров

def task_1_1():
    t1 = torch.rand(3, 4)
    print("3x4 random [0,1):\n", t1)

    t2 = torch.zeros(2, 3, 4)
    print("\n2x3x4 zeros:\n", t2)

    t3 = torch.ones(5, 5)
    print("\n5x5 ones:\n", t3)

    t4 = torch.arange(0, 16).reshape(4, 4)
    print("\n4x4 arange 0..15:\n", t4)

    return t1, t2, t3, t4


# 1.2 операции с тензорами

def task_1_2():
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    A_T = A.T
    print("A транспонированное (4x3):\n", A_T)

    AB = torch.matmul(A, B)
    print("\nA @ B (3x3):\n", AB)

    # поэлементное умножение
    elem = A * B.T
    print("\nA * B.T:\n", elem)

    total = A.sum()
    print("\nсумма элементов A:", total.item())

    return A_T, AB, elem, total


# 1.3 индексация и срезы

def task_1_3():
    t = torch.arange(125).reshape(5, 5, 5).float()

    first_row = t[0, 0, :]
    print("первая строка:", first_row)

    last_col = t[:, :, -1]
    print("\nпоследний столбец (5x5):\n", last_col)

    center = t[0, 1:3, 1:3]
    print("\nподматрица 2x2 из центра:\n", center)

    even = t[::2, ::2, ::2]
    print("\nчётные индексы (3x3x3):\n", even)

    return first_row, last_col, center, even


# 1.4 reshape

def task_1_4():
    base = torch.arange(24).float()
    for shape in [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]:
        print(f"форма {shape}:\n", base.reshape(shape), "\n")


if __name__ == "__main__":
    print("--- 1.1 создание тензоров ---")
    task_1_1()
    print("\n--- 1.2 операции ---")
    task_1_2()
    print("\n--- 1.3 индексация ---")
    task_1_3()
    print("\n--- 1.4 формы ---")
    task_1_4()
