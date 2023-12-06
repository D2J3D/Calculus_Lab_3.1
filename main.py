import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def z_fun(x, y):
    return x ** 2 + y ** 2 - x ** 3 - y ** 3 + 2 * x * y


def draw_function(function):
    samples = np.arange(0, 10, 0.1)
    x, y = np.meshgrid(samples, samples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, function(x, y))
    ax.set_zlim(0, 9)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# подсчет градиента в точке
def grad(point):
    x0, y0 = point[0], point[1]
    dx = 2 * x0 - 3 * x0 ** 2 + 2 * y0  # частная производная по x функции z(x,y) в точке (x0, y0)
    dy = 2 * y0 - 3 * y0 ** 2 + 2 * x0  # частная производная по y функции z(x,y) в точке (x0, y0)
    return np.array([dx, dy])


# подсчет нормы вектора
def norm(vector):
    return np.sqrt(sum([i ** 2 for i in list(vector)]))


# нормализация вектора
def normalize_vector(vector):
    return [vector[i] / norm(vector) for i in range(len(vector))]


def find_extremum(gradient_function, norm_function, start_point, epsilon, step):
    # Начальные условия
    x0, y0 = start_point[0], start_point[1]  # точка, в окрестности точки максимума функции, (4/3, 4/3) - max(z(x, y))
    k = 0  # счетчик количества итераций

    xk = [x0]  # массив с узловыми значениями x
    yk = [y0]  # массив с узловыми значениями y
    grad_k = [gradient_function([x0, y0])]  # значения градиента в узлах
    start_time = time.time()
    while True:
        grad_k.append(grad([xk[k], yk[k]]))
        xk.append(xk[k] + step * grad_k[k][0])
        yk.append(yk[k] + step * grad_k[k][1])
        # условие остановки - длина вектора (dx, dy) < eps
        if (norm_function([xk[k] - xk[k - 1], yk[k] - yk[k - 1]])) < epsilon:
            return [xk, yk], k, norm_function([xk[k] - xk[k - 1], yk[k] - yk[k - 1]]), time.time() - start_time
        k += 1


if __name__ == '__main__':
    epsilon = 0.0000001
    step = 0.00001
    start_point = [0.2, 2.5]

    answer = find_extremum(grad, norm, start_point, epsilon, step)
    print(f"Точка экстремума (x,y) = {round(answer[0][0][-1], 3), round(answer[0][1][-1], 3)}")
    print(f"Количество итераций: {answer[1]}")
    print(f"Время работы программы: {answer[-1]}с")
    print(norm(normalize_vector([1/np.sqrt(2), np.sqrt(3)/2, -1])))

    # построим график путешествий в поисках экстремума
    extremum_approximation = answer[0]
    extremum_approximation_by_x = extremum_approximation[0]  # подходим к точке экстремума по таким x
    extremum_approximation_by_y = extremum_approximation[1]  # и по таким y

