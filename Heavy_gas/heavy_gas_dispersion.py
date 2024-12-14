import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simulate_dispersion():
    # Параметры сетки
    Nx, Ny = 100, 100
    Nt = 50
    D = 0.1
    vx = 2.0
    porosity = 0.3  # Пористость препятствия (0 - непроницаемое, 1 - полностью проницаемое)

    # Создаем массивы
    concentration = np.zeros((Nt, Ny, Nx))
    obstacle = np.zeros((Ny, Nx))

    # Создаем пористое препятствие в центре
    for i in range(Ny):
        for j in range(Nx):
            if (i - Ny / 2) ** 2 + (j - Nx / 2) ** 2 <= (Nx / 8) ** 2:
                # Устанавливаем степень проницаемости препятствия
                obstacle[i, j] = 1 - porosity

    # Начальное облако слева
    concentration[0, Ny // 3:2 * Ny // 3, Nx // 6:Nx // 3] = 1.0

    # Параметры расчета
    dt = 1
    dx = 1
    r = D * dt / (dx * dx)

    # Моделирование
    for t in range(Nt - 1):
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                # Теперь вещество частично проходит через препятствие
                local_concentration = concentration[t, i, j] * (1 - obstacle[i, j])

                # Диффузия (с учетом пористости)
                diff_x = (concentration[t, i, j + 1] + concentration[t, i, j - 1] - 2 * local_concentration) * (
                            1 - obstacle[i, j])
                diff_y = (concentration[t, i + 1, j] + concentration[t, i - 1, j] - 2 * local_concentration) * (
                            1 - obstacle[i, j])

                # Конвекция (с учетом пористости)
                local_vx = vx * (1 - obstacle[i, j])  # Скорость уменьшается в пористой среде
                if Nx / 2 - Nx / 6 < j < Nx / 2 + Nx / 6:
                    dy = i - Ny / 2
                    local_vx *= np.exp(-abs(dy) / (Ny / 8))

                conv = -local_vx * (concentration[t, i, j] - concentration[t, i, j - 1]) / dx

                # В пористой среде часть вещества задерживается
                retention = obstacle[i, j] * local_concentration * 0.1

                concentration[t + 1, i, j] = np.clip(
                    local_concentration + r * (diff_x + diff_y) + dt * conv - retention,
                    0, 1
                )

    return concentration, obstacle


# Моделируем
concentration, obstacle = simulate_dispersion()

# Создаем график
plt.figure(figsize=(12, 3))

# Показываем 4 момента времени
for i, t in enumerate([0, 15, 30, 45]):
    plt.subplot(1, 4, i + 1)
    plt.imshow(obstacle, cmap='YlOrBr')
    plt.imshow(concentration[t], alpha=0.7, cmap='Blues')
    plt.title(f'Время: {t * 0.1:.1f} с')
    plt.axis('off')

plt.tight_layout()
plt.savefig('porous_dispersion.png')
plt.close()

print("График сохранен в файл 'porous_dispersion.png'")