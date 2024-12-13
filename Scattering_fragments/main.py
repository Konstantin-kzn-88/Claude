import math
import numpy as np
from scipy.special import gamma
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда до импорта pyplot
import matplotlib.pyplot as plt


def calculate_initial_velocity(P0, M_ob, rho_ob, V0, is_spherical=False):
    """
    Расчет начальной скорости осколков по официальной методике
    P0 - избыточное давление (атм)
    M_ob - масса оболочки (кг)
    rho_ob - плотность материала оболочки (кг/м3)
    V0 - объем сосуда (м3)
    is_spherical - сферический (True) или цилиндрический (False) сосуд
    """
    # Переводим атмосферы в Паскали
    P0_pa = P0 * 101325

    # Для цилиндрического резервуара
    if not is_spherical:
        return 0.37 * math.sqrt(P0_pa * V0 / M_ob)  # Убрана плотность из формулы
    # Для сферического резервуара
    else:
        return 0.35 * math.sqrt(P0_pa * V0 / M_ob)  # Убрана плотность из формулы


def calculate_effective_energy(P0, V0):
    """
    Расчет эффективной энергии взрыва резервуара
    P0 - избыточное давление (атм)
    V0 - объем сосуда (м3)
    """
    # Переводим атмосферы в Паскали
    P0_pa = P0 * 101325
    return 0.6 * P0_pa * V0


def calculate_fragment_parameters(m_osk, U0):
    """
    Расчет параметров движения осколка
    m_osk - масса осколка (кг)
    U0 - начальная скорость (м/с)
    """
    rho_air = 1.29  # плотность воздуха, кг/м3
    Cx = 1.0  # коэффициент сопротивления (уменьшен)
    S_mid = (m_osk / 7850) ** (2 / 3) * 0.5  # площадь миделя с корректировкой

    # Расчет приведенного коэффициента сопротивления
    A = (Cx * S_mid * rho_air) / (2 * m_osk)

    # Расчет параметра W с корректирующим коэффициентом
    W = (A * U0 ** 2) / (2 * 9.81) * 0.1  # Добавлен корректирующий коэффициент

    return A, W


def calculate_max_distance(W, U0):
    """
    Расчет максимальной дальности разлета осколков
    """
    if W < 4.6:
        R_max = (U0 ** 2 / (2 * 9.81)) * math.exp(-0.45 * W)
    else:
        R_max = (U0 ** 2 / (2 * 9.81)) * 0.13

    return R_max


def calculate_hit_probability(R, R_max, dS):
    """
    Расчет вероятности попадания осколка в заданную область
    R - расстояние от центра разгерметизации (м)
    R_max - максимальная дальность разлета (м)
    dS - площадь выделенной области (м2)
    """
    # Проверка на нулевое расстояние
    if R < 0.1:  # Минимальное расстояние 0.1 м
        R = 0.1

    if R_max < 0.1:  # Защита от деления на ноль
        R_max = 0.1

    # Параметры бета-распределения (изменены для более реалистичного распределения)
    alpha = 1.5
    beta = 2.0

    # Нормированное расстояние
    r_norm = R / R_max

    # Плотность вероятности (бета-распределение)
    if 0 <= r_norm <= 1:
        f_R = (r_norm ** (alpha - 1) * (1 - r_norm) ** (beta - 1) *
               gamma(alpha + beta) / (gamma(alpha) * gamma(beta)))
    else:
        f_R = 0

    # Увеличенная площадь поражения (учитываем размер человека)
    effective_dS = dS * 2.0  # м2 (приблизительная площадь проекции человека)

    # Вероятность попадания с учетом количества осколков
    P = (effective_dS * f_R) / (2 * math.pi * R)  # Убрано R_max из знаменателя

    # Учитываем вероятность поражения несколькими осколками
    n_fragments = 5  # количество осколков
    P_total = 1 - (1 - min(1, P)) ** n_fragments  # вероятность поражения хотя бы одним осколком

    return max(0, min(1, P_total))  # Ограничиваем вероятность диапазоном [0, 1]


def analyze_fragments(
        P0,  # атм - избыточное давление
        V0,  # м3 - объем резервуара
        M_ob,  # кг - масса оболочки
        rho_ob,  # кг/м3 - плотность материала оболочки
        n_fragments,  # количество осколков
        is_spherical=False  # тип резервуара
):
    """
    Комплексный анализ разлета осколков
    """
    # Расчет начальной скорости
    U0 = calculate_initial_velocity(P0, M_ob, rho_ob, V0, is_spherical)

    # Расчет эффективной энергии взрыва
    E_eff = calculate_effective_energy(P0, V0)

    # Средняя масса одного осколка
    m_osk = M_ob / n_fragments

    # Расчет параметров движения
    A, W = calculate_fragment_parameters(m_osk, U0)

    # Расчет максимальной дальности
    R_max = calculate_max_distance(W, U0)

    return {
        'initial_velocity': U0,
        'effective_energy': E_eff,
        'fragment_mass': m_osk,
        'parameter_W': W,
        'max_distance': R_max
    }


def calculate_trajectories(U0, W, angles=None):
    """
    Расчет траекторий для различных углов вылета
    """
    if angles is None:
        angles = [15, 30, 45, 60, 75]  # углы в градусах

    g = 9.81  # ускорение свободного падения
    dt = 0.1  # шаг по времени

    trajectories = []

    for angle in angles:
        # Начальные условия
        theta = math.radians(angle)
        Vx = U0 * math.cos(theta)
        Vy = U0 * math.sin(theta)
        x = 0
        y = 0
        points = []

        while y >= 0:
            points.append((x, y))

            # Обновление скоростей с учетом сопротивления воздуха
            V = math.sqrt(Vx ** 2 + Vy ** 2)
            ax = -(W * g * Vx * V) / U0 ** 2
            ay = -g - (W * g * Vy * V) / U0 ** 2

            Vx += ax * dt
            Vy += ay * dt
            x += Vx * dt
            y += Vy * dt

        trajectories.append({
            'angle': angle,
            'points': np.array(points)
        })

    return trajectories


def plot_results(results, save_path='fragment_analysis.png'):
    """
    Построение графиков анализа разлета осколков
    """
    # Создаем фигуру с подграфиками
    plt.figure(figsize=(15, 10))

    # График траекторий
    plt.subplot(121)
    trajectories = calculate_trajectories(results['initial_velocity'], results['parameter_W'])

    for traj in trajectories:
        points = traj['points']
        plt.plot(points[:, 0], points[:, 1],
                 label=f"{traj['angle']}°")

    plt.title('Траектории осколков')
    plt.xlabel('Дальность (м)')
    plt.ylabel('Высота (м)')
    plt.grid(True)
    plt.legend(title='Угол вылета')

    # График вероятности поражения
    plt.subplot(122)
    distances = np.linspace(0, results['max_distance'], 100)
    probabilities = [calculate_hit_probability(r, results['max_distance'], 1.0)
                     for r in distances]

    plt.plot(distances, probabilities)
    plt.title('Вероятность поражения')
    plt.xlabel('Расстояние (м)')
    plt.ylabel('Вероятность')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Пример использования
def example_calculation():
    params = {
        'P0': 2,  # 8 атм
        'V0': 100,  # 100 м3
        'M_ob': 4000,  # 4000 кг масса оболочки
        'rho_ob': 7850,  # плотность стали
        'n_fragments': 3,  # количество осколков
        'is_spherical': False  # цилиндрический резервуар
    }

    results = analyze_fragments(**params)

    print(f"Результаты расчета:")
    print(f"Начальная скорость осколков: {results['initial_velocity']:.1f} м/с")
    print(f"Эффективная энергия взрыва: {results['effective_energy'] / 1e6:.1f} МДж")
    print(f"Масса осколка: {results['fragment_mass']:.1f} кг")
    print(f"Параметр W: {results['parameter_W']:.2f}")
    print(f"Максимальная дальность разлета: {results['max_distance']:.1f} м")

    # Построение и сохранение графиков
    plot_results(results)


if __name__ == "__main__":
    example_calculation()