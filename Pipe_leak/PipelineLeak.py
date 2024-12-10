import math
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда перед импортом pyplot
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def mpa_to_pa(pressure_mpa: float) -> float:
    """Конвертация давления из МПа в Па"""
    return pressure_mpa * 1e6


def pa_to_mpa(pressure_pa: float) -> float:
    """Конвертация давления из Па в МПа"""
    return pressure_pa / 1e6


def calculate_leak_flow(
        pressure_start_mpa: float,
        pressure_end_mpa: float,
        height_diff: float,
        pipe_diameter: float,
        hole_diameter: float,
        pipe_length: float,
        fluid_density: float,
        roughness: float = 0.0001,
        discharge_coef: float = 0.62,
        kinematic_viscosity: float = 1e-6
) -> dict:
    """
    Расчет параметров аварийного истечения жидкости из трубопровода

    Args:
        pressure_start_mpa: Начальное давление в точке истечения, МПа
        pressure_end_mpa: Конечное давление (обычно атмосферное), МПа
        height_diff: Разница высот между точкой истечения и уровнем сравнения, м
        pipe_diameter: Диаметр трубопровода, м
        hole_diameter: Диаметр отверстия истечения, м
        pipe_length: Длина трубопровода, м
        fluid_density: Плотность жидкости, кг/м³
        roughness: Шероховатость трубы, м
        discharge_coef: Коэффициент расхода
        kinematic_viscosity: Кинематическая вязкость жидкости, м²/с

    Returns:
        dict: Словарь с результатами расчета
    """
    g = 9.81  # Ускорение свободного падения, м/с²

    # Конвертация давления в Па
    pressure_start = mpa_to_pa(pressure_start_mpa)
    pressure_end = mpa_to_pa(pressure_end_mpa)

    # Площади сечений
    hole_area = math.pi * (hole_diameter / 2) ** 2
    pipe_area = math.pi * (pipe_diameter / 2) ** 2
    pipe_volume = pipe_area * pipe_length

    # Расчет напора
    pressure_head = (pressure_start - pressure_end) / (fluid_density * g)
    total_head = pressure_head + height_diff

    # Скорость истечения по формуле Торричелли
    velocity = discharge_coef * math.sqrt(2 * g * abs(total_head))

    # Расход жидкости
    volume_flow = velocity * hole_area
    mass_flow = volume_flow * fluid_density

    # Расчет потерь давления по длине
    dynamic_viscosity = kinematic_viscosity * fluid_density
    reynolds = fluid_density * volume_flow * pipe_diameter / (pipe_area * dynamic_viscosity)

    # Коэффициент трения (формула Колбрука-Уайта)
    def colebrook_white(f):
        return -2 * math.log10(roughness / (3.7 * pipe_diameter) + 2.51 / (reynolds * math.sqrt(f)))

    # Итерационное решение для коэффициента трения
    f = 0.02  # Начальное приближение
    for _ in range(10):
        f = 1 / (colebrook_white(f)) ** 2

    # Потери давления по формуле Дарси-Вейсбаха
    velocity_pipe = volume_flow / pipe_area
    pressure_drop = f * pipe_length * fluid_density * velocity_pipe ** 2 / (2 * pipe_diameter)
    pressure_drop_mpa = pa_to_mpa(pressure_drop)

    # Время опорожнения
    if hole_diameter < pipe_diameter:
        emptying_time = pipe_volume / (hole_area * velocity)
    else:
        emptying_time = math.sqrt(2 * pipe_length / g)  # Для случая полного разрыва

    return {
        "velocity": velocity,  # Скорость истечения, м/с
        "volume_flow": volume_flow,  # Объемный расход, м³/с
        "volume_flow_hour": volume_flow * 3600,  # Объемный расход, м³/ч
        "mass_flow": mass_flow,  # Массовый расход, кг/с
        "emptying_time": emptying_time,  # Время опорожнения всего участка, с
        "emptying_time_min": emptying_time / 60,  # Время опорожнения, мин
        "reynolds": reynolds,  # Число Рейнольдса
        "total_head": total_head,  # Полный напор, м
        "pressure_drop_mpa": pressure_drop_mpa,  # Потери давления, МПа
        "effective_pressure_mpa": pressure_start_mpa - pressure_drop_mpa  # Эффективное давление с учетом потерь, МПа
    }


def calculate_time_series(
        pressure_start_mpa: float,
        pressure_end_mpa: float,
        height_diff: float,
        pipe_diameter: float,
        hole_diameter: float,
        pipe_length: float,
        fluid_density: float,
        roughness: float = 0.0001,
        discharge_coef: float = 0.62,
        kinematic_viscosity: float = 1e-6,
        time_steps: int = 100
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Расчет изменения параметров истечения во времени

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]: Массив времени и словарь с массивами параметров
    """
    # Начальные условия
    g = 9.81
    pipe_area = math.pi * (pipe_diameter / 2) ** 2
    hole_area = math.pi * (hole_diameter / 2) ** 2
    pipe_volume = pipe_area * pipe_length
    initial_mass = pipe_volume * fluid_density

    # Создаем временную шкалу
    # Расчет начального напора с учетом давления и высоты
    initial_pressure_head = mpa_to_pa(pressure_start_mpa) / (fluid_density * g)
    total_head = abs(initial_pressure_head + height_diff)
    # Расчет времени опорожнения через начальную скорость истечения
    initial_velocity = discharge_coef * math.sqrt(2 * g * total_head)
    emptying_time = pipe_volume / (hole_area * initial_velocity)
    time_array = np.linspace(0, emptying_time, time_steps)

    # Инициализация массивов для хранения результатов
    mass_flow_array = np.zeros(time_steps)
    pressure_array = np.zeros(time_steps)
    volume_array = np.zeros(time_steps)
    velocity_array = np.zeros(time_steps)

    # Расчет параметров для каждого момента времени
    for i, t in enumerate(time_array):
        # Оставшийся объем жидкости
        remaining_volume = max(0, pipe_volume * (1 - t / emptying_time))
        volume_array[i] = remaining_volume

        # Текущее давление с учетом столба жидкости
        current_pressure_pa = mpa_to_pa(pressure_start_mpa) * (remaining_volume / pipe_volume)
        pressure_array[i] = pa_to_mpa(current_pressure_pa)

        # Расчет полного напора
        current_head = current_pressure_pa / (fluid_density * g) + height_diff
        # Скорость истечения
        velocity = discharge_coef * math.sqrt(2 * g * abs(current_head))
        velocity_array[i] = velocity

        # Массовый расход
        mass_flow_array[i] = velocity * hole_area * fluid_density

    return time_array, {
        "mass_flow": mass_flow_array,
        "pressure": pressure_array,
        "volume": volume_array,
        "velocity": velocity_array
    }


def plot_time_series(
        time_array: np.ndarray,
        results: Dict[str, np.ndarray],
        params: Dict[str, float],
        save_path: str = 'leak_analysis.png'
) -> None:
    """
    Построение графиков изменения параметров во времени

    Args:
        time_array: Массив времени
        results: Словарь с результатами расчетов
        params: Словарь с исходными параметрами
        save_path: Путь для сохранения графиков
    """
    # Настройка стиля графиков
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # График массового расхода
    ax1.plot(time_array / 60, results["mass_flow"], 'b-', linewidth=2, label='Массовый расход')
    ax1.set_xlabel('Время, мин')
    ax1.set_ylabel('Массовый расход, кг/с')
    ax1.set_title('Изменение массового расхода во времени')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # График давления
    ax2.plot(time_array / 60, results["pressure"], 'r-', linewidth=2, label='Давление')
    ax2.set_xlabel('Время, мин')
    ax2.set_ylabel('Давление, МПа')
    ax2.set_title('Изменение давления во времени')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Добавление исходных данных на график
    info_text = (
        f'Исходные данные:\n'
        f'Начальное давление: {params["pressure_start_mpa"]:.1f} МПа\n'
        f'Длина трубопровода: {params["pipe_length"]:.0f} м\n'
        f'Диаметр трубы: {params["pipe_diameter"] * 1000:.0f} мм\n'
        f'Диаметр отверстия: {params["hole_diameter"] * 1000:.0f} мм\n'
        f'Разница высот: {params["height_diff"]:.1f} м\n'
        f'Плотность жидкости: {params["fluid_density"]:.0f} кг/м³'
    )

    fig.text(0.05, 0.05, info_text,
             fontsize=10, family='monospace',
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom',
             horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Пример использования
if __name__ == "__main__":
    # Параметры расчета
    params = {
        "pressure_start_mpa": 5.0,  # 5 МПа
        "pressure_end_mpa": 0.101325,  # Атмосферное давление
        "height_diff": -2,  # Утечка ниже уровня сравнения на 2 метра
        "pipe_diameter": 0.1,  # Труба диаметром 100 мм
        "hole_diameter": 0.01,  # Отверстие диаметром 10 мм
        "pipe_length": 1000,  # Длина трубопровода 1000 м
        "fluid_density": 998,  # Плотность воды при 20°C
        "roughness": 0.0001  # Шероховатость трубы 0.1 мм
    }

    # Расчет изменения параметров во времени
    time_array, results = calculate_time_series(**params)

    # Построение графиков
    plot_time_series(time_array, results, params, 'leak_analysis.png')
    print("\nГрафик сохранен в файл 'leak_analysis.png'")

    # Расчет и вывод начальных значений
    initial_state = calculate_leak_flow(**params)

    print("\nНачальные значения:")
    print(f"Скорость истечения: {initial_state['velocity']:.2f} м/с")
    print(f"Объемный расход: {initial_state['volume_flow_hour']:.2f} м³/ч")
    print(f"Массовый расход: {initial_state['mass_flow']:.2f} кг/с")
    print(f"Время опорожнения: {initial_state['emptying_time_min']:.1f} мин")
    print(f"Число Рейнольдса: {initial_state['reynolds']:.0f}")
    print(f"Потери давления: {initial_state['pressure_drop_mpa']:.3f} МПа")
    print(f"Эффективное давление: {initial_state['effective_pressure_mpa']:.3f} МПа")