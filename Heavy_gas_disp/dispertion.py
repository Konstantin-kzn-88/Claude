# -----------------------------------------------------------
# Расчет рассеивания тяжелого газа
#
# Fires, explosions, and toxic gas dispersions :
# effects calculation and risk analysis /
# Marc J. Assael, Konstantinos E. Kakosimos.
#
# (C) 2024
# -----------------------------------------------------------

import matplotlib

matplotlib.use('Qt5Agg')  # Используем Qt5 backend
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Константы
GRAVITY = 9.81  # ускорение свободного падения, м/с²
KGM3_TO_MGLITER = 1000  # коэффициент перевода кг/м³ в мг/л


class ConcentrationLevel(Enum):
    """Предопределенные уровни концентрации для расчетов"""
    VERY_HIGH = 0.1  # Очень высокая концентрация
    HIGH = 0.05  # Высокая концентрация
    MEDIUM = 0.02  # Средняя концентрация
    LOW = 0.01  # Низкая концентрация
    VERY_LOW = 0.005  # Очень низкая концентрация
    TRACE_HIGH = 0.002  # Следовая высокая концентрация
    TRACE_LOW = 0.001  # Следовая низкая концентрация


@dataclass
class DispersionResult:
    """Класс для хранения результатов расчета рассеивания"""
    concentrations: List[float]  # Концентрации, кг/м³
    distances: List[float]  # Расстояния, м
    cloud_widths: List[float]  # Ширина облака, м
    toxic_doses: List[float]  # Токсодозы, мг·мин/л
    arrival_times: List[float] = None  # Время подхода облака, с (только для мгновенного выброса)

    def plot_concentration_distance(self):
        """Построение графика зависимости концентрации от расстояния"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.distances, self.concentrations, 'b-o', linewidth=2)
        plt.grid(True)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Концентрация (кг/м³)')
        plt.title('Зависимость концентрации от расстояния')
        plt.yscale('log')
        plt.show()

    def plot_cloud_width(self):
        """Построение графика ширины облака"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.distances, self.cloud_widths, 'r-o', linewidth=2)
        plt.grid(True)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Ширина облака (м)')
        plt.title('Изменение ширины облака с расстоянием')
        plt.show()

    def plot_toxic_dose(self):
        """Построение графика токсодозы"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.distances, self.toxic_doses, 'g-o', linewidth=2)
        plt.grid(True)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Токсодоза (мг·мин/л)')
        plt.title('Изменение токсодозы с расстоянием')
        plt.yscale('log')
        plt.show()

    def plot_arrival_time(self):
        """Построение графика времени подхода облака (только для мгновенного выброса)"""
        if self.arrival_times is None:
            print("Время подхода доступно только для мгновенного выброса")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.distances, self.arrival_times, 'm-o', linewidth=2)
        plt.grid(True)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Время подхода (с)')
        plt.title('Время подхода облака')
        plt.show()

    def plot_cloud_shape(self, distance_index: int = -1):
        """
        Построение формы облака для выбранного расстояния

        Args:
            distance_index (int): Индекс расстояния (-1 для последнего значения)
        """
        distance = self.distances[distance_index]
        width = self.cloud_widths[distance_index]

        # Создаем форму облака
        fig, ax = plt.subplots(figsize=(12, 8))

        # Рисуем направление ветра
        plt.arrow(0, 0, distance / 5, 0, head_width=width / 10,
                  head_length=distance / 20, fc='gray', ec='gray')
        plt.text(distance / 10, width / 5, 'Направление ветра',
                 rotation=0, ha='center')

        # Рисуем облако
        cloud_shape = Polygon([(0, -width / 2),
                               (distance, -width / 2),
                               (distance, width / 2),
                               (0, width / 2)],
                              alpha=0.3, color='blue')
        ax.add_patch(cloud_shape)

        # Настройка графика
        plt.grid(True)
        plt.xlabel('Расстояние (м)')
        plt.ylabel('Ширина (м)')
        plt.title(f'Форма облака на расстоянии {distance:.1f} м')

        # Устанавливаем одинаковый масштаб по осям
        ax.set_aspect('equal')

        # Устанавливаем пределы осей
        plt.xlim(-distance / 10, distance * 1.1)
        plt.ylim(-width, width)

        plt.show()

    def plot_all(self):
        """Построение всех графиков"""
        plt.style.use('default')

        if self.arrival_times is not None:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # График концентрации
        ax1.plot(self.distances, self.concentrations, 'b-o', linewidth=2)
        ax1.set_xlabel('Расстояние (м)')
        ax1.set_ylabel('Концентрация (кг/м³)')
        ax1.set_title('Зависимость концентрации\nот расстояния')
        ax1.set_yscale('log')
        ax1.grid(True)

        # График ширины облака
        ax2.plot(self.distances, self.cloud_widths, 'r-o', linewidth=2)
        ax2.set_xlabel('Расстояние (м)')
        ax2.set_ylabel('Ширина облака (м)')
        ax2.set_title('Изменение ширины облака\nс расстоянием')
        ax2.grid(True)

        # График токсодозы
        ax3.plot(self.distances, self.toxic_doses, 'g-o', linewidth=2)
        ax3.set_xlabel('Расстояние (м)')
        ax3.set_ylabel('Токсодоза (мг·мин/л)')
        ax3.set_title('Изменение токсодозы\nс расстоянием')
        ax3.set_yscale('log')
        ax3.grid(True)

        # График времени подхода (только для мгновенного выброса)
        if self.arrival_times is not None:
            ax4.plot(self.distances, self.arrival_times, 'm-o', linewidth=2)
            ax4.set_xlabel('Расстояние (м)')
            ax4.set_ylabel('Время подхода (с)')
            ax4.set_title('Время подхода облака')
            ax4.grid(True)

        plt.tight_layout()
        plt.show()


class HeavyGasDispersionBase:
    """Базовый класс для расчетов рассеивания тяжелого газа"""

    def __init__(self, wind_speed: float, air_density: float, gas_density: float):
        """
        Инициализация базовых параметров расчета рассеивания

        Аргументы:
            wind_speed (float): Скорость ветра, м/с
            air_density (float): Плотность воздуха, кг/м³
            gas_density (float): Плотность газа, кг/м³
        """
        self._validate_inputs(wind_speed, air_density, gas_density)
        self.wind_speed = wind_speed
        self.air_density = air_density
        self.gas_density = gas_density

    def _validate_inputs(self, wind_speed: float, air_density: float, gas_density: float):
        """Проверка входных параметров"""
        if wind_speed <= 0:
            raise ValueError("Скорость ветра должна быть положительной")
        if air_density <= 0:
            raise ValueError("Плотность воздуха должна быть положительной")
        if gas_density <= 0:
            raise ValueError("Плотность газа должна быть положительной")
        if gas_density <= air_density:
            raise ValueError("Эта модель только для тяжелых газов (плотность газа > плотности воздуха)")

    def _calculate_g0(self) -> float:
        """Расчет начального параметра гравитационного растекания"""
        return GRAVITY * (self.gas_density - self.air_density) / self.air_density

    def calculate_toxic_dose(self, concentration: float) -> float:
        """
        Расчет токсодозы

        Аргументы:
            concentration (float): Концентрация газа, кг/м³

        Возвращает:
            float: Токсодоза, мг·мин/л
        """
        return ((2 * math.pow(2 * math.pi, 2)) / self.wind_speed) * concentration * 16.67


class InstantaneousRelease(HeavyGasDispersionBase):
    """Класс для расчета мгновенного выброса тяжелого газа"""

    def __init__(self, wind_speed: float, air_density: float, gas_density: float, gas_volume: float):
        """
        Инициализация параметров мгновенного выброса

        Аргументы:
            wind_speed (float): Скорость ветра, м/с
            air_density (float): Плотность воздуха, кг/м³
            gas_density (float): Плотность газа, кг/м³
            gas_volume (float): Объем выброшенного газа, м³
        """
        super().__init__(wind_speed, air_density, gas_density)
        if gas_volume <= 0:
            raise ValueError("Объем газа должен быть положительным")
        self.gas_volume = gas_volume

    def calculate_alpha(self) -> float:
        """Расчет параметра альфа для диаграммы Бриттера и Макквайда"""
        g0 = self._calculate_g0()
        return (1 / 2) * math.log10(g0 * math.pow(self.gas_volume, 1 / 3) /
                                    math.pow(self.wind_speed, 2))

    def calculate_beta(self, alpha: float, concentration: float) -> float:
        """
        Расчет параметра бета для заданной концентрации

        Аргументы:
            alpha (float): Параметр альфа
            concentration (float): Целевой уровень концентрации

        Возвращает:
            float: Параметр бета
        """
        if concentration == ConcentrationLevel.VERY_HIGH.value:
            if alpha <= -0.44:
                return 0.7
            elif alpha <= 0.43:
                return 0.26 * alpha + 0.81
            else:
                return 0.93
        elif concentration == ConcentrationLevel.HIGH.value:
            if alpha <= -0.56:
                return 0.85
            elif alpha <= 0.31:
                return 0.26 * alpha + 1.0
            else:
                return -0.12 * alpha + 1.12
        elif concentration == ConcentrationLevel.MEDIUM.value:
            if alpha <= -0.66:
                return 0.95
            elif alpha <= 0.32:
                return 0.36 * alpha + 1.19
            else:
                return -0.26 * alpha + 1.38
        elif concentration == ConcentrationLevel.LOW.value:
            if alpha <= -0.71:
                return 1.15
            elif alpha <= 0.37:
                return 0.34 * alpha + 1.39
            else:
                return -0.38 * alpha + 1.66
        elif concentration == ConcentrationLevel.VERY_LOW.value:
            if alpha <= -0.52:
                return 1.48
            elif alpha <= 0.24:
                return 0.26 * alpha + 1.62
            else:
                return -0.30 * alpha + 1.75
        elif concentration == ConcentrationLevel.TRACE_HIGH.value:
            if alpha <= 0.27:
                return 1.83
            else:
                return -0.32 * alpha + 1.92
        elif concentration == ConcentrationLevel.TRACE_LOW.value:
            if alpha <= -0.1:
                return 2.075
            else:
                return -0.27 * alpha + 2.05

        return self._default_beta(alpha)

    def _default_beta(self, alpha: float) -> float:
        """Расчет бета по умолчанию для неопределенных уровней концентрации"""
        return -0.27 * alpha + 2.05

    def calculate_distance(self, beta: float) -> float:
        """Расчет расстояния для заданного параметра бета"""
        return math.pow(10, beta) * math.pow(self.gas_volume, 1 / 3)

    def calculate_arrival_time(self, distance: float) -> Tuple[float, float]:
        """
        Расчет времени подхода облака и его ширины

        Аргументы:
            distance (float): Расстояние от точки выброса, м

        Возвращает:
            Tuple[float, float]: (время подхода, с; ширина облака, м)
        """
        cloud_diameter = math.pow(self.gas_volume / math.pi, 1 / 3)
        g0 = self._calculate_g0()

        def time_equation(t: float) -> float:
            return (math.sqrt(cloud_diameter ** 2 + 1.2 * t *
                              math.sqrt(g0 * self.gas_volume)) -
                    (distance - 0.4 * self.wind_speed * t))

        # Численное решение методом бинарного поиска
        t_min, t_max = 0, distance / (0.4 * self.wind_speed)
        while t_max - t_min > 1e-6:
            t_mid = (t_min + t_max) / 2
            if time_equation(t_mid) > 0:
                t_max = t_mid
            else:
                t_min = t_mid

        width = distance - 0.4 * self.wind_speed * t_min
        return t_min, width

    def calculate_dispersion(self) -> DispersionResult:
        """
        Расчет параметров рассеивания для различных уровней концентрации

        Возвращает:
            DispersionResult: Рассчитанные параметры рассеивания
        """
        concentrations = []
        distances = []
        widths = []
        times = []
        toxic_doses = []

        for conc_level in ConcentrationLevel:
            alpha = self.calculate_alpha()
            beta = self.calculate_beta(alpha, conc_level.value)
            distance = self.calculate_distance(beta)
            concentration = self.gas_density * conc_level.value
            time, width = self.calculate_arrival_time(distance)

            concentrations.append(concentration)
            distances.append(distance)
            widths.append(width)
            times.append(time)
            toxic_doses.append(self.calculate_toxic_dose(concentration))

        return DispersionResult(
            concentrations=concentrations,
            distances=distances,
            cloud_widths=widths,
            toxic_doses=toxic_doses,
            arrival_times=times
        )


class ContinuousRelease(HeavyGasDispersionBase):
    """Класс для расчета непрерывного выброса тяжелого газа"""

    def __init__(self, wind_speed: float, air_density: float, gas_density: float,
                 gas_flow: float, release_radius: float):
        """
        Инициализация параметров непрерывного выброса

        Аргументы:
            wind_speed (float): Скорость ветра, м/с
            air_density (float): Плотность воздуха, кг/м³
            gas_density (float): Плотность газа, кг/м³
            gas_flow (float): Расход газа, кг/с
            release_radius (float): Радиус отверстия выброса, м
        """
        super().__init__(wind_speed, air_density, gas_density)
        if gas_flow <= 0:
            raise ValueError("Расход газа должен быть положительным")
        if release_radius <= 0:
            raise ValueError("Радиус выброса должен быть положительным")
        self.gas_flow = gas_flow
        self.release_radius = release_radius

    def calculate_alpha(self) -> float:
        """Расчет параметра альфа для непрерывного выброса"""
        volumetric_flow = self.gas_flow / self.gas_density
        g0 = self._calculate_g0()
        return (1 / 5) * math.log10(g0 * g0 * volumetric_flow /
                                    math.pow(self.wind_speed, 5))

    def calculate_beta(self, alpha: float, concentration: float) -> float:
        """Расчет параметра бета для непрерывного выброса"""
        if concentration == ConcentrationLevel.VERY_HIGH.value:
            if alpha <= -0.55:
                return 1.75
            elif alpha <= -0.14:
                return 0.24 * alpha + 1.88
            else:
                return 0.5 * alpha + 1.78
        elif concentration == ConcentrationLevel.HIGH.value:
            if alpha <= -0.68:
                return 1.92
            elif alpha <= -0.29:
                return 0.36 * alpha + 2.16
            elif alpha <= -0.18:
                return 2.06
            else:
                return 0.56 * alpha + 1.96
        elif concentration == ConcentrationLevel.MEDIUM.value:
            if alpha <= -0.69:
                return 2.08
            elif alpha <= -0.31:
                return 0.45 * alpha + 2.39
            elif alpha <= -0.16:
                return 2.25
            else:
                return 0.54 * alpha + 2.16
        elif concentration == ConcentrationLevel.LOW.value:
            if alpha <= -0.70:
                return 2.25
            elif alpha <= -0.29:
                return 0.49 * alpha + 2.59
            elif alpha <= -0.20:
                return 2.45
            else:
                return 0.52 * alpha + 2.35
        elif concentration == ConcentrationLevel.VERY_LOW.value:
            if alpha <= -0.67:
                return 2.4
            elif alpha <= -0.28:
                return 0.59 * alpha + 2.80
            elif alpha <= -0.15:
                return 2.63
            else:
                return 0.49 * alpha + 2.56
        elif concentration == ConcentrationLevel.TRACE_HIGH.value:
            if alpha <= -0.69:
                return 2.6
            elif alpha <= -0.25:
                return 0.39 * alpha + 2.87
            elif alpha <= -0.13:
                return 2.77
            else:
                return 0.50 * alpha + 2.71

        return 0.50 * alpha + 2.71

    def calculate_distance(self, beta: float) -> float:
        """Расчет расстояния для непрерывного выброса"""
        volumetric_flow = self.gas_flow / self.gas_density
        return math.pow(10, beta) * math.pow(volumetric_flow / self.wind_speed, 0.5)

    def calculate_plume_width(self, distance: float) -> float:
        """Расчет ширины струи на заданном расстоянии"""
        volumetric_flow = self.gas_flow / self.gas_density
        g0 = self._calculate_g0()
        l_b = g0 * volumetric_flow / math.pow(self.wind_speed, 3)
        return (2 * self.release_radius + 8 * l_b +
                2.5 * math.pow(l_b, 1 / 3) * math.pow(distance, 2 / 3))

    def calculate_dispersion(self) -> DispersionResult:
        """Расчет параметров рассеивания для непрерывного выброса"""
        concentrations = []
        distances = []
        widths = []
        toxic_doses = []

        for conc_level in ConcentrationLevel:
            alpha = self.calculate_alpha()
            beta = self.calculate_beta(alpha, conc_level.value)
            distance = self.calculate_distance(beta)
            concentration = self.gas_density * conc_level.value
            width = self.calculate_plume_width(distance)

            concentrations.append(concentration)
            distances.append(distance)
            widths.append(width)
            toxic_doses.append(self.calculate_toxic_dose(concentration))

        return DispersionResult(
            concentrations=concentrations,
            distances=distances,
            cloud_widths=widths,
            toxic_doses=toxic_doses
        )


def example_instantaneous():
    """Пример расчета для мгновенного выброса"""
    print("Расчет мгновенного выброса тяжелого газа")
    print("-" * 50)

    # Создаем экземпляр класса для мгновенного выброса
    release = InstantaneousRelease(
        wind_speed=4.0,  # скорость ветра, м/с
        air_density=1.21,  # плотность воздуха, кг/м³
        gas_density=6.0,  # плотность газа, кг/м³
        gas_volume=10.0  # объем выброса, м³
    )

    # Получаем результаты расчета
    results = release.calculate_dispersion()

    # Выводим результаты
    print("\nРезультаты расчета:")
    print(
        f"{'Концентрация (кг/м³)':20} {'Расстояние (м)':15} {'Ширина (м)':15} {'Время (с)':10} {'Токсодоза (мг·мин/л)':20}")
    print("-" * 80)

    for i in range(len(results.concentrations)):
        print(f"{results.concentrations[i]:20.3f} {results.distances[i]:15.2f} {results.cloud_widths[i]:15.2f} "
              f"{results.arrival_times[i]:10.2f} {results.toxic_doses[i]:20.2f}")

    # Строим графики
    print("\nПостроение графиков...")
    results.plot_all()
    results.plot_cloud_shape()


def example_continuous():
    """Пример расчета для непрерывного выброса"""
    print("\nРасчет непрерывного выброса тяжелого газа")
    print("-" * 50)

    # Создаем экземпляр класса для непрерывного выброса
    release = ContinuousRelease(
        wind_speed=4.0,  # скорость ветра, м/с
        air_density=1.21,  # плотность воздуха, кг/м³
        gas_density=6.0,  # плотность газа, кг/м³
        gas_flow=1.0,  # расход газа, кг/с
        release_radius=0.5  # радиус отверстия, м
    )

    # Получаем результаты расчета
    results = release.calculate_dispersion()

    # Выводим результаты
    print("\nРезультаты расчета:")
    print(f"{'Концентрация (кг/м³)':20} {'Расстояние (м)':15} {'Ширина (м)':15} {'Токсодоза (мг·мин/л)':20}")
    print("-" * 70)

    for i in range(len(results.concentrations)):
        print(f"{results.concentrations[i]:20.3f} {results.distances[i]:15.2f} {results.cloud_widths[i]:15.2f} "
              f"{results.toxic_doses[i]:20.2f}")

    # Строим графики
    print("\nПостроение графиков...")
    results.plot_all()
    results.plot_cloud_shape()


if __name__ == "__main__":
    # Устанавливаем русский шрифт для графиков
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Запускаем примеры расчетов
    example_instantaneous()
    example_continuous()