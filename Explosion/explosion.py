import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from enum import Enum


class SubstanceClass(Enum):
    """Классы взрывоопасных веществ по чувствительности"""
    CLASS_1 = 1  # Особо чувствительные вещества
    CLASS_2 = 2  # Чувствительные вещества
    CLASS_3 = 3  # Средне чувствительные вещества
    CLASS_4 = 4  # Слабо чувствительные вещества


@dataclass
class TVSParams:
    """Параметры ТВС для расчета"""
    M: float  # Масса горючего вещества, кг
    sigma: float  # Степень расширения продуктов сгорания
    class_space: int  # Класс пространства (1-4)
    substance_class: SubstanceClass  # Класс вещества
    Cst: float  # Стехиометрическая концентрация, кг/м³
    Qst: float  # Теплота сгорания стехиометрической смеси, Дж/кг


class DeflagrationCalculator:
    """Калькулятор параметров дефлаграционного взрыва"""

    # Коэффициенты для различных классов пространства
    SPACE_COEFFS = {
        1: (0.34, 0.33),  # Сильно загроможденное пространство
        2: (0.26, 0.33),  # Средне загроможденное пространство
        3: (0.14, 0.33),  # Слабо загроможденное пространство
        4: (0.07, 0.33)  # Свободное пространство
    }

    # Корреляционные коэффициенты для различных классов веществ
    SUBSTANCE_COEFFS = {
        SubstanceClass.CLASS_1: (2.0, 0.34),  # Особо чувствительные
        SubstanceClass.CLASS_2: (1.6, 0.33),  # Чувствительные
        SubstanceClass.CLASS_3: (1.2, 0.32),  # Средне чувствительные
        SubstanceClass.CLASS_4: (1.0, 0.31)  # Слабо чувствительные
    }

    def __init__(self, params: TVSParams):
        self.params = params
        self._validate_params()
        self.E = self._calculate_energy()  # Расчет энергозапаса

    def _validate_params(self):
        """Проверка входных параметров"""
        if self.params.M <= 0:
            raise ValueError("Масса горючего должна быть положительной")
        if self.params.sigma <= 1:
            raise ValueError("Степень расширения должна быть больше 1")
        if self.params.class_space not in self.SPACE_COEFFS:
            raise ValueError("Недопустимый класс пространства")

    def _calculate_energy(self) -> float:
        """
        Расчет эффективного энергозапаса смеси

        Returns:
            float: Эффективный энергозапас, Дж
        """
        # Коэффициент участия горючего во взрыве
        Z = 0.5  # типичное значение для углеводородов

        # Расчет объема смеси
        V = self.params.M / self.params.Cst

        # Эффективный энергозапас
        E = Z * self.params.M * self.params.Qst

        return E

    def calculate_flame_speed(self, distance: float) -> float:
        """
        Расчет видимой скорости фронта пламени
        """
        M_k, n = self.SUBSTANCE_COEFFS[self.params.substance_class]
        R_pr = distance / (self.E / 101325) ** (1 / 3)
        alpha, beta = self.SPACE_COEFFS[self.params.class_space]

        V_f = M_k * (self.params.sigma - 1) * (R_pr ** (-n)) * 20  # м/с
        return V_f

    def calculate_pressure(self, distance: float) -> float:
        """
        Расчет избыточного давления на фронте ударной волны
        """
        R_pr = distance / (self.E / 101325) ** (1 / 3)
        alpha, beta = self.SPACE_COEFFS[self.params.class_space]
        M_k, n = self.SUBSTANCE_COEFFS[self.params.substance_class]

        # Модифицированная формула с учетом затухания
        delta_P = alpha * M_k * (R_pr ** (-beta)) * 101325 * np.exp(-0.1 * R_pr)

        return delta_P

    def calculate_impulse(self, distance: float) -> float:
        """
        Расчет удельного импульса положительной фазы
        """
        R_pr = distance / (self.E / 101325) ** (1 / 3)
        M_k, n = self.SUBSTANCE_COEFFS[self.params.substance_class]

        I_plus = 123 * M_k * (self.E ** (1 / 3)) / (R_pr * np.exp(0.05 * R_pr))
        return I_plus

    def calculate_tnt_equivalent(self) -> float:
        """Расчет тротилового эквивалента"""
        Q_tnt = 4.52e6  # Теплота взрыва тротила, Дж/кг
        Z = 0.5  # Коэффициент участия

        M_tnt = (self.params.M * self.params.Qst * Z) / Q_tnt
        return M_tnt

    def _find_radius(self, target_pressure: float) -> float:
        """
        Находит радиус зоны с заданным давлением
        """
        initial_step = 1.0
        radius = initial_step
        max_iterations = 1000
        iterations = 0

        while self.calculate_pressure(radius) > target_pressure:
            radius *= 1.5
            iterations += 1
            if iterations > 50:
                return 0  # Если не найдено

        left = radius / 1.5
        right = radius

        while (right - left) > 0.1 and iterations < max_iterations:
            mid = (left + right) / 2
            pressure = self.calculate_pressure(mid)

            if abs(pressure - target_pressure) < target_pressure * 0.01:
                return mid
            elif pressure > target_pressure:
                left = mid
            else:
                right = mid

            iterations += 1

        return (left + right) / 2

    def calculate_damage_zones(self) -> Dict[str, float]:
        """Расчет радиусов зон повреждений"""
        zones = {
            "полные_разрушения": self._find_radius(100000),  # 100 кПа
            "сильные_разрушения": self._find_radius(53000),  # 53 кПа
            "средние_разрушения": self._find_radius(28000),  # 28 кПа
            "умеренные_разрушения": self._find_radius(12000),  # 12 кПа
            "малые_повреждения": self._find_radius(5000)  # 5 кПа
        }
        return zones

    def plot_pressure_and_speed(self, max_distance: float = 100, points: int = 1000,
                                filename: str = 'explosion_parameters.png'):
        """Построение графиков распределения давления и скорости фронта пламени"""
        distances = np.linspace(1, max_distance, points)
        pressures = [self.calculate_pressure(r) / 1000 for r in distances]  # кПа
        speeds = [self.calculate_flame_speed(r) for r in distances]  # м/с

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # График давления
        ax1.plot(distances, pressures, 'b-', label='Давление')
        ax1.set_xlabel('Расстояние, м')
        ax1.set_ylabel('Избыточное давление, кПа')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()

        # График скорости
        ax2.plot(distances, speeds, 'r-', label='Скорость фронта пламени')
        ax2.set_xlabel('Расстояние, м')
        ax2.set_ylabel('Скорость, м/с')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# Пример использования
if __name__ == "__main__":
    # Пример расчета для пропано-воздушной смеси
    params = TVSParams(
        M=100,  # 100 кг пропана
        sigma=7.0,  # степень расширения
        class_space=4,  # средне загроможденное пространство
        substance_class=SubstanceClass.CLASS_3,  # чувствительное вещество
        Cst=0.074,  # стехиометрическая концентрация
        Qst=46.3e6  # теплота сгорания стехиометрической смеси
    )

    calculator = DeflagrationCalculator(params)

    # Расчет параметров на расстоянии 50 м
    distance = 50
    pressure = calculator.calculate_pressure(distance)
    impulse = calculator.calculate_impulse(distance)
    flame_speed = calculator.calculate_flame_speed(distance)
    tnt_equivalent = calculator.calculate_tnt_equivalent()

    print(f"\nПараметры ТВС:")
    print(f"Масса пропана: {params.M} кг")
    print(f"Энергозапас: {calculator.E / 1e6:.1f} МДж")
    print(f"Тротиловый эквивалент: {tnt_equivalent:.1f} кг")

    print(f"\nРезультаты расчета для расстояния {distance} м:")
    print(f"Избыточное давление: {pressure / 1000:.2f} кПа")
    print(f"Удельный импульс: {impulse:.2f} Па·с")
    print(f"Скорость фронта пламени: {flame_speed:.2f} м/с")

    # Расчет зон повреждения
    zones = calculator.calculate_damage_zones()
    print("\nРадиусы зон повреждений:")
    for zone, radius in zones.items():
        print(f"{zone}: {radius:.1f} м")

    # Построение графиков
    calculator.plot_pressure_and_speed()