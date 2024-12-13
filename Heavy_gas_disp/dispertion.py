import numpy as np
from scipy.integrate import odeint


class BritterMcQuaidModel:
    """Базовая модель рассеивания Бриттера-Макквейда"""

    def __init__(self, Q0, rho_a, rho_g, wind_speed, stability_class):
        """
        Инициализация модели
        Q0 - начальный расход газа (кг/с)
        rho_a - плотность воздуха (кг/м3)
        rho_g - плотность газа (кг/м3)
        wind_speed - скорость ветра (м/с)
        stability_class - класс устойчивости атмосферы (A-F)
        """
        self.Q0 = Q0
        self.rho_a = rho_a
        self.rho_g = rho_g
        self.wind_speed = wind_speed
        self.stability_class = stability_class

        # Параметры в зависимости от класса устойчивости
        self.stability_params = {
            'A': {'a': 0.32, 'b': 0.0004},
            'B': {'a': 0.32, 'b': 0.0004},
            'C': {'a': 0.22, 'b': 0.0004},
            'D': {'a': 0.16, 'b': 0.0004},
            'E': {'a': 0.11, 'b': 0.0004},
            'F': {'a': 0.11, 'b': 0.0004}
        }

    def calculate_spread_parameters(self, x):
        """
        Расчет параметров распространения
        x - расстояние от источника (м)
        """
        params = self.stability_params[self.stability_class]

        # Модифицированные формулы для тяжелого газа
        sigma_y = params['a'] * x * (1 + params['b'] * x) ** (-0.5)
        sigma_z = min(params['a'] * x * (1 + params['b'] * x) ** (-0.5), 50.0)  # Ограничение высоты

        return sigma_y, sigma_z

    def calculate_concentration(self, x, y, z):
        """
        Расчет концентрации газа в точке (x, y, z)
        """
        # Защита от отрицательных координат
        x = max(0.1, abs(x))  # Минимальное расстояние 0.1 м

        sigma_y, sigma_z = self.calculate_spread_parameters(x)

        # Учет плавучести и ветра
        g_prime = 9.81 * (self.rho_g - self.rho_a) / self.rho_a
        h_eff = min((g_prime * self.Q0 / (np.pi * self.wind_speed ** 3)) ** (1 / 3), 10.0)

        # Модифицированный расчет концентрации для тяжелого газа
        C = (self.Q0 / (2 * np.pi * sigma_y * sigma_z * self.wind_speed)) * \
            np.exp(-0.5 * (y / sigma_y) ** 2) * \
            (np.exp(-0.5 * ((z - h_eff) / sigma_z) ** 2) + \
             np.exp(-0.5 * ((z + h_eff) / sigma_z) ** 2))

        # Учет гравитационного оседания
        if z < h_eff:
            C *= (1 + (h_eff - z) / h_eff)

        return C


import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ReleaseParameters:
    """Параметры выброса"""
    mass_rate: float  # скорость выброса, кг/с
    duration: float  # длительность выброса, с
    temperature: float  # температура выброса, K
    pressure: float  # давление выброса, Па
    hole_diameter: float  # диаметр отверстия, м


@dataclass
class AtmosphericConditions:
    """Атмосферные условия"""
    temperature: float  # температура воздуха, K
    pressure: float  # атмосферное давление, Па
    humidity: float  # относительная влажность, %
    wind_speed: float  # скорость ветра, м/с
    stability_class: str  # класс устойчивости атмосферы (A-F)


class PrimaryCloud:
    """Модель первичного облака"""

    def __init__(self, release_params: ReleaseParameters, atm_conditions: AtmosphericConditions):
        self.release = release_params
        self.atm = atm_conditions

        # Физические константы
        self.R = 8.314  # универсальная газовая постоянная
        self.g = 9.81  # ускорение свободного падения

    def calculate_flash_fraction(self) -> float:
        """Расчет доли мгновенно испарившейся жидкости"""
        # Упрощенный расчет для примера
        delta_T = self.release.temperature - self.atm.temperature
        Cp = 2000  # удельная теплоемкость (зависит от вещества)
        L = 400000  # удельная теплота парообразования
        flash_fraction = min(1.0, max(0.1, Cp * delta_T / L))  # Минимум 10%
        print(f"Flash fraction: {flash_fraction}")
        return flash_fraction

    def calculate_initial_volume(self) -> float:
        """Расчет начального объема облака"""
        flash_fraction = self.calculate_flash_fraction()

        # Проверка и корректировка температуры
        if self.release.temperature <= 0:
            print(f"Предупреждение: некорректная температура {self.release.temperature}K")
            self.release.temperature = 273.15

        # Расчет плотности газа
        gas_density = max(self.release.pressure / (self.R * self.release.temperature), 0.1)
        print(f"Gas density: {gas_density} kg/m³")

        # Расчет объема с учетом минимального значения
        volume = max(
            1.0,  # минимальный объем 1 м³
            self.release.mass_rate * self.release.duration * flash_fraction / gas_density
        )

        print(f"Calculated volume: {volume} m³")
        return volume

    def calculate_initial_dimensions(self) -> Tuple[float, float, float]:
        """Расчет начальных размеров облака (длина, ширина, высота)"""
        V = max(self.calculate_initial_volume(), 1e-6)  # минимальный объем

        # Предполагаем цилиндрическую форму облака с соотношением сторон
        aspect_ratio = 0.5  # отношение высоты к диаметру

        # Расчет размеров с учетом соотношения сторон
        radius = (V / (np.pi * aspect_ratio)) ** (1 / 3)
        height = 2 * radius * aspect_ratio

        # Проверка минимальных размеров
        min_size = 0.1  # минимальный размер в метрах
        radius = max(radius, min_size)
        height = max(height, min_size)

        return 2 * radius, 2 * radius, height


class CloudTransition:
    """Модель перехода от первичного ко вторичному облаку"""

    def __init__(self, primary_cloud: PrimaryCloud):
        self.primary = primary_cloud
        initial_dimensions = primary_cloud.calculate_initial_dimensions()
        # Добавляем положение центра облака
        self.center_x = 0.0
        self.center_y = 0.0
        self.current_dimensions = initial_dimensions
        self.time = 0.0
        self.spreading_coefficient = 0.3

    def gravity_spreading_rate(self, dimensions: Tuple[float, float, float]) -> Tuple[float, float]:
        """Расчет скорости гравитационного растекания"""
        length, width, height = dimensions
        # Расчет g_prime с учетом реальных плотностей
        rho_gas = self.primary.release.pressure / (self.primary.R * self.primary.release.temperature)
        rho_air = self.primary.atm.pressure / (self.primary.R * self.primary.atm.temperature)
        g_prime = self.primary.g * (rho_gas - rho_air) / rho_air

        # Расчет скорости растекания с затуханием
        spreading_factor = np.exp(-self.time / 20.0)
        dx = self.spreading_coefficient * spreading_factor * np.sqrt(g_prime * height)
        dy = 0.6 * dx  # Поперечное растекание медленнее

        return dx, dy

    def step(self, dt: float) -> None:
        """Выполнить шаг по времени"""
        length, width, height = self.current_dimensions

        # Растекание
        dx, dy = self.gravity_spreading_rate(self.current_dimensions)
        new_length = length + 2 * dx * dt
        new_width = width + 2 * dy * dt

        # Перенос ветром
        self.center_x += self.primary.atm.wind_speed * dt

        # Изменение высоты с сохранением массы
        new_height = height * (length * width) / (new_length * new_width)
        new_height = max(new_height, 0.1)  # Минимальная высота

        self.current_dimensions = (new_length, new_width, new_height)
        self.time += dt

    def get_cloud_bounds(self) -> Tuple[float, float, float, float]:
        """Получить границы облака с учетом положения центра"""
        length, width, _ = self.current_dimensions
        x_min = self.center_x - length / 2
        x_max = self.center_x + length / 2
        y_min = self.center_y - width / 2
        y_max = self.center_y + width / 2
        return x_min, x_max, y_min, y_max


class SecondaryCloud(BritterMcQuaidModel):
    """Расширенная модель вторичного облака"""

    def __init__(self, transition: CloudTransition):
        length, width, height = transition.current_dimensions
        initial_volume = length * width * height

        # Вызов родительского конструктора с рассчитанными параметрами
        super().__init__(
            Q0=initial_volume * transition.primary.release.mass_rate / transition.primary.calculate_initial_volume(),
            rho_a=transition.primary.atm.pressure / (transition.primary.R * transition.primary.atm.temperature),
            rho_g=transition.primary.release.pressure / (transition.primary.R * transition.primary.release.temperature),
            wind_speed=transition.primary.atm.wind_speed,
            stability_class=transition.primary.atm.stability_class
        )

        self.initial_dimensions = transition.current_dimensions

    def calculate_concentration(self, x: float, y: float, z: float) -> float:
        """Переопределенный метод расчета концентрации"""
        # Учитываем начальные размеры облака
        if x < self.initial_dimensions[0] / 2:
            return super().calculate_concentration(0.1, y, z)

        return super().calculate_concentration(
            x - self.initial_dimensions[0] / 2,
            y,
            z
        )


class CompleteDispersionModel:
    """Полная модель рассеивания"""

    def __init__(self, release_params: ReleaseParameters, atm_conditions: AtmosphericConditions):
        self.primary = PrimaryCloud(release_params, atm_conditions)
        self.transition = CloudTransition(self.primary)
        self.secondary = None

    def simulate(self, total_time: float, dt: float = 1.0) -> None:
        """Моделирование всего процесса рассеивания"""
        # Моделируем переход до достижения критериев
        while self.transition.time < total_time:
            self.transition.step(dt)

            # Проверяем критерии перехода к вторичному облаку
            if self.check_transition_criteria():
                self.secondary = SecondaryCloud(self.transition)
                break

    def check_transition_criteria(self) -> bool:
        """Проверка критериев перехода к вторичному облаку"""
        length, width, height = self.transition.current_dimensions

        # Критерии перехода:
        # 1. Высота облака менее 10% от ширины
        height_criterion = height / width < 0.1

        # 2. Скорость растекания мала
        dx, dy = self.transition.gravity_spreading_rate(self.transition.current_dimensions)
        spreading_speed = np.sqrt(dx ** 2 + dy ** 2)
        speed_criterion = spreading_speed < 0.05 * self.primary.atm.wind_speed

        # 3. Прошло минимальное время
        time_criterion = self.transition.time > 30.0

        return height_criterion and speed_criterion and time_criterion

    def plot_evolution(self, times: list) -> None:
        """Построение эволюции облака во времени"""
        import matplotlib
        matplotlib.use('TkAgg', force=True)
        from matplotlib import pyplot as plt

        plt.figure(figsize=(12, 4 * len(times)))

        for idx, t in enumerate(times):
            plt.subplot(len(times), 1, idx + 1)

            # Моделируем до времени t
            self.simulate(t)

            if self.secondary is None:
                # Получаем границы облака
                x_min, x_max, y_min, y_max = self.transition.get_cloud_bounds()

                # Рисуем прямоугольник облака
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=True,
                    facecolor='blue',
                    alpha=0.5
                )
                plt.gca().add_patch(rect)

            else:
                # Рисуем вторичное облако
                x = np.linspace(-50, 200, 100)
                y = np.linspace(-50, 50, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)

                for i in range(len(x)):
                    for j in range(len(y)):
                        Z[j, i] = self.secondary.calculate_concentration(
                            X[j, i], Y[j, i], 1.5
                        )

                # Используем логарифмическую шкалу для концентраций
                plt.contourf(X, Y, Z, levels=20,
                             norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1),
                             cmap='Blues')
                plt.colorbar(label='Относительная концентрация')

            plt.xlim(-50, 200)
            plt.ylim(-50, 50)
            plt.title(f't = {t:.1f} с')
            plt.xlabel('Расстояние по направлению ветра (м)')
            plt.ylabel('Поперечное расстояние (м)')
            plt.grid(True)

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Параметры выброса
    release_params = ReleaseParameters(
        mass_rate=10.0,  # 10 кг/с
        duration=60.0,  # 1 минута
        temperature=250.0,  # K
        pressure=5e5,  # 5 бар
        hole_diameter=0.1  # 10 см
    )

    # Атмосферные условия
    atm_conditions = AtmosphericConditions(
        temperature=293.15,  # 20°C
        pressure=101325.0,  # 1 атм
        humidity=70.0,  # 70%
        wind_speed=5.0,  # 5 м/с
        stability_class='D'  # нейтральная устойчивость
    )

    # Создание и запуск модели
    model = CompleteDispersionModel(release_params, atm_conditions)
    model.plot_evolution([10.0, 30.0, 60.0, 120.0])