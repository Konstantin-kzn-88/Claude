import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class LPGProperties:
    """Свойства СУГ"""
    name: str  # Название вещества
    molecular_weight: float  # Молекулярная масса, кг/кмоль
    boiling_temp: float  # Температура кипения при атм. давлении, К
    heat_capacity: float  # Удельная теплоемкость жидкости, Дж/(кг·К)
    vaporization_heat: float  # Удельная теплота парообразования, Дж/кг
    liquid_density: float  # Плотность жидкости, кг/м³


@dataclass
class SurfaceProperties:
    """Свойства подстилающей поверхности"""
    thermal_conductivity: float  # Коэффициент теплопроводности, Вт/(м·К)
    density: float  # Плотность, кг/м³
    heat_capacity: float  # Удельная теплоемкость, Дж/(кг·К)

    @property
    def thermal_diffusivity(self) -> float:
        """Коэффициент температуропроводности, м²/с"""
        return self.thermal_conductivity / (self.density * self.heat_capacity)


class LPGSpillEvaporation:
    """Модель испарения СУГ из пролива"""

    def __init__(
            self,
            lpg: LPGProperties,
            surface: SurfaceProperties,
            initial_mass: float,  # Начальная масса пролива, кг
            spill_area: float,  # Площадь пролива, м²
            initial_temp: float,  # Начальная температура жидкости, К
            surface_temp: float  # Температура подстилающей поверхности, К
    ):
        self.lpg = lpg
        self.surface = surface
        self.initial_mass = initial_mass
        self.spill_area = spill_area
        self.initial_temp = initial_temp
        self.surface_temp = surface_temp

        # Рассчитываем долю мгновенного испарения
        self.flash_fraction = self._calculate_flash_fraction()

        # Начальная масса после мгновенного испарения
        self.remaining_mass = initial_mass * (1 - self.flash_fraction)

    def _calculate_flash_fraction(self) -> float:
        """Расчет доли мгновенного испарения"""
        return max(0, min(1,
                          self.lpg.heat_capacity *
                          (self.initial_temp - self.lpg.boiling_temp) /
                          self.lpg.vaporization_heat
                          ))

    def _calculate_evaporation_rate(self, time: float) -> float:
        """Расчет интенсивности испарения в момент времени t"""
        if time <= 0:
            return 0

        # Тепловой поток от подстилающей поверхности
        heat_flux = (
                self.surface.thermal_conductivity *
                self.spill_area *
                (self.surface_temp - self.lpg.boiling_temp) /
                np.sqrt(np.pi * self.surface.thermal_diffusivity * time)
        )

        # Интенсивность испарения
        return heat_flux / (self.lpg.vaporization_heat * self.spill_area)

    def simulate(
            self,
            duration: float,  # Время симуляции, с
            time_step: float = 0.1  # Шаг по времени, с
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Моделирование процесса испарения

        Returns:
            times: массив временных точек
            masses: массив значений массы жидкости
            results: словарь с дополнительными результатами
        """
        # Создаем временную сетку
        times = np.arange(0, duration + time_step, time_step)
        masses = np.zeros_like(times)
        evaporation_rates = np.zeros_like(times)

        # Начальные условия
        masses[0] = self.remaining_mass

        # Расчет по времени
        for i in range(1, len(times)):
            # Расчет интенсивности испарения
            evaporation_rates[i] = self._calculate_evaporation_rate(times[i])

            # Изменение массы
            mass_change = evaporation_rates[i] * self.spill_area * time_step
            masses[i] = max(0, masses[i - 1] - mass_change)

            # Если вся жидкость испарилась, прекращаем расчет
            if masses[i] == 0:
                times = times[:i + 1]
                masses = masses[:i + 1]
                evaporation_rates = evaporation_rates[:i + 1]
                break

        # Формируем результаты
        results = {
            'initial_mass': self.initial_mass,
            'flash_fraction': self.flash_fraction,
            'flash_mass': self.initial_mass * self.flash_fraction,
            'evaporation_rates': evaporation_rates,
            'total_time': times[-1],
            'average_rate': np.mean(evaporation_rates[1:])
        }

        return times, masses, results

    def plot_results(self, times: np.ndarray, masses: np.ndarray):
        """Построение графиков результатов"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # График массы жидкости
        ax1.plot(times, masses)
        ax1.set_xlabel('Время, с')
        ax1.set_ylabel('Масса жидкости, кг')
        ax1.grid(True)
        ax1.set_title('Изменение массы жидкости во времени')

        # График скорости испарения
        evaporation_rate = -np.gradient(masses, times)
        ax2.plot(times[1:], evaporation_rate[1:])
        ax2.set_xlabel('Время, с')
        ax2.set_ylabel('Скорость испарения, кг/с')
        ax2.grid(True)
        ax2.set_title('Скорость испарения')

        plt.tight_layout()
        return fig


# Пример использования:
if __name__ == "__main__":
    # Свойства пропана
    propane = LPGProperties(
        name="Пропан",
        molecular_weight=44.1,
        boiling_temp=231.1,  # -42°C
        heat_capacity=2500,
        vaporization_heat=428000,
        liquid_density=500
    )

    # Свойства бетонной поверхности
    concrete = SurfaceProperties(
        thermal_conductivity=1.28,
        density=2200,
        heat_capacity=840
    )

    # Создание модели
    model = LPGSpillEvaporation(
        lpg=propane,
        surface=concrete,
        initial_mass=1000,  # 1 тонна
        spill_area=100,  # 100 м²
        initial_temp=273.15,  # 0°C
        surface_temp=293.15  # 20°C
    )

    # Запуск расчета
    times, masses, results = model.simulate(duration=3600)  # 1 час

    # Вывод результатов
    print(f"Начальная масса: {results['initial_mass']:.1f} кг")
    print(f"Доля мгновенного испарения: {results['flash_fraction'] * 100:.1f}%")
    print(f"Масса мгновенного испарения: {results['flash_mass']:.1f} кг")
    print(f"Время полного испарения: {results['total_time']:.1f} с")
    print(f"Средняя скорость испарения: {results['average_rate']:.3f} кг/с")

    # Построение графиков
    fig = model.plot_results(times, masses)
    plt.savefig('plot_result')