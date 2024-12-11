import numpy as np
from scipy.integrate import odeint
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда перед импортом pyplot
import matplotlib.pyplot as plt
from pathlib import Path


class HeavyGasDispersion:
    def __init__(self,
                 gas_density,  # плотность газа, кг/м3
                 air_density,  # плотность воздуха, кг/м3
                 wind_speed,  # скорость ветра, м/с
                 release_rate,  # скорость выброса, кг/с
                 release_duration,  # длительность выброса, с
                 z0):  # шероховатость поверхности, м

        self.gas_density = gas_density
        self.air_density = air_density
        self.wind_speed = max(0.1, wind_speed)  # Минимальная скорость ветра
        self.release_rate = release_rate
        self.release_duration = max(0.1, release_duration)  # Минимальная длительность
        self.z0 = max(0.001, z0)  # Минимальная шероховатость

        # Константы модели
        self.g = 9.81  # ускорение свободного падения
        self.k = 0.4  # постоянная Кармана

    def primary_cloud(self, time, distance):
        """Расчет параметров первичного облака"""

        # Защита от нулевых значений
        time = max(0.1, time)
        distance = max(0.1, distance)

        # Начальный объем облака
        V0 = self.release_rate * self.release_duration / self.gas_density

        # Начальная высота облака (по формуле Бриттера-Маккуина)
        h0 = (V0 / (np.pi * (self.gas_density / self.air_density))) ** (1 / 3)

        # Скорость гравитационного растекания
        u_grav = 1.1 * np.sqrt(self.g * h0 * (self.gas_density / self.air_density - 1))

        # Эффективная скорость переноса
        u_eff = np.sqrt(u_grav ** 2 + self.wind_speed ** 2)

        # Размеры облака с защитой от нулевых значений
        width = max(0.1, 2 * np.sqrt(distance * u_grav / u_eff))
        height = max(0.1, h0 * np.exp(-0.4 * distance / u_eff))
        length = max(0.1, self.wind_speed * time)

        # Концентрация в центре облака
        concentration = (self.release_rate * self.release_duration) / (width * height * length)

        return {
            'width': width,
            'height': height,
            'length': length,
            'concentration': concentration
        }

    def secondary_cloud(self, time, distance):
        """Расчет параметров вторичного облака"""

        # Защита от нулевых значений
        time = max(0.1, time)
        distance = max(0.1, distance)

        # Динамическая скорость
        u_star = self.wind_speed * self.k / np.log(10 / self.z0)

        # Коэффициенты дисперсии с учетом времени
        sigma_y = max(0.1, 0.08 * distance * (1 + 0.0001 * distance) ** (-0.5) *
                      (1 + 0.0015 * time) ** 0.25)
        sigma_z = max(0.1, 0.06 * distance * (1 + 0.0015 * distance) ** (-0.5) *
                      (1 + 0.0003 * time) ** 0.15)

        # Плотностные эффекты с затуханием по времени
        density_factor = np.exp(-0.1 * time / self.release_duration)
        transport_speed = max(0.1, self.wind_speed *
                              (1 - 0.5 * (self.gas_density / self.air_density - 1) * density_factor))

        # Эффективная скорость выброса
        if time <= self.release_duration:
            effective_release_rate = self.release_rate
        else:
            effective_release_rate = self.release_rate * np.exp(-0.5 * (time - self.release_duration) /
                                                                self.release_duration)

        # Концентрация по модифицированной гауссовой модели
        concentration = (effective_release_rate / (2 * np.pi * sigma_y * sigma_z * transport_speed) *
                         np.exp(-0.5 * (0 / sigma_y) ** 2) *
                         (np.exp(-0.5 * (0 / sigma_z) ** 2) +
                          np.exp(-0.5 * (2 * 0 / sigma_z) ** 2)))

        return {
            'sigma_y': sigma_y,
            'sigma_z': sigma_z,
            'concentration': concentration
        }

    def calculate_total_concentration(self, time, distance):
        """Расчет суммарной концентрации от обоих облаков"""

        primary = self.primary_cloud(time, distance)
        secondary = self.secondary_cloud(time, distance)

        # Весовые коэффициенты для смешивания концентраций
        w1 = np.exp(-0.3 * time / self.release_duration)
        w2 = 1 - w1

        total_concentration = (w1 * primary['concentration'] +
                               w2 * secondary['concentration'])

        return total_concentration

    def calculate_concentration_field(self, times, distances):
        """Расчет поля концентраций"""
        X, T = np.meshgrid(distances, times)
        Z = np.zeros_like(X)

        for i in range(len(times)):
            for j in range(len(distances)):
                Z[i, j] = self.calculate_total_concentration(times[i], distances[j])

        return X, T, Z

    def plot_concentration_profile(self, times, distances, save_path='concentration_profile.png', dpi=300):
        """Построение и сохранение профиля концентрации с улучшенной визуализацией

        Args:
            times: массив временных точек
            distances: массив расстояний
            save_path: путь для сохранения графика
            dpi: разрешение изображения
        """
        try:
            X, T, Z = self.calculate_concentration_field(times, distances)

            # Создаем фигуру большего размера
            plt.figure(figsize=(12, 8), facecolor='white')

            # Создаем градиентную карту с более четкими уровнями
            levels = np.linspace(Z.min(), Z.max(), 15)
            contour = plt.contourf(X, T, Z, levels=levels, cmap='YlOrRd')

            # Добавляем контурные линии для лучшей читаемости
            CS = plt.contour(X, T, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)

            # Настраиваем цветовую шкалу
            cbar = plt.colorbar(contour, label='Концентрация (кг/м³)', pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_title('кг/м³', pad=10)

            # Улучшаем подписи осей
            plt.xlabel('Расстояние (м)', fontsize=12)
            plt.ylabel('Время (с)', fontsize=12)

            # Формируем заголовок с параметрами
            title = 'Профиль концентрации тяжелого газа\n'
            params = [
                f'Скорость ветра: {self.wind_speed:.1f} м/с',
                f'Скорость выброса: {self.release_rate:.1f} кг/с',
                f'Длительность: {self.release_duration:.1f} с'
            ]
            title += ', '.join(params)
            plt.title(title, fontsize=12, pad=20)

            # Настраиваем сетку и фон
            plt.grid(True, linestyle=':', color='gray', alpha=0.3)
            ax = plt.gca()
            ax.set_facecolor('white')

            # Улучшаем разметку осей
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Добавляем аннотацию с максимальной концентрацией
            max_conc = np.max(Z)
            max_distance = X[np.unravel_index(np.argmax(Z), Z.shape)]
            max_time = T[np.unravel_index(np.argmax(Z), Z.shape)]

            info_text = (f'Максимальная концентрация: {max_conc:.2e} кг/м³\n'
                         f'на расстоянии {max_distance:.1f} м\n'
                         f'через {max_time:.1f} с')

            plt.text(0.02, 0.98, info_text,
                     transform=plt.gca().transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round',
                               facecolor='white',
                               edgecolor='gray',
                               alpha=0.9))

            # Добавляем подписи к контурным линиям
            plt.clabel(CS, inline=True, fontsize=8, fmt='%.1e')

            # Настраиваем макет
            plt.tight_layout()

            # Сохраняем с высоким разрешением
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"График сохранен в файл: {save_path}")
            return X, T, Z

        except Exception as e:
            print(f"Ошибка при построении графика: {str(e)}")
            return None

    def calculate_xy_concentration(self, x_points, y_points, time):
        """Расчет распределения концентрации в плоскости X-Y для заданного момента времени

        Args:
            x_points: массив координат по оси X
            y_points: массив координат по оси Y
            time: момент времени для расчета
        """
        X, Y = np.meshgrid(x_points, y_points)
        Z = np.zeros_like(X)

        for i in range(len(y_points)):
            for j in range(len(x_points)):
                # Расчет расстояния от источника
                distance = np.sqrt(x_points[j] ** 2 + y_points[i] ** 2)
                # Расчет концентрации
                base_concentration = self.calculate_total_concentration(time, distance)
                # Учет поперечного распределения по гауссовой модели
                if distance > 0:
                    sigma_y = 0.08 * distance * (1 + 0.0001 * distance) ** (-0.5)
                    y_factor = np.exp(-0.5 * (y_points[i] / sigma_y) ** 2)
                    Z[i, j] = base_concentration * y_factor
                else:
                    Z[i, j] = base_concentration

        return X, Y, Z

    def plot_xy_concentration(self, time, x_range=(-10, 50), y_range=(-30, 30),
                              num_points=100, save_path='xy_concentration.png', dpi=300):
        """Построение распределения концентрации в плоскости X-Y

        Args:
            time: момент времени для визуализации
            x_range: диапазон по оси X (м)
            y_range: диапазон по оси Y (м)
            num_points: количество точек сетки
            save_path: путь для сохранения графика
            dpi: разрешение изображения
        """
        try:
            # Создаем сетку точек
            x_points = np.linspace(x_range[0], x_range[1], num_points)
            y_points = np.linspace(y_range[0], y_range[1], num_points)

            # Рассчитываем концентрации
            X, Y, Z = self.calculate_xy_concentration(x_points, y_points, time)

            # Создаем фигуру
            plt.figure(figsize=(15, 10), facecolor='white')

            # Создаем градиентную карту
            levels = np.linspace(Z.min(), Z.max(), 20)
            contour = plt.contourf(X, Y, Z, levels=levels, cmap='YlOrRd')

            # Добавляем контурные линии
            CS = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)

            # Настраиваем цветовую шкалу
            cbar = plt.colorbar(contour, label='Концентрация (кг/м³)', pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_title('кг/м³', pad=10)

            # Улучшаем подписи осей
            plt.xlabel('Расстояние по оси X (м)', fontsize=12)
            plt.ylabel('Расстояние по оси Y (м)', fontsize=12)

            # Формируем заголовок
            title = f'Распределение концентрации в плоскости X-Y\n'
            params = [
                f'Время: {time:.1f} с',
                f'Скорость ветра: {self.wind_speed:.1f} м/с',
                f'Скорость выброса: {self.release_rate:.1f} кг/с'
            ]
            title += ', '.join(params)
            plt.title(title, fontsize=12, pad=20)

            # Настраиваем сетку
            plt.grid(True, linestyle=':', color='gray', alpha=0.3)
            ax = plt.gca()
            ax.set_facecolor('white')
            ax.set_aspect('equal')  # Равный масштаб по осям

            # Добавляем подписи к контурным линиям
            plt.clabel(CS, inline=True, fontsize=8, fmt='%.1e')

            # Отмечаем источник выброса
            plt.plot(0, 0, 'k*', markersize=15, label='Источник')
            plt.legend()

            # Добавляем стрелку направления ветра
            plt.arrow(-8, 25, 4, 0, head_width=1, head_length=1, fc='black', ec='black')
            plt.text(-8, 27, 'Направление ветра', ha='left', va='bottom')

            # Настраиваем макет
            plt.tight_layout()

            # Сохраняем с высоким разрешением
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"График сохранен в файл: {save_path}")
            return X, Y, Z

        except Exception as e:
            print(f"Ошибка при построении графика: {str(e)}")
            return None


def save_results_to_file(times, distances, concentrations, filename='heavy_gas_results.csv'):
    """Сохранение результатов в файл"""
    try:
        with open(filename, 'w') as f:
            f.write('Time,Distance,Concentration\n')
            for i in range(len(times)):
                for j in range(len(distances)):
                    f.write(f'{times[i]},{distances[j]},{concentrations[i, j]}\n')
        print(f"Результаты сохранены в файл: {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {str(e)}")


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    model = HeavyGasDispersion(
        gas_density=2.0,  # кг/м3
        air_density=1.225,  # кг/м3
        wind_speed=3.0,  # м/с
        release_rate=10.0,  # кг/с
        release_duration=300.0,  # с
        z0=0.1  # м
    )

    # Расчет распространения
    times = np.linspace(0.1, 600, 50)  # Начинаем с 0.1 с
    distances = np.linspace(0.1, 1000, 50)  # Начинаем с 0.1 м

    # Расчет и сохранение результатов
    results = model.plot_concentration_profile(times, distances)

    if results is not None:
        X, T, Z = results
        save_results_to_file(times, distances, Z)
        print("Расчет успешно завершен")
    else:
        print("Ошибка при выполнении расчета")