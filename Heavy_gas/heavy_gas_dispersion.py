import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HeavyGasDispersion:
    def __init__(self, Q, T_amb, wind_speed, stability_class, z0, duration):
        # Параметры выброса
        self.Q = Q
        self.T_amb = T_amb + 273.15
        self.wind_speed = wind_speed
        self.stability_class = stability_class
        self.z0 = z0
        self.duration = duration

        # Параметры H2S
        self.M = 34.08
        self.R = 8.314

        # Расчет плотностей
        self.rho_air = 353.18 / self.T_amb
        self.rho_gas = (self.M * 101325) / (self.R * self.T_amb) / 1000
        self.density_ratio = self.rho_gas / self.rho_air

        # Параметры модели
        self.g = 9.81
        self.initial_height = 3.0  # начальная высота выброса

    def calculate_gravity_spreading(self, x):
        """Расчет растекания под действием силы тяжести"""
        if x <= 0:
            return 0

        # Характерный размер растекания
        spread = np.sqrt((self.density_ratio - 1) * self.g * self.initial_height * x / self.wind_speed ** 2)
        return max(spread, 0.1)

    def calculate_concentration(self, x, y, z):
        """Расчет концентрации с учетом эффектов тяжелого газа"""
        if x <= 0:
            return 0

        # Расчет начального растекания
        spread_height = self.initial_height * np.exp(-0.2 * x / self.initial_height)
        spread_width = self.calculate_gravity_spreading(x)

        # Модифицированные параметры дисперсии
        sigma_y = spread_width * (1 + 0.1 * x / spread_width) ** 0.5
        sigma_z = spread_height * (1 + 0.15 * x / self.initial_height) ** 0.5

        # Учет прижимания к земле
        effective_height = self.initial_height * np.exp(-0.3 * x / self.initial_height)

        # Расчет концентрации с учетом отражения от поверхности и прижимания
        C = (self.Q / (2 * np.pi * self.wind_speed * sigma_y * sigma_z) *
             np.exp(-0.5 * (y / sigma_y) ** 2) *
             (np.exp(-0.5 * ((z - effective_height) / sigma_z) ** 2) +
              np.exp(-0.5 * ((z + effective_height) / sigma_z) ** 2)))

        # Учет эффекта "обрезания" на границах облака
        edge_factor = 1 / (1 + np.exp((np.abs(z) - spread_height * 1.5) / 0.1))

        return C * edge_factor

    def calculate_toxdose(self, x, y, z):
        """Расчет токсодозы"""
        C = self.calculate_concentration(x, y, z)
        toxdose = C * self.duration / 60  # перевод в мг*мин/л
        return toxdose


def plot_results(model, x_range, z_range, output_file='toxdose_distribution.png'):
    y_pos = 0
    X, Z = np.meshgrid(x_range, z_range)
    C = np.zeros_like(X)

    for i in range(len(z_range)):
        for j in range(len(x_range)):
            C[i, j] = model.calculate_toxdose(x_range[j], y_pos, z_range[i])

    plt.figure(figsize=(12, 8))

    # Настройка цветовой схемы для соответствия примеру
    levels = np.linspace(0, 5, 20)
    contour = plt.contourf(X, Z, C, levels=levels, cmap='jet')
    plt.colorbar(contour, label='Токсодоза (мг*мин/л)')

    # Добавление линии пороговой токсодозы
    plt.contour(X, Z, C, levels=[1.0], colors='red', linewidths=2,
                linestyles='dashed', label='Пороговая токсодоза')

    plt.xlabel('Расстояние (м)')
    plt.ylabel('Высота (м)')
    plt.title('Распределение токсодозы H2S')
    plt.grid(True)

    # Настройка осей для соответствия примеру
    plt.ylim(-5, 12.5)
    plt.xlim(0, 13)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


if __name__ == "__main__":
    # Параметры задачи
    Q = 5  # кг/с
    T_amb = 30  # °C
    wind_speed = 1  # м/с
    stability_class = 'F'
    z0 = 0.1  # м (трава)
    duration = 600  # с

    # Создание модели
    model = HeavyGasDispersion(Q, T_amb, wind_speed, stability_class, z0, duration)

    # Расчетная область
    x_range = np.linspace(0.1, 13, 100)
    z_range = np.linspace(-5, 12.5, 100)

    # Построение и сохранение графика
    output_file = plot_results(model, x_range, z_range)
    print(f"График сохранен в файл: {output_file}")