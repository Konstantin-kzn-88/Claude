import math
import numpy as np


class FireBlockingTime:
    def __init__(self, room_width, room_length, room_height, h_pl=0, delta=0,
                 initial_temp=20, material_type="1.1"):
        """
        Инициализация параметров помещения

        Args:
            room_width (float): Ширина помещения, м
            room_length (float): Длина помещения, м
            room_height (float): Высота помещения, м
            h_pl (float): Высота площадки над полом, м
            delta (float): Разность высот пола, м
            initial_temp (float): Начальная температура воздуха, °C
            material_type (str): Тип помещения по таблице 1
        """
        self.width = room_width
        self.length = room_length
        self.H = room_height
        self.h_pl = h_pl
        self.delta = delta
        self.t0 = initial_temp

        # Параметры по типу помещения из таблицы 1
        self.fire_props = {
            # 1. Учебные аудитории
            "1.1": {  # Лекционная аудитория
                "name": "Лекционная аудитория",
                "Qn": 14.0, "psi": 0.0137, "v": 0.015, "Dm": 47.7,
                "LCO2": 1.478, "LCO": 0.03, "LHCl": 0.0058, "LO2": 1.369
            },
            "1.2": {  # Аудитория для практических занятий
                "name": "Аудитория для практических занятий",
                "Qn": 14.0, "psi": 0.0137, "v": 0.015, "Dm": 47.7,
                "LCO2": 1.478, "LCO": 0.03, "LHCl": 0.0058, "LO2": 1.369
            },
            # 2. Компьютерные классы
            "2.0": {  # Компьютерный класс (15 комп)
                "name": "Компьютерный класс",
                "Qn": 20.9, "psi": 0.0076, "v": 0.0125, "Dm": 327.0,
                "LCO2": 0.375, "LCO": 0.0556, "LHCl": 0.0054, "LO2": 1.95
            },
            # 3. Спортивные помещения
            "3.1": {  # Большой спортивный зал
                "name": "Большой спортивный зал",
                "Qn": 13.9, "psi": 0.0225, "v": 0.0151, "Dm": 64.1,
                "LCO2": 0.724, "LCO": 0.0205, "LHCl": 0.0005, "LO2": 1.191
            },
            "3.5": {  # Раздевалка
                "name": "Раздевалка",
                "Qn": 23.3, "psi": 0.013, "v": 0.0835, "Dm": 129.0,
                "LCO2": 0.467, "LCO": 0.0145, "LHCl": 0.0, "LO2": 3.698
            },
            # 4. Специализированные учебные аудитории
            "4.1": {  # Военная кафедра
                "name": "Военная кафедра",
                "Qn": 14.0, "psi": 0.0152, "v": 0.0163, "Dm": 53.0,
                "LCO2": 1.423, "LCO": 0.023, "LHCl": 0.0001, "LO2": 1.218
            },
            # 5. Учебно-вспомогательные помещения
            "5.3": {  # Читальный зал
                "name": "Читальный зал",
                "Qn": 14.5, "psi": 0.011, "v": 0.0103, "Dm": 49.5,
                "LCO2": 1.1087, "LCO": 0.0974, "LHCl": 0.0, "LO2": 1.154
            },
            "5.6": {  # Препараторская
                "name": "Препараторская",
                "Qn": 26.6, "psi": 0.033, "v": 812.8, "Dm": 88.1,
                "LCO2": 1.912, "LCO": 0.262, "LHCl": 0.0, "LO2": 2.304
            },
            # 6. Административные помещения
            "6.1": {  # Кабинет руководителя
                "name": "Кабинет руководителя",
                "Qn": 14.0, "psi": 0.021, "v": 0.022, "Dm": 53.0,
                "LCO2": 1.434, "LCO": 0.043, "LHCl": 0.0, "LO2": 1.161
            },
            # 7. Научно-исследовательские помещения
            "7.1": {  # Аспирантская
                "name": "Аспирантская",
                "Qn": 14.0, "psi": 0.0129, "v": 0.042, "Dm": 53.0,
                "LCO2": 0.642, "LCO": 0.0317, "LHCl": 0.0, "LO2": 1.161
            },
            # 8. Служебные помещения
            "8.1": {  # АТС
                "name": "АТС",
                "Qn": 31.7, "psi": 0.0233, "v": 0.0068, "Dm": 487.0,
                "LCO2": 1.295, "LCO": 0.097, "LHCl": 0.0109, "LO2": 2.64
            },
            "8.8": {  # Гардероб
                "name": "Гардероб",
                "Qn": 23.3, "psi": 0.013, "v": 0.0835, "Dm": 129.0,
                "LCO2": 0.467, "LCO": 0.0145, "LHCl": 0.0, "LO2": 3.698
            },
            "8.11": {  # Коридор, лестничная клетка, холл
                "name": "Коридор, лестничная клетка, холл",
                "Qn": 14.7, "psi": 0.0145, "v": 0.0108, "Dm": 82.0,
                "LCO2": 1.285, "LCO": 0.0022, "LHCl": 0.006, "LO2": 1.437
            },
            "8.12": {  # Котельная (нефть)
                "name": "Котельная (нефть)",
                "Qn": 44.2, "psi": 0.02410, "v": 885.0, "Dm": 438.0,
                "LCO2": 3.104, "LCO": 0.161, "LHCl": 0.0, "LO2": 3.24
            },
            "8.15": {  # Подсобное помещение
                "name": "Подсобное помещение",
                "Qn": 13.8, "psi": 0.0344, "v": 0.0465, "Dm": 270.0,
                "LCO2": 0.203, "LCO": 0.0022, "LHCl": 0.014, "LO2": 1.03
            },
            "8.19": {  # Серверная
                "name": "Серверная",
                "Qn": 30.7, "psi": 0.0244, "v": 0.0071, "Dm": 521.0,
                "LCO2": 0.65, "LCO": 0.1295, "LHCl": 0.0202, "LO2": 2.19
            }
        }

        # Константы
        self.cp = 0.001  # МДж/(кг·К), удельная изобарная теплоемкость воздуха
        self.phi = 0.3  # коэффициент теплопотерь
        self.Xox_a = 0.21  # начальная концентрация кислорода
        self.alpha = 0.3  # коэффициент отражения предметов
        self.E = 50  # начальная освещенность, лк
        self.l_pr = 20  # предельная дальность видимости, м

        # Критические значения ОФП
        self.t_cr = 70  # °C
        self.X_CO2_cr = 0.11  # кг/м³
        self.X_CO_cr = 1.16e-3  # кг/м³
        self.X_HCl_cr = 23e-6  # кг/м³

        # Установка параметров материала
        self.props = self.fire_props[material_type]

        # Расчет параметров помещения
        self.n = 3  # показатель степени (для кругового развития пожара)
        self.z = self._calc_z()
        self.V = self._calc_V()
        self.eta = self._calc_eta()
        self.A = self._calc_A()
        self.B = self._calc_B()

    def _calc_h(self):
        """Расчет высоты рабочей зоны"""
        return self.h_pl + 1.7 - 0.5 * self.delta

    def _calc_z(self):
        """Расчет параметра z"""
        h = self._calc_h()
        if self.H <= 6:
            return (h / self.H) * math.exp(1.4 * h / self.H)
        return 1.0

    def _calc_V(self):
        """Расчет свободного объема помещения"""
        return 0.8 * self.width * self.length * self.H

    def _calc_A(self):
        """Расчет параметра A для кругового распространения пожара"""
        return 1.05 * self.props["psi"] * self.props["v"] ** 2

    def _calc_eta(self):
        """Расчет коэффициента полноты горения"""
        return 0.63 + 0.2 * self.Xox_a + 1500 * self.Xox_a ** 6

    def _calc_B(self):
        """Расчет комплекса B"""
        return 353 * self.cp * self.V / ((1 - self.phi) * self.eta * self.props["Qn"])

    def calc_temperature_time(self):
        """Расчет критического времени по повышенной температуре"""
        try:
            arg = 1 + (70 - self.t0) / (273 + self.t0) * self.z
            if arg <= 0:
                return float('inf')
            base = self.B / self.A * math.log(arg)
            if base < 0:
                return float('inf')
            return base ** (1 / self.n)
        except:
            return float('inf')

    def calc_visibility_time(self):
        """Расчет критического времени по потере видимости"""
        try:
            ln_arg = 1 - (self.V * math.log(1.05 * self.alpha * self.E)) / \
                     (self.l_pr * self.B * self.props["Dm"] * self.z)
            if ln_arg <= 0:
                return float('inf')
            base = self.B / self.A * math.log(ln_arg) ** (-1)
            if base < 0:
                return float('inf')
            return base ** (1 / self.n)
        except:
            return float('inf')

    def calc_oxygen_time(self):
        """Расчет критического времени по пониженному содержанию кислорода"""
        try:
            ln_arg = 1 - 0.044 / ((self.B * self.props["LO2"] / self.V + 0.27) * self.z)
            if ln_arg <= 0:
                return float('inf')
            base = self.B / self.A * math.log(ln_arg) ** (-1)
            if base < 0:
                return float('inf')
            return base ** (1 / self.n)
        except:
            return float('inf')

    def calc_toxic_time(self, X_cr, L):
        """Расчет критического времени по токсичным продуктам"""
        try:
            ln_arg = 1 - (self.V * X_cr) / (self.B * L * self.z)
            if ln_arg <= 0:
                return float('inf')
            base = self.B / self.A * math.log(ln_arg) ** (-1)
            if base < 0:
                return float('inf')
            return base ** (1 / self.n)
        except:
            return float('inf')

    def calculate_blocking_time(self):
        """Расчет времени блокирования путей эвакуации"""
        times = {
            "temperature": self.calc_temperature_time(),
            "visibility": self.calc_visibility_time(),
            "oxygen": self.calc_oxygen_time(),
            "CO2": self.calc_toxic_time(self.X_CO2_cr, self.props["LCO2"]),
            "CO": self.calc_toxic_time(self.X_CO_cr, self.props["LCO"]),
            "HCl": self.calc_toxic_time(self.X_HCl_cr, self.props["LHCl"])
        }

        # Отфильтруем бесконечные значения
        finite_times = {k: v for k, v in times.items() if v != float('inf')}

        if finite_times:
            min_time = min(finite_times.values())
            times["minimum"] = min_time
        else:
            times["minimum"] = float('inf')

        return times


# Пример использования
if __name__ == "__main__":
    # Параметры помещения
    room = FireBlockingTime(
        room_width=4.0,  # ширина помещения 4 м
        room_length=5.0,  # длина помещения 5 м
        room_height=3.0,  # высота помещения 3 м
        material_type="1.1"  # тип помещения - лекционная аудитория
    )

    # Расчет времени блокирования
    blocking_times = room.calculate_blocking_time()

    # Вывод результатов
    print("Время блокирования по ОФП (секунды):")
    for factor, time in blocking_times.items():
        if time == float('inf'):
            print(f"{factor}: не представляет опасности")
        else:
            print(f"{factor}: {time:.2f}")