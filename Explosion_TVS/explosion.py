import math
import numpy as np
from enum import Enum


class SpaceClass(Enum):
    """Классы загроможденности пространства"""
    ZAGR_1 = 1  # сильно загроможденное пространство
    ZAGR_2 = 2  # средне загроможденное пространство
    ZAGR_3 = 3  # слабо загроможденное пространство
    ZAGR_4 = 4  # свободное пространство


class SensitivityClass(Enum):
    """Классы чувствительности горючих веществ"""
    CLASS_1 = 1  # особо чувствительные вещества
    CLASS_2 = 2  # чувствительные вещества
    CLASS_3 = 3  # средне чувствительные вещества
    CLASS_4 = 4  # слабо чувствительные вещества


class TVSExplosion:
    def __init__(self):
        self.P0 = 101325  # атмосферное давление, Па
        self.C0 = 340  # скорость звука, м/с

        # Таблица режимов горения в зависимости от класса пространства и класса вещества
        self.explosion_mode_table = {
            # Класс 1 (особо чувствительные вещества)
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_1): 1,  # детонация
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_2): 2,
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_3): 3,
            (SensitivityClass.CLASS_1, SpaceClass.ZAGR_4): 3,

            # Класс 2 (чувствительные вещества)
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_1): 2,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_2): 3,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_3): 4,
            (SensitivityClass.CLASS_2, SpaceClass.ZAGR_4): 4,

            # Класс 3 (средне чувствительные вещества)
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_1): 3,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_2): 4,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_3): 5,
            (SensitivityClass.CLASS_3, SpaceClass.ZAGR_4): 5,

            # Класс 4 (слабо чувствительные вещества)
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_1): 4,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_2): 5,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_3): 6,
            (SensitivityClass.CLASS_4, SpaceClass.ZAGR_4): 6,
        }

    def calculate_energy(self, M_g, q_g, c_g=None, c_st=None, beta=1, is_ground_level=False):
        """
        Расчет эффективного энергозапаса

        Args:
            M_g: масса горючего вещества, кг
            q_g: теплота сгорания горючего газа, кДж/кг
            c_g: концентрация горючего вещества, кг/м³
            c_st: стехиометрическая концентрация, кг/м³
            beta: корректировочный параметр
            is_ground_level: находится ли облако на поверхности земли
        """
        # Переводим теплоту сгорания из кДж/кг в Дж/кг
        q_g = q_g * 1000

        # Расчет базового энергозапаса
        if c_g is None or c_st is None:
            E = M_g * q_g * beta
        else:
            if c_g <= c_st:
                E = M_g * q_g * beta
            else:
                E = M_g * q_g * beta * c_st / c_g

        # Удвоение энергозапаса для приземного облака
        if is_ground_level:
            E *= 2

        return E

    def get_explosion_mode(self, sensitivity_class: SensitivityClass, space_class: SpaceClass):
        """Определение режима взрывного превращения по классам чувствительности и загроможденности"""
        return self.explosion_mode_table.get((sensitivity_class, space_class), 6)

    def calculate_flame_velocity(self, M_g, mode):
        """Расчет скорости фронта пламени"""
        if mode == 1:  # детонация
            return 500
        elif mode == 2:
            return 400  # среднее между 300 и 500
        elif mode == 3:
            return 250  # среднее между 200 и 300
        elif mode == 4:
            return 175  # среднее между 150 и 200
        elif mode == 5:
            return 43 * math.pow(M_g, 1 / 6)
        elif mode == 6:
            return 26 * math.pow(M_g, 1 / 6)
        else:
            return None

    def calculate_dimensionless_distance(self, r, E):
        """Расчет безразмерного расстояния"""
        return r / math.pow(E / self.P0, 1 / 3)

    def calculate_pressure_deflagration(self, V_g, R_x, sigma):
        """Расчет безразмерного давления при дефлаграции"""
        term1 = math.pow(V_g / self.C0, 2)
        term2 = (sigma - 1) / sigma
        term3 = 0.83 / R_x - 0.14 / math.pow(R_x, 2)
        return term1 * term2 * term3

    def calculate_impulse_deflagration(self, V_g, R_x, sigma):
        """Расчет безразмерного импульса при дефлаграции"""
        term1 = (V_g / self.C0) * (sigma - 1) / sigma
        term2 = 1 - 0.4 * (sigma - 1) * V_g / (sigma * self.C0)
        term3 = 0.06 / R_x + 0.01 / math.pow(R_x, 2) - 0.0025 / math.pow(R_x, 3)
        return term1 * term2 * term3

    def calculate_final_parameters(self, P_x, I_x, E):
        """Расчет размерных величин давления и импульса"""
        P = P_x * self.P0
        I = I_x * math.pow(self.P0, 2 / 3) * math.pow(E, 1 / 3) / self.C0
        return P, I

    def calculate_explosion_parameters(self, r, M_g, q_g, sensitivity_class: SensitivityClass,
                                       space_class: SpaceClass, c_g=None, c_st=None, beta=1,
                                       is_gas=True, is_ground_level=False):
        """
        Расчет всех параметров взрыва

        Args:
            r: расстояние от центра взрыва, м
            M_g: масса горючего вещества, кг
            q_g: теплота сгорания горючего газа, кДж/кг
            sensitivity_class: класс чувствительности вещества
            space_class: класс загроможденности пространства
            c_g: концентрация горючего вещества, кг/м³
            c_st: стехиометрическая концентрация, кг/м³
            beta: корректировочный параметр
            is_gas: тип смеси (True - газовая, False - гетерогенная)
            is_ground_level: находится ли облако на поверхности земли
        """
        # Определяем режим взрывного превращения
        explosion_mode = self.get_explosion_mode(sensitivity_class, space_class)

        # Определяем сигму в зависимости от типа смеси
        sigma = 7 if is_gas else 4

        # Рассчитываем эффективный энергозапас
        E = self.calculate_energy(M_g, q_g, c_g, c_st, beta, is_ground_level)

        # Рассчитываем скорость фронта пламени
        V_g = self.calculate_flame_velocity(M_g, explosion_mode)

        # Рассчитываем безразмерное расстояние
        R_x = self.calculate_dimensionless_distance(r, E)

        # Проверяем критическое значение R_x
        R_kr = 0.34
        if R_x < R_kr:
            R_x = R_kr

        # Рассчитываем безразмерные параметры
        P_x = self.calculate_pressure_deflagration(V_g, R_x, sigma)
        I_x = self.calculate_impulse_deflagration(V_g, R_x, sigma)

        # Рассчитываем конечные размерные величины
        P, I = self.calculate_final_parameters(P_x, I_x, E)

        return {
            'режим_взрывного_превращения [-]': explosion_mode,
            'эффективный_энергозапас [Дж]': E,
            'скорость_фронта_пламени [м/с]': V_g,
            'безразмерное_расстояние [-]': R_x,
            'безразмерное_давление [-]': P_x,
            'безразмерный_импульс [-]': I_x,
            'избыточное_давление [Па]': P,
            'импульс_фазы_сжатия [Па·с]': I
        }


# Пример использования
if __name__ == "__main__":
    tvs = TVSExplosion()

    # Пример расчета для газовой смеси
    params = {
        'r': 100,  # расстояние от центра взрыва, м
        'M_g': 1000,  # масса горючего вещества, кг
        'q_g': 46000,  # теплота сгорания, кДж/кг
        'sensitivity_class': SensitivityClass.CLASS_2,  # класс чувствительности
        'space_class': SpaceClass.ZAGR_2,  # класс загроможденности
        'beta': 1,  # корректировочный параметр
        'is_gas': True,  # газовая смесь
        'is_ground_level': True  # облако на поверхности земли
    }

    results = tvs.calculate_explosion_parameters(**params)

    print("Результаты расчета:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")