from typing import Dict, Union
from enum import Enum


class SpecialDistance(str, Enum):
    NOT_NORMALIZED = "+"  # Расстояния не нормируются
    USE_SP_4_13130 = "++"  # Расстояния принимаются в соответствии с СП 4.13130


class OilGasFacilities:
    def __init__(self):
        # Справочник объектов
        self.facilities = {
            1: "Устья эксплуатационных нефтяных и газлифтных скважин",
            2: "Замерные и сепарационные установки",
            3: "ДНС (технологические площадки)",
            4: "УПСВ",
            5: "Печи и блоки огневого нагрева нефти",
            6: "Свечи для сброса газа",
            7: "Компрессорные станции газлифта",
            8: "УПГ",
            9: "БГРА, узлы учета нефти и газа",
            10: "КНС системы ППД",
            11: "ВРП, БГ",
            12: "Дренажные, канализационные емкости",
            13: "Компрессорные воздуха",
            14: "Аппараты воздушного охлаждения",
            15: "Вспомогательные здания"
        }

        # Матрица расстояний
        self.distances: Dict[int, Dict[int, Union[int, str]]] = {
            1: {1: 5, 2: 9, 3: 30, 4: 40, 5: 40, 6: 30, 7: 40, 8: 40, 9: 9, 10: 30, 11: 9, 12: 9, 13: 15, 14: 30,
                15: 40},
            2: {2: SpecialDistance.NOT_NORMALIZED, 3: SpecialDistance.NOT_NORMALIZED, 4: SpecialDistance.NOT_NORMALIZED,
                5: 15, 6: 30, 7: 9, 8: 9, 9: SpecialDistance.NOT_NORMALIZED, 10: 9, 11: SpecialDistance.NOT_NORMALIZED,
                12: 9, 13: 9, 14: 15, 15: 40},
            3: {3: SpecialDistance.NOT_NORMALIZED, 4: SpecialDistance.NOT_NORMALIZED, 5: 15, 6: 30,
                7: SpecialDistance.NOT_NORMALIZED, 8: SpecialDistance.NOT_NORMALIZED, 9: SpecialDistance.NOT_NORMALIZED,
                10: 15, 11: 9, 12: 9, 13: 9, 14: 15, 15: 40},
            4: {4: SpecialDistance.NOT_NORMALIZED, 5: 15, 6: 30, 7: SpecialDistance.NOT_NORMALIZED,
                8: SpecialDistance.NOT_NORMALIZED, 9: SpecialDistance.NOT_NORMALIZED, 10: 15, 11: 9, 12: 9, 13: 9,
                14: 15, 15: 40},
            5: {5: SpecialDistance.NOT_NORMALIZED, 6: 30, 7: 18, 8: 18, 9: 15, 10: 15, 11: 15, 12: 9, 13: 9, 14: 9,
                15: 40},
            6: {6: SpecialDistance.NOT_NORMALIZED, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30, 13: 30, 14: 30,
                15: 100},
            7: {7: SpecialDistance.NOT_NORMALIZED, 8: 9, 9: SpecialDistance.NOT_NORMALIZED, 10: 15, 11: 9, 12: 9, 13: 9,
                14: 15, 15: 30},
            8: {8: SpecialDistance.NOT_NORMALIZED, 9: SpecialDistance.NOT_NORMALIZED, 10: 15, 11: 9, 12: 9, 13: 9,
                14: 15, 15: 30},
            9: {9: SpecialDistance.NOT_NORMALIZED, 10: 15, 11: 9, 12: 9, 13: 9, 14: 15, 15: 30},
            10: {10: SpecialDistance.NOT_NORMALIZED, 11: SpecialDistance.NOT_NORMALIZED, 12: 9, 13: 9, 14: 15, 15: 30},
            11: {11: SpecialDistance.NOT_NORMALIZED, 12: 9, 13: 9, 14: 15, 15: 30},
            12: {12: SpecialDistance.NOT_NORMALIZED, 13: 9, 14: 9, 15: 30},
            13: {13: SpecialDistance.NOT_NORMALIZED, 14: SpecialDistance.NOT_NORMALIZED, 15: 9},
            14: {14: SpecialDistance.NOT_NORMALIZED, 15: 9},
            15: {15: SpecialDistance.USE_SP_4_13130}
        }

    def get_distance(self, facility1_id: int, facility2_id: int) -> Union[int, str]:
        """
        Получить противопожарное расстояние между двумя объектами

        Args:
            facility1_id: ID первого объекта
            facility2_id: ID второго объекта

        Returns:
            Union[int, str]: Расстояние в метрах или специальное значение ("+" или "++")
        """
        # Убедимся, что ID существуют
        if facility1_id not in self.facilities or facility2_id not in self.facilities:
            raise ValueError("Неверный ID объекта")

        # Для симметричной матрицы порядок объектов не важен
        if facility1_id > facility2_id:
            facility1_id, facility2_id = facility2_id, facility1_id

        try:
            return self.distances[facility1_id][facility2_id]
        except KeyError:
            raise ValueError("Нет данных о расстоянии между указанными объектами")

    def get_facility_name(self, facility_id: int) -> str:
        """Получить название объекта по его ID"""
        if facility_id not in self.facilities:
            raise ValueError("Неверный ID объекта")
        return self.facilities[facility_id]

    def get_all_distances_for_facility(self, facility_id: int) -> Dict[str, Union[int, str]]:
        """
        Получить все расстояния для конкретного объекта

        Args:
            facility_id: ID объекта

        Returns:
            Dict[str, Union[int, str]]: Словарь с расстояниями до других объектов
        """
        if facility_id not in self.facilities:
            raise ValueError("Неверный ID объекта")

        result = {}
        for other_id in self.facilities:
            try:
                distance = self.get_distance(facility_id, other_id)
                result[self.get_facility_name(other_id)] = distance
            except ValueError:
                continue
        return result


# Пример использования
if __name__ == "__main__":
    facilities = OilGasFacilities()

    # Получить расстояние между устьем скважины и замерной установкой
    try:
        distance = facilities.get_distance(1, 2)
        print(f"Расстояние между {facilities.get_facility_name(1)} и {facilities.get_facility_name(2)}: {distance} м")
    except ValueError as e:
        print(f"Ошибка: {e}")

    # Получить все расстояния для УПСВ
    try:
        distances = facilities.get_all_distances_for_facility(4)
        print("\nРасстояния от УПСВ до других объектов:")
        for facility, distance in distances.items():
            print(f"- До {facility}: {distance}")
    except ValueError as e:
        print(f"Ошибка: {e}")