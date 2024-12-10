# Математическое описание модели испарения жидкости

## 1. Введение

Представленная математическая модель описывает процесс испарения жидкости с учетом тепло- и массообмена с окружающей средой. Модель учитывает основные физические механизмы:
- Испарение с поверхности жидкости
- Теплообмен с окружающим воздухом
- Теплообмен с подстилающей поверхностью
- Поглощение солнечного излучения

## 2. Основные уравнения

### 2.1 Давление насыщенных паров

Давление насыщенных паров рассчитывается по уравнению Клаузиуса-Клапейрона [1]:

```
P_vap = P₀ exp[(L·M/R)·(1/T_boil - 1/T)]
```

где:
- P_vap - давление насыщенных паров [Па]
- P₀ - атмосферное давление [Па]
- L - удельная теплота парообразования [Дж/кг]
- M - молекулярная масса [кг/моль]
- R - универсальная газовая постоянная [Дж/(моль·К)]
- T_boil - температура кипения [К]
- T - текущая температура жидкости [К]

### 2.2 Массоперенос

Коэффициент массопереноса определяется на основе теории подобия [2]. Используются следующие критериальные уравнения:

Для ламинарного режима (Re < 5·10⁵):
```
Sh = 0.664·Re^(1/2)·Sc^(1/3)
```

Для турбулентного режима (Re ≥ 5·10⁵):
```
Sh = 0.037·Re^(0.8)·Sc^(1/3)
```

где:
- Sh - число Шервуда
- Re - число Рейнольдса
- Sc - число Шмидта (принято 0.7 для воздуха)

Число Рейнольдса рассчитывается по формуле:
```
Re = (u·L)/ν
```

где:
- u - скорость ветра [м/с]
- L - характерный размер (√S) [м]
- ν - кинематическая вязкость воздуха [м²/с]

### 2.3 Скорость испарения

Скорость испарения определяется по закону массопереноса [3]:
```
dm/dt = k_g·M·P_vap/(R·T)·S
```

где:
- dm/dt - скорость испарения [кг/с]
- k_g - коэффициент массопереноса [м/с]
- S - площадь поверхности испарения [м²]

### 2.4 Теплообмен

Суммарный коэффициент теплопередачи учитывает вынужденную и естественную конвекцию [4]:
```
h_total = (h_forced³ + h_natural³)^(1/3)
```

где:
- h_forced = 5.7 + 3.8·u - коэффициент вынужденной конвекции
- h_natural = 1.31·ΔT^(1/3) - коэффициент естественной конвекции
- u - скорость ветра [м/с]
- ΔT - разность температур [К]

### 2.5 Энергетический баланс

Полное уравнение энергетического баланса [5]:
```
m·Cp·dT/dt = q_solar + q_conv + q_ground + q_evap
```

где:
- q_solar = α·S·I - тепловой поток от солнечного излучения
- q_conv = h·S·(T_amb - T) - конвективный теплообмен с воздухом
- q_ground = h_g·S·(T_ground - T) - теплообмен с поверхностью
- q_evap = -L·dm/dt - тепловой поток за счет испарения
- α - коэффициент поглощения солнечного излучения
- I - интенсивность солнечного излучения [Вт/м²]
- h_g - коэффициент теплопередачи с поверхностью [Вт/(м²·К)]

## 3. Численное решение

Система дифференциальных уравнений решается методом LSODA [6]:
```
dy/dt = f(t, y)
```
где y = [T, m] - вектор состояния системы.

## 4. Ограничения модели

1. Предполагается равномерное распределение температуры по объему жидкости
2. Физические свойства жидкости считаются постоянными
3. Поверхность испарения принимается плоской и горизонтальной
4. Процесс массопереноса рассматривается как квазистационарный

## Литература

[1] Райст П. "Аэрозоли. Введение в теорию", М.: Мир, 1987. стр. 145-150.

[2] Bird, R.B., Stewart, W.E., Lightfoot, E.N. "Transport Phenomena", 2nd Ed., Wiley, 2002, Chapter 21.

[3] Mackay, D., Matsugu, R.S. "Evaporation rates of liquid hydrocarbon spills on land and water", Can. J. Chem. Eng., 51, 1973, pp. 434-439.

[4] McAdams, W.H. "Heat Transmission", 3rd ed., McGraw-Hill, 1954.

[5] Kawamura, P., MacKay, D. "The evaporation of volatile liquids", J. Hazard. Mater., 15, 1987, pp. 343-364.

[6] Hindmarsh, A.C. "ODEPACK, A Systematized Collection of ODE Solvers", Scientific Computing, North-Holland, Amsterdam, 1983, pp. 55-64.

[7] CCPS (Center for Chemical Process Safety). "Guidelines for Use of Vapor Cloud Dispersion Models", 2nd Ed., Wiley-AIChE, 1996.

[8] van den Bosch, C.J.H., Weterings, R.A.P.M. "Methods for the calculation of physical effects", TNO Yellow Book, 3rd Ed., 2005.

[9] Churchill, S.W., Chu, H.H.S. "Correlating equations for laminar and turbulent free convection from a vertical plate", Int. J. Heat Mass Transfer, 18, 1975, pp. 1323-1329.