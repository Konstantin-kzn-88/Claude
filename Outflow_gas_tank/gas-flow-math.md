# Математическое описание модели истечения газа из резервуара

## 1. Основные допущения

1. Газ считается идеальным
2. Процесс истечения адиабатический (отсутствует теплообмен с окружающей средой)
3. Течение газа считается одномерным
4. Пренебрегаем эффектами трения в отверстии
5. Температура и давление в резервуаре в каждый момент времени считаются одинаковыми во всем объеме

## 2. Основные уравнения

### 2.1 Уравнение состояния идеального газа

Для описания состояния газа в резервуаре используется уравнение Менделеева-Клапейрона:

```
P·V = m·R·T
```

где:
- P - давление газа [Па]
- V - объем резервуара [м³]
- m - масса газа [кг]
- R - газовая постоянная [Дж/(кг·К)]
- T - температура газа [К]

### 2.2 Массовый расход через отверстие

Для расчета массового расхода используются уравнения Сен-Венана-Ванцеля [1]. Различают два режима истечения:

#### 2.2.1 Докритический режим (β > β_кр)
```
ṁ = A·P·√(2γ/(R·T) · (β^(2/γ) - β^((γ+1)/γ))/(γ-1))
```

#### 2.2.2 Критический режим (β ≤ β_кр)
```
ṁ = A·P·√(γ/(R·T)) · (2/(γ+1))^((γ+1)/(2(γ-1)))
```

где:
- ṁ - массовый расход [кг/с]
- A - площадь отверстия [м²]
- β = P_a/P - отношение давлений (атмосферное к внутреннему)
- β_кр = (2/(γ+1))^(γ/(γ-1)) - критическое отношение давлений
- γ - показатель адиабаты

### 2.3 Система дифференциальных уравнений

Изменение параметров газа во времени описывается следующей системой уравнений:

#### 2.3.1 Изменение массы
```
dm/dt = -ṁ
```

#### 2.3.2 Изменение температуры (из уравнения адиабаты PV^γ = const)
```
dT/dt = T·(γ-1)·(dm/dt)/m
```

#### 2.3.3 Изменение давления (из уравнения состояния идеального газа)
```
dP/dt = P·((dm/dt)/m + (dT/dt)/T)
```

## 3. Начальные условия

В начальный момент времени задаются:
```
t = 0:
m(0) = P₀V/(RT₀)
P(0) = P₀
T(0) = T₀
```

## 4. Критерии остановки процесса

Процесс истечения прекращается при выполнении любого из условий:
1. P ≤ 1.01·P_a (давление близко к атмосферному)
2. m ≤ 0.1 кг (минимальная масса газа)
3. T ≤ 200 К (минимальная температура)

## Литература

[1] Абрамович Г.Н. Прикладная газовая динамика. – М.: Наука, 1991.

[2] Дейч М.Е. Техническая газодинамика. – М.: Энергия, 1974.

[3] Anderson J.D. Modern Compressible Flow: With Historical Perspective. – McGraw-Hill Education, 2003.

## Примечания по численной реализации

1. Для численного интегрирования системы дифференциальных уравнений используется метод LSODA (через scipy.integrate.odeint)

2. Шаг интегрирования подбирается автоматически для обеспечения заданной точности

3. При реализации важно соблюдать последовательность вычисления производных во избежание циклических зависимостей