import matplotlib.pyplot as plt

# Параметры моделирования
initial_capital = 1000  # Начальный капитал
weekly_growth_rate = 0.06  # Прирост в неделю: 6%
weeks = 52  # Количество недель

# Список для хранения значений капитала по неделям
capital_history = [initial_capital]

# Моделирование роста капитала с реинвестированием
for week in range(1, weeks + 1):
    new_capital = capital_history[-1] * (1 + weekly_growth_rate)
    capital_history.append(new_capital)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(capital_history, marker='o')
plt.title("Прирост капитала с реинвестированием (еженедельно +6%)")
plt.xlabel("Недели")
plt.ylabel("Капитал (USD)")
plt.grid(True)
plt.xticks(range(0, weeks + 1, 4))
plt.tight_layout()
plt.show()

# Вывод финального результата
capital_history[-1]
