import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Загрузка данных
data = pd.read_csv('water.csv')

# Задача 1: Модель линейной регрессии, коэффициент детерминации и график остатков

# Построение модели линейной регрессии
X = data[['hardness']]
y = data['mortality']

X = sm.add_constant(X) # добавляем константу для расчета свободного члена

model = sm.OLS(y, X)
results = model.fit()

# Вывод результатов модели
print(results.summary())

# Расчет коэффициента детерминации
print("R-squared:", results.rsquared)

# График остатков
sns.residplot(x=results.fittedvalues, y=results.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()

# Задача 2: Зависимость для северных и южных городов по отдельности

# Разделение данных на северные и южные города
north_data = data[data['location'] == 'North']
south_data = data[data['location'] == 'South']

# Модель линейной регрессии для северных городов
X_north = north_data[['hardness']]
y_north = north_data['mortality']

X_north = sm.add_constant(X_north)

model_north = sm.OLS(y_north, X_north)
results_north = model_north.fit()

# Вывод результатов модели для северных городов
print("North:")
print(results_north.summary())
print("R-squared:", results_north.rsquared)

# График остатков для северных городов
sns.residplot(x=results_north.fittedvalues, y=results_north.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual plot (North)')
plt.show()

# Модель линейной регрессии для южных городов
X_south = south_data[['hardness']]
y_south = south_data['mortality']

X_south = sm.add_constant(X_south)

model_south = sm.OLS(y_south, X_south)
results_south = model_south.fit()

# Вывод результатов модели для южных городов
print("South:")
print(results_south.summary())
print("R-squared:", results_south.rsquared)

# График остатков для южных городов
sns.residplot(x=results_south.fittedvalues, y=results_south.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual plot (South)')
plt.show()
