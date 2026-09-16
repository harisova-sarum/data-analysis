# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:58:00 2024

@author: Sarum
"""

# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

students_performance = pd.read_csv('StudentsPerformance.csv')

# %% Первые 5 строк
print(students_performance.head())

# %% Срез строк с позиции 0 до 4 (пять первых строк) и столбцов с позиции 0 до 2 
print(students_performance.iloc[:5,:3])

# %% Замена индексов на имена
students_performance_with_names = students_performance.iloc[[0,3,4,7,8]]
students_performance_with_names.index = ["Cersei", "Tywin", "Gregor", "Joffrey", "Ilyn Pane"]

print(students_performance_with_names)

# %% Выбрать конкретные строки и столбцы
print("")
students_performance_with_names.loc[['Cersei','Joffrey'],['gender','writing score']]

# %% Выбрать все строки из первого столбца
print(students_performance.iloc[:,0])
print(type(students_performance.iloc[:,0]))

# %% Выбрать первые 7 строк
print(students_performance.iloc[:7])

# %% Выбрать строки, значения пола которых равно gemale и столбцы 'gender','writing score'
print(students_performance.loc[students_performance.gender == 'female',['gender','writing score']])

# %% Выбрать строки,  у которых оценка по письму больше среднего значения
mean_writing_score = students_performance['writing score'].mean()
print(mean_writing_score)
print(students_performance.loc[students_performance['writing score'] > mean_writing_score])

# %% Выбрать строки женщин, у которых оценка по письму 100
students_performance[(students_performance['writing score'] == 100) & (students_performance.gender == 'female')]

# %% Выбирать столбцы
print(students_performance.columns)

# %% Доля людей, у которых бесплатный|сниженный по цене обед
free_reduced_lunch = students_performance.loc[students_performance['lunch'] == 'free/reduced']
print(len(free_reduced_lunch)/len(students_performance))

# %% 

#print(students_performance.mean(), students_performance.var())

# %% Женщины, у которых оценка по письму больше 74
students_performance = students_performance \
  .rename(columns = 
         {'parental level of education': 'parental_level_of_education', 
          'test preparation course': 'test_preparation_course',
          'math score': 'math_score',
          'reading score': 'reading_score',
          'writing score': 'writing_score'})

print(students_performance.query("gender == 'female' & writing_score > 74"))
# %% Фильтрация тех записей, у которых есть слово lunch
print(students_performance_with_names.filter(regex = 'lunch'))
# %% Группировка по полу и вычисление среднего
print(students_performance.groupby('gender', as_index = False)\
  .aggregate({'math_score': 'mean'})\
  .rename(columns = {'math_score': 'mean_math_score'}))

# %% Группировка по полу и расе/этничности и вычисление среднего    
mean_scores = students_performance.groupby(['gender','race/ethnicity'])\
  .aggregate({'math_score': 'mean','reading_score': 'mean','writing_score': 'mean'})\
  .rename(columns = {'math_score': 'mean_math_score', 'reading_score': 'mean_reading_score', 'writing_score': 'mean_writing_score'})    

print(mean_scores)
# %% Средние женщин группы А и мужчин группы А   
print(mean_scores.loc[[('female', 'group A'),('male', 'group A')]])

# %% Количество уникальных оценок по чтению 
print(students_performance.groupby(['gender','race/ethnicity'])\
  .reading_score.nunique())

# %% Записи отсортированные по полу и оценкам по математике, сгруппированные по полу   
print(students_performance.sort_values(['gender','math_score'], ascending = False)\
  .groupby(['gender'], as_index = False).head(5))


# %% Логарифмическая оценка
students_performance['total_score'] = students_performance.math_score + students_performance.reading_score + students_performance.writing_score 

students_performance = students_performance.assign(log_total_score = np.log(students_performance.total_score))

students_performance.drop(['total_score'], axis = 1)

students_performance.total_score.hist()
plt.title('Гистограмма общей оценки')

# %% 

students_performance.math_score.hist()
plt.title('Гистограмма оценки по математике')
plt.show()

# %% Гистограмма
students_performance.plot.scatter(x='math_score', y='reading_score')
plt.title('Диаграмма рассеяния')
plt.show()

# %% Гистограмма
ax = sns.lmplot(x='math_score', y='reading_score', hue='gender', data=students_performance, fit_reg=False)
ax.set_xlabels('Math scores')
ax.set_ylabels('Reading scores')
plt.title('Scatter plot')
plt.show()

# %%
student_stats = pd.read_csv('StudentsPerformance.csv')
# %% Выбрать бакалавров и магистров
print(student_stats[student_stats['parental level of education'].isin(["bachelor's degree", "master's degree"])])

# %% Выбрать столбцы со score
score_columns = [i for i in list(student_stats) if 'score' in i]
print(student_stats[score_columns].head())

print(student_stats.filter(like = 'score'))




