import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

"""
plt.figure(figsize=(10, 6))
plt.hist(data['label'], bins=20, color='blue', alpha=0.7)
plt.title('Гистограмма Column1')
plt.xlabel('Column1')
plt.ylabel('Частота')
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=data['label'])
plt.title('Коробчатая диаграмма для Column1')
plt.grid()
plt.show()

# Выводим информацию о форме (размере) данных
print("Размер данных:", data.shape)

# Выводим названия столбцов
print("Названия столбцов:", data.columns)

# Выводим типы данных в столбцах
print("Типы данных в столбцах:", data.dtypes)

# Выводим первые 5 строк данных
print("Первые 5 строк данных:")
print(data.head())

# Выводим последние 5 строк данных
print("Последние 5 строк данных:")
print(data.tail())

# Выводим основную статистику по данным
print("Основная статистика:")
print(data.describe())
"""
#обучение
"""
data_train = pd.read_csv('train.csv')  # Загружаем весь набор данных
X_train = data_train['email']  # Сообщения
y_train = data_train['label']   # Метки (spam/ham)

data_test = pd.read_csv('test.csv')  # Загружаем тестовый набор
X_test = data_test['email']  # Сообщения для тестирования
y_test = data_test['label']   # Метки для тестирования
# Обработка пустых значений
X_train = X_train.fillna("")  # Заполнение NaN пустой строкой
y_train = y_train.fillna("")    # Заполнение NaN пустой строкой (если есть)
X_test = X_test.fillna("")      # Заполнение NaN пустой строкой
y_test = y_test.fillna("")      # Заполнение NaN пустой строкой

# Создание векторизатора
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  # Векторизация обучающей выборки
X_test_vectorized = vectorizer.transform(X_test)        # Векторизация тестовой выборки

# Обучение модели Naive Bayes
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)  # Обучение модели на обучающей выборке

# Оценка модели на тестовой выборке
y_test_pred = model.predict(X_test_vectorized)  # Прогнозирование на тестовой выборке
test_accuracy = accuracy_score(y_test, y_test_pred)  # Вычисление точности
print("Test Accuracy:", test_accuracy)

# Вывод матрицы ошибок
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

# Вывод отчета о классификации
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
"""

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Загрузка данных
X_train = pd.read_csv('train.csv')['email']
y_train = pd.read_csv('train.csv')['label']
X_test = pd.read_csv('test.csv')['email']
y_test = pd.read_csv('test.csv')['label']

# Обработка пустых значений
X_train = X_train.fillna("")
y_train = y_train.fillna("")
X_test = X_test.fillna("")
y_test = y_test.fillna("")

# Объединение обучающих данных в один DataFrame
train_data = pd.DataFrame({'text': X_train, 'label': y_train})

# Отделение "spam" и "ham"
ham = train_data[train_data['label'] == 'ham']
spam = train_data[train_data['label'] == 'spam']

# Oversampling spam
spam_oversampled = resample(spam, 
                            replace=True,     # Позволяем дублирование
                            n_samples=len(ham),    # Увеличиваем до количества "ham"
                            random_state=42)  # Для воспроизводимости

# Объединяем обратно в один DataFrame
train_data_balanced = pd.concat([ham, spam_oversampled])

# Создание TF-IDF векторизатора
vectorizer = TfidfVectorizer(stop_words='english')

# Векторизация сбалансированного набора
X_resampled = vectorizer.fit_transform(train_data_balanced['text'])
y_resampled = train_data_balanced['label']

# Векторизация тестового набора с использованием того же векторизатора
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Оценка модели на тестовой выборке
y_test_pred = model.predict(X_test_vectorized)


test_accuracy = accuracy_score(y_test, y_test_pred)  # Вычисление точности
print("Test Accuracy:", test_accuracy)

# Вывод матрицы ошибок
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))
# Вывод отчета о классификации
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
"""
"""Test Accuracy: 0.9799238490827276
Confusion Matrix (Test):
[[2436   31]
 [  27  395]]
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.99      0.99      2467
        spam       0.93      0.94      0.93       422

    accuracy                           0.98      2889
   macro avg       0.96      0.96      0.96      2889
weighted avg       0.98      0.98      0.98      2889"""
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import numpy as np

# Загрузка данных
X_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['email']
y_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['label']
X_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['email']
y_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['label']

# Обработка пустых значений
X_train = X_train.fillna("")
y_train = y_train.fillna("")
X_test = X_test.fillna("")
y_test = y_test.fillna("")

# Объединение обучающих данных в один DataFrame
train_data = pd.DataFrame({'text': X_train, 'label': y_train})

# Отделение "spam" и "ham"
ham = train_data[train_data['label'] == 'ham']
spam = train_data[train_data['label'] == 'spam']

# Oversampling spam
spam_oversampled = resample(spam, 
                            replace=True,     # Позволяем дублирование
                            n_samples=len(ham),    # Увеличиваем до количества "ham"
                            random_state=42)  # Для воспроизводимости

# Объединяем обратно в один DataFrame
train_data_balanced = pd.concat([ham, spam_oversampled])

# Создание TF-IDF векторизатора
vectorizer = TfidfVectorizer(stop_words='english')

# Векторизация сбалансированного набора
X_resampled = vectorizer.fit_transform(train_data_balanced['text'])
y_resampled = train_data_balanced['label'].map({'ham': 0, 'spam': 1})

# Векторизация тестового набора с использованием того же векторизатора
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели логистической регрессии
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_resampled, y_resampled)

# Обучение модели Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Получение предсказаний от двух моделей
lr_preds = lr_model.predict(X_test_vectorized)
rf_preds = rf_model.predict(X_test_vectorized)

# Объединение предсказаний моделей (ансамблирование)
y_test_pred = np.round((lr_preds + rf_preds) / 2).astype(int)

# Преобразование предсказаний обратно в строковые метки
y_test_pred = np.where(y_test_pred == 0, 'ham', 'spam')

# Вычисление точности
test_accuracy = accuracy_score(y_test, y_test_pred)  
print("Test Accuracy:", test_accuracy)

# Вывод матрицы ошибок
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

# Вывод отчета о классификации
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

"""
"""Test Accuracy: 0.980269989615784
Confusion Matrix (Test):
[[2465    2]
 [  55  367]]
Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99      2467
        spam       0.99      0.87      0.93       422

    accuracy                           0.98      2889
   macro avg       0.99      0.93      0.96      2889
weighted avg       0.98      0.98      0.98      2889"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import numpy as np

# Загрузка данных
X_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['email']
y_train = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/train.csv')['label']
X_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['email']
y_test = pd.read_csv('/home/xlordplay/test_tasks/task_type_1/test.csv')['label']

# Обработка пустых значений
X_train = X_train.fillna("")
y_train = y_train.fillna("")
X_test = X_test.fillna("")
y_test = y_test.fillna("")

# Объединение обучающих данных в один DataFrame
train_data = pd.DataFrame({'text': X_train, 'label': y_train})

# Отделение "spam" и "ham"
ham = train_data[train_data['label'] == 'ham']
spam = train_data[train_data['label'] == 'spam']

# Oversampling spam
spam_oversampled = resample(spam, 
                            replace=True,     
                            n_samples=len(ham),    
                            random_state=42)  

# Объединяем обратно в один DataFrame
train_data_balanced = pd.concat([ham, spam_oversampled])

# Создание TF-IDF векторизатора
vectorizer = TfidfVectorizer(stop_words='english')

# Векторизация сбалансированного набора
X_resampled = vectorizer.fit_transform(train_data_balanced['text'])
y_resampled = train_data_balanced['label'].map({'ham': 0, 'spam': 1})

# Векторизация тестового набора
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели логистической регрессии
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_resampled, y_resampled)

# Обучение модели Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_resampled, y_resampled)

# Получение предсказаний с изменением порога
threshold = 0.37  # Попробуйте увеличить порог
y_test_prob = (lr_model.predict_proba(X_test_vectorized)[:, 1] + rf_model.predict_proba(X_test_vectorized)[:, 1]) / 2
y_test_pred = np.where(y_test_prob > threshold, 'spam', 'ham')

# Вычисление точности
test_accuracy = accuracy_score(y_test, y_test_pred)  
print("Test Accuracy:", test_accuracy)
# Вывод матрицы ошибок
print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))
# Вывод отчета о классификации
print("Classification Report:")
print(classification_report(y_test, y_test_pred))