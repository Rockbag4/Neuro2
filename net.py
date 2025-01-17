import os  #для работы с файловой системой 
import pandas as pd  #для работы с таблицами данных
from sklearn.model_selection import train_test_split  #для разделения данных на обучающую и тестовую выборки
from sklearn.preprocessing import StandardScaler  #для масштабирования данных
from tensorflow.keras.models import Sequential  #для создания последовательных моделей нейронных сетей
from tensorflow.keras.layers import Dense, Dropout  #полносвязные слои и слой Dropout для регуляризации
from tensorflow.keras import regularizers  #для добавления регуляризации 
from tensorflow.keras.optimizers import Adam  #оптимизатор Adam для обучения модели
from sklearn.metrics import mean_absolute_error, mean_squared_error  #метрики для оценки модели
import numpy as np  #для работы с массивами и математических операций
import tgt  #чтения файлов TextGrid

#путь к директории с файлами .TextGrid
data_dir = r'C:\Games\network\output_phonemes'

#проверка существования директории
if not os.path.exists(data_dir):
    print(f"Директория {data_dir} не существует.")
    exit()  #завершение программы, если директория не существует

#загрузка и предобработка данных
def load_data(data_dir):
    data = []  #список для хранения данных
    for filename in os.listdir(data_dir):  #перебор всех файлов в директории
        if filename.lower().endswith('.textgrid'):  #проверка, что файл имеет расширение .TextGrid
            filepath = os.path.join(data_dir, filename)  #полный путь к файлу
            print(f"Обработка файла: {filepath}")
            try:
                #чтение файла TextGrid с помощью библиотеки 
                tg = tgt.io.read_textgrid(filepath)
                words_tier = tg.get_tier_by_name('words')  #получение слоя с словами
                phonemes_tier = tg.get_tier_by_name('phonemes')  #получение слоя с фонемами

                if words_tier and phonemes_tier:  #проверка, что оба слоя существуют
                    for word_interval in words_tier:  # Перебор всех интервалов в слое слов
                        word_start = word_interval.start_time  #начало слова
                        word_end = word_interval.end_time  #конец слова
                        #выбор фонем, которые находятся внутри текущего слова
                        word_phonemes = [p for p in phonemes_tier if p.start_time >= word_start and p.end_time <= word_end]
                        num_phonemes = len(word_phonemes)  #количество фонем в слове
                        for i, phoneme_interval in enumerate(word_phonemes):  #перебор всех фонем в слове
                            phoneme_label = phoneme_interval.text  #название фонемы
                            phoneme_start = phoneme_interval.start_time  #начало фонемы
                            phoneme_end = phoneme_interval.end_time  #конец фонемы
                            duration = phoneme_end - phoneme_start  #длительность фонемы
                            if duration <= 0:
                                continue  #пропустить интервалы с нулевой или отрицательной длительностью

                            #обработка ударения и фонемы
                            if phoneme_label[-1].isdigit():  #проверка, есть ли ударение 
                                stress = phoneme_label[-1]  #Ударение 
                                phoneme = phoneme_label[:-1]  #фонема без ударения
                            else:
                                stress = '0'  #если ударения нет, ставим 0
                                phoneme = phoneme_label

                            #определение позиции фонемы в слове
                            if num_phonemes == 1:
                                position_in_word = 'single'  #если фонема одна в слове
                            elif i == 0:
                                position_in_word = 'first'  #первая фонема в слове
                            elif i == num_phonemes - 1:
                                position_in_word = 'last'  #последняя фонема в слове
                            else:
                                position_in_word = 'middle'  #фонема в середине слова

                            # Добавление данных в список
                            data.append({
                                'phoneme': phoneme,
                                'stress': stress, 
                                'position_in_word': position_in_word, 
                                'duration': duration 
                            })
                else:
                    print(f"Отсутствуют 'words' или 'phonemes' в файле {filepath}")
                    continue
            except Exception as e:
                print(f"Ошибка обработки файла {filepath}: {e}")
                continue

    print(f"Длина данных: {len(data)}")
    if data:
        print(f"Первый элемент: {data[0]}")
    else:
        print("Данные не собраны.")
        return None, None, None, None, None

    #создание DataFrame
    df = pd.DataFrame(data)  #преобразование списка данных в таблицу
    df.dropna(inplace=True)  #удаление строк с пропущенными значениями
    df['duration'] = df['duration'].astype(float)  #преобразование длительности в тип float
    df.dropna(subset=['duration'], inplace=True)  #удаление строк с пропущенными значениями в колонке duration

    #преобразование категориальных признаков в one-hot encoding
    df = pd.get_dummies(df, columns=['phoneme', 'stress', 'position_in_word'])

    #разделение данных на признаки (X) и целевую переменную (y)
    X = df.drop('duration', axis=1)  #признаки (все колонки, кроме 'duration')
    y = df['duration']  #целевая переменная (длительность фонемы)

    #разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #масштабирование признаков
    scaler = StandardScaler()  #инициализация стандартизатора
    X_train = scaler.fit_transform(X_train)  #масштабирование обучающих данных
    X_test = scaler.transform(X_test)  #масштабирование тестовых данных

    print(f"Форма X_train: {X_train.shape}")  #размерность обучающих данных
    print(f"Форма y_train: {y_train.shape}")  #размерность целевой переменной
    return X_train, X_test, y_train, y_test, scaler

#определение архитектур моделей с улучшениями
def build_models(input_shape):
    models = []
    #модель 1
    model1 = Sequential()  #создание последовательной модели
    model1.add(Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(0.01)))  #полносвязный слой с 128 нейронами и L2-регуляризацией
    model1.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  #второй полносвязный слой
    model1.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  #третий полносвязный слой
    model1.add(Dense(1))  #выходной слой с одним нейроном (предсказание длительности)
    model1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  #компиляция модели с оптимизатором Adam
    models.append(model1)  #добавление модели в список

    #модель 2
    model2 = Sequential()
    model2.add(Dense(128, activation='relu', input_shape=(input_shape,)))  #первый полносвязный слой
    model2.add(Dropout(0.5))  #слой Dropout для предотвращения переобучения
    model2.add(Dense(64, activation='relu'))  #второй полносвязный слой
    model2.add(Dropout(0.5))  #еще один слой Dropout
    model2.add(Dense(32, activation='relu'))  #третий полносвязный слой
    model2.add(Dense(1))  #выходной слой
    model2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  #компиляция модели
    models.append(model2)  #добавление модели в список

    #модель 3
    model3 = Sequential()
    model3.add(Dense(256, activation='relu', input_shape=(input_shape,)))  #первый полносвязный слой с 256 нейронами
    model3.add(Dense(128, activation='relu'))  #второй полносвязный слой
    model3.add(Dense(64, activation='relu'))  #третий полносвязный слой
    model3.add(Dense(32, activation='relu'))  #четвертый полносвязный слой
    model3.add(Dense(1))  #выходной слой
    model3.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Компиляция модели
    models.append(model3)  #добавление модели в список

    return models  #возврат списка моделей

#обучение моделей с увеличенным количеством эпох
def train_models(models, X_train, y_train):
    for i, model in enumerate(models):  #перебор всех моделей
        print(f'Обучение Модели {i+1}')
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)  #обучение модели
    return models  #возврат обученных моделей

#функция для расчета процента правильно предсказанных фонем
def calculate_accuracy(y_true, y_pred, threshold=0.2):
 
    deviations = np.abs((y_true - y_pred) / y_true)  #вычисление отклонений
    correct_predictions = np.sum(deviations <= threshold)  #подсчет правильных предсказаний
    accuracy = (correct_predictions / len(y_true)) * 100  #расчет процента правильных предсказаний
    return accuracy  #возврат точности

#оценка моделей с измененным порогом
def evaluate_models(models, X_test, y_test):
    for i, model in enumerate(models):  #перебор всех моделей
        print(f'\nОценка для Модели {i+1}')
        y_pred = model.predict(X_test).flatten()  #предсказание модели
        mae = mean_absolute_error(y_test, y_pred)  #расчет MAE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  #расчет RMSE
        accuracy = calculate_accuracy(y_test, y_pred, threshold=0.1)  #расчет точности с порогом 10%
        print(f'Test MAE: {mae:.4f}')  #вывод MAE
        print(f'Test RMSE: {rmse:.4f}')  #вывод RMSE
        print(f'Процент правильно предсказанных фонем: {accuracy:.2f}%')  #вывод точности

#основное выполнение
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = load_data(data_dir)  #загрузка данных
    if X_train is not None:
        models = build_models(X_train.shape[1])  #создание моделей
        models = train_models(models, X_train, y_train)  #обучение моделей
        evaluate_models(models, X_test, y_test)  #оценка моделей