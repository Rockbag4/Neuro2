import os
import textgrid

# Функция для разделения предложения на слова
def split_sentence_into_words(textgrid_file, output_dir):
    try:
        # Загружаем TextGrid файл
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        words = []
        
        # Ищем уровень со словами
        word_tier = None
        for tier in tg.tiers:
            if tier.name == "words":  # Ищем уровень с именем "words"
                word_tier = tier
                break
        
        # Если уровень со словами не найден, пропускаем файл
        if word_tier is None:
            print(f"Уровень со словами не найден в файле {textgrid_file}.")
            return
        
        # Проверяем, содержит ли интервал предложение или отдельные слова
        for interval in word_tier:
            if interval.mark:  # Пропускаем пустые интервалы
                sentence = interval.mark
                if " " in sentence:  # Если интервал содержит предложение
                    word_list = sentence.split()  # Разделяем предложение на слова
                    total_duration = interval.maxTime - interval.minTime
                    word_duration = total_duration / len(word_list)  # Равномерное распределение
                    
                    # Создаем интервалы для каждого слова
                    for i, word in enumerate(word_list):
                        start_time = interval.minTime + i * word_duration
                        end_time = start_time + word_duration
                        words.append({
                            'word': word,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time
                        })
                else:  # Если интервал уже содержит отдельное слово
                    words.append({
                        'word': sentence,
                        'start_time': interval.minTime,
                        'end_time': interval.maxTime,
                        'duration': interval.maxTime - interval.minTime
                    })
        
        # Проверяем, есть ли данные для обработки
        if not words:
            print(f"Файл {textgrid_file} не содержит данных для обработки.")
            return
        
        # Создаем новый TextGrid для слов
        new_tg = textgrid.TextGrid()
        new_tier = textgrid.IntervalTier(name="words", minTime=word_tier.minTime, maxTime=word_tier.maxTime)
        
        # Добавляем интервалы в новый уровень
        for word in words:
            new_tier.addInterval(textgrid.Interval(word['start_time'], word['end_time'], word['word']))
        
        # Сохраняем измененный TextGrid
        output_file = os.path.join(output_dir, os.path.basename(textgrid_file))
        new_tg.append(new_tier)
        new_tg.write(output_file)
        print(f"Файл {textgrid_file} обработан и сохранен в {output_file}")
    
    except Exception as e:
        print(f"Ошибка при обработке файла {textgrid_file}: {e}")

# Обработка всех файлов в директории
def process_textgrid_files(input_dir, output_dir):
    # Создаем директорию для сохранения обработанных файлов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем список всех .TextGrid файлов в директории
    textgrid_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".TextGrid")]
    
    # Обрабатываем каждый файл
    for file in textgrid_files:
        split_sentence_into_words(file, output_dir)

# Укажите пути к директориям
input_directory = "C:/Games/network/output"  # Директория с исходными файлами
output_directory = "C:/Games/network/output_processed"  # Директория для сохранения обработанных файлов

# Запуск обработки
process_textgrid_files(input_directory, output_directory)