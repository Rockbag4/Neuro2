import os


transcriptions_file = "C:/Games/network/text/1221-135767.trans.txt"

output_folder = "C:/Games/network/text"

os.makedirs(output_folder, exist_ok=True)

with open(transcriptions_file, "r", encoding="utf-8") as f:
    lines = f.readlines()


for line in lines:
    
    audio_name, transcription = line.strip().split(maxsplit=1)

    
    output_file = os.path.join(output_folder, f"{audio_name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)

print(f"Транскрипции успешно разделены! Создано {len(lines)} файлов.")
