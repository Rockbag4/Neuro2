import os
import textgrid
from g2p import G2P

input_dir = r'C:\Games\network\output_processed'
output_dir = r'C:\Games\network\output_phonemes'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convert_words_to_phonemes(input_file, output_file):
   
    tg = textgrid.TextGrid.fromFile(input_file)
    
    
    words_tier = None
    for tier in tg.tiers:
        if tier.name == "words":
            words_tier = tier
            break
    if not words_tier:
        print(f"No 'words' tier found in {input_file}. Skipping.")
        return
    
    
    phoneme_tier = textgrid.IntervalTier(name="phonemes", minTime=tg.minTime, maxTime=tg.maxTime)
    g2p = G2P()
    
    current_time = tg.minTime
    for interval in words_tier.intervals:
        word = interval.mark
        start_time = interval.minTime
        end_time = interval.maxTime
        duration = end_time - start_time
        
    
        if not word.strip():
            current_time += duration
            continue
        
       
        try:
            phonemes = g2p(word)
        except Exception as e:
            print(f"Error converting word '{word}' in {input_file}: {e}")
            phonemes = []
        
        
        if phonemes:
            phoneme_duration = duration / len(phonemes)
            for phoneme in phonemes:
                phoneme_tier.addInterval(textgrid.Interval(current_time, current_time + phoneme_duration, phoneme))
                current_time += phoneme_duration
        else:
            
            current_time += duration
    

    tg.tiers.append(phoneme_tier)
    
    
    tg.write(output_file)


for filename in os.listdir(input_dir):
    if filename.endswith(".TextGrid"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        convert_words_to_phonemes(input_file, output_file)