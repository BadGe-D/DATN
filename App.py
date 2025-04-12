import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from numpy import array
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import re
from gtts import gTTS
from IPython.display import Audio
import googletrans
import asyncio
import nest_asyncio
from pydub.utils import which

from pydub import AudioSegment
from pydub.playback import play
import os
import cv2
from io import BytesIO
# import pyttsx3
# from playsound import playsound

# tts_engine=pyttsx3.init()
# Try reading the file with a different encoding, such as 'latin1'
try:
    df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='latin1')
except UnicodeDecodeError:
    # If 'latin1' doesn't work, try 'utf-16'
    try:
        df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='utf-16')
    except UnicodeDecodeError:
        # If neither 'latin1' nor 'utf-16' work, try 'ISO-8859-1'
        df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='ISO-8859-1')

# model needs to be defined before being used
model = InceptionV3()

# Tạo model mới, bỏ layer cuối từ inception v3
model_new = Model(model.input, model.layers[-2].output)

# Image embedding thành vector (2048, )
def encode(image):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = np.resize(image, (299, 299, 3 ))
    # Add one more dimension
    img = np.expand_dims(img, axis=0)
    # preprocess the images using preprocess_input() from inception module
    img = preprocess_input(img)

    fea_vec = model_new.predict(img) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def caption_preprocessing(text, remove_digits=True):
  # removw punctuation
  pattern=r'[^a-zA-z0-9\s]'
  text=re.sub(pattern,'',text)

  # tokenize
  text=text.split()
  # convert to lower case
  text = [word.lower() for word in text]

  # remove tokens with numbers in them
  text = [word for word in text if word.isalpha()]
  # concat string
  text =  ' '.join(text)

  # insert 'startseq', 'endseq'
  text = 'startseq ' + text + ' endseq'
  return text
df['caption(English)'] = df['caption(English)'].apply(caption_preprocessing)

word_counts = {}  # a dict : { word : number of appearances}
max_length = 0
for text in df['caption(English)']:
  words = text.split()
  max_length = len(words) if (max_length < len(words)) else max_length
  for w in words:
    try:
      word_counts[w] +=1
    except:
        word_counts[w] = 1
# Chỉ lấy các từ xuất hiện trên 1 lần
word_count_threshold = 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]


i2w = {}
w2i = {}

id = 1
for w in vocab:
    w2i[w] = id
    i2w[id] = w
    id += 1


# images = {}
# captions = {}

# start = time()
# for i in range(len(df)):
#   images[df['image'][i]] = np.array(Image.open(image_path + df['image'][i])) # Make sure 'image_path' is correctly defined
#   try:
#     captions[df['image'][i]].append(df['caption(English)'][i])
#   except:
#     captions[df['image'][i]] = [df['caption(English)'][i]]

MyModel= tf.keras.models.load_model('D:\Python Project\DATN\model.keras')

def TestingBeamSearch(photo, beam_width=3):
    sequences = [[list(), 0.0]]  # Initialize with an empty sequence and score of 0.0
    for _ in range(42):  # Iterate up to the maximum sequence length
        all_candidates = []
        for seq, score in sequences:
            sequence = [w2i[w] for w in seq if w in w2i]
            sequence = pad_sequences([sequence], maxlen=42, padding="post")
            yhat = MyModel.predict([photo, sequence], verbose=0)[0]
            # Consider the top beam_width predictions
            top_indices = np.argsort(yhat)[-beam_width:]
            for idx in top_indices:
                candidate = seq + [i2w[idx]]
                candidate_score = score - np.log(yhat[idx])  # Use log probability for numerical stability
                all_candidates.append([candidate, candidate_score])
        # Select the top beam_width sequences
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]
    # Return the sequence with the best score
    final_sequence = sequences[0][0]
    # Remove all occurrences of 'startseq' and 'endseq'
    final_sequence = [word for word in final_sequence if word not in ['startseq', 'endseq']]
    return ' '.join(final_sequence)



try:
    screenshot_filename = 'D:\\Python Project\\DATN\\test.jpg'  # Use the screenshot file saved earlier
    img = Image.open(screenshot_filename)
    # Assuming 'encode' and 'TestingBeamSearch' functions are defined as in your previous code
    image = encode(np.array(img))  # Encode the image
    image = image.reshape((1, 2048))
    predict = TestingBeamSearch(photo=image)
    print(f"Predicted caption: {predict}")
except Exception as e:
    print(f"Error processing image: {e}")

# # Khởi tạo video capture từ camera mặc định
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Không thể mở camera")
#     exit()

# # Lấy kích thước khung hình
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Đặt FPS và codec
# fps = 30
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# # Biến để theo dõi thời gian và trạng thái chụp
# start_time = time()
# delay = 5  # Thời gian chờ trước khi chụp (giây)
# has_captured = False  # Cờ để đảm bảo chỉ chụp một lần

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không nhận được khung hình. Kết thúc...")
#         break
    
#     # Ghi khung hình vào video
#     out.write(frame)
    
#     # Hiển thị khung hình
#     cv2.imshow('Camera Feed', frame)
    
#     # Kiểm tra thời gian để chụp ảnh một lần
#     current_time = time()
#     if not has_captured and (current_time - start_time >= delay):
#         screenshot_filename = 'screenshot.png'
#         cv2.imwrite(screenshot_filename, frame)
#         print(f"Đã tự động lưu ảnh chụp màn hình: {screenshot_filename}")
#         has_captured = True  # Đặt cờ để không chụp lại
    
#     # Thoát khi nhấn 'q'
#     if has_captured:
#         break

# # Giải phóng tài nguyên
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def translate_text(text, target_language='vi'):
    translator = googletrans.Translator()
    # Use await to get the result of the coroutine
    translated = await translator.translate(text, dest=target_language)
    return translated.text

def text_to_speech_translation(text, target_language='vi'):
    try:
        loop = asyncio.get_event_loop()
        translated_text = loop.run_until_complete(translate_text(text, target_language))
        
        tts = gTTS(text=translated_text, lang=target_language)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        
        mp3_fp.seek(0)
        audio = AudioSegment.from_file(mp3_fp, format="mp3")
        play(audio)
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

if predict:
    text_to_speech_translation(predict)
    # os.system(f"start {audio_file}")  # Automatically play the audio file
else:
    print("No prediction available for text-to-speech.")