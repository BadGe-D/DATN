from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import re
from gtts import gTTS
import googletrans
import asyncio
import nest_asyncio
from pydub import AudioSegment
from pydub.playback import play
import os
import cv2
from io import BytesIO
import base64

app = Flask(__name__)

# Đoạn code model của bạn
try:
    df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='latin1')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='utf-16')
    except UnicodeDecodeError:
        df = pd.read_csv("D:\\Python Project\\DATN\\dataset\\caption.csv", encoding='ISO-8859-1')

# model needs to be defined before being used
model = InceptionV3()

# Tạo model mới, bỏ layer cuối từ inception v3
model_new = Model(model.input, model.layers[-2].output)

# Image embedding thành vector (2048, )
def encode(image):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = np.resize(image, (299, 299, 3))
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

# Load model của bạn
MyModel = tf.keras.models.load_model('D:\\Python Project\\DATN\\model.keras')

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

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def translate_text(text, target_language='vi'):
    translator = googletrans.Translator()
    # Use await to get the result of the coroutine
    translated = await translator.translate(text, dest=target_language)
    return translated.text

def audio_suggestion(text, target_language='vi'):
    try:
        loop = asyncio.get_event_loop()        
        tts = gTTS(text=text, lang=target_language)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Encode audio to base64 for sending to frontend
        audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return {
            'suggestion': text,
            'audio_base64': audio_base64
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def generate_audio(text, target_language='vi'):
    try:
        loop = asyncio.get_event_loop()
        translated_text = loop.run_until_complete(translate_text(text, target_language))
        
        tts = gTTS(text=translated_text, lang=target_language)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Encode audio to base64 for sending to frontend
        audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return {
            'original_text': text,
            'translated_text': translated_text,
            'audio_base64': audio_base64
        }
    except Exception as e:
        return {
            'error': str(e)
        }

ClassifierModel=tf.keras.models.load_model('D:\\Python Project\\DATN\\ClassifierModel.keras')
def classification(photo):
    photo = cv2.resize(photo, (224, 224))
    x = np.expand_dims(photo, axis=0)
    images = np.vstack([x])
    classes = ClassifierModel.predict(images, batch_size=10)

    if classes[0] > 0.5:
        class_label = "Bạn có thể đi được"
    else:
        class_label = "Bạn không thể đi được"

    # # Dịch sang tiếng Việt
    # loop = asyncio.get_event_loop()
    # translated_text = loop.run_until_complete(translate_text(class_label, target_language='vi'))

    # print("\n Dự đoán của mô hình: ", translated_text)
    return class_label

# Route để render template HTML
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint để xử lý upload ảnh
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    
    try:
        # Xử lý ảnh
        suggestion=classification(photo=np.array(img))
        image = encode(np.array(img))
        image = image.reshape((1, 2048))
        
        # Dự đoán caption
        prediction = TestingBeamSearch(photo=image)
        
        # Tạo audio
        audio_data = generate_audio(prediction, target_language='vi')
        suggestion_audio=audio_suggestion(suggestion, target_language='vi')
        return jsonify({
            'prediction': prediction,
            'suggestion': suggestion,
            'translation': audio_data.get('translated_text', ''),
            'audio_suggestion': suggestion_audio.get('audio_base64', ''),
            'audio': audio_data.get('audio_base64', '')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)