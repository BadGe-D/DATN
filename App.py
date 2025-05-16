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
import torch
import argparse
import json
import math
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import sys
sys.path.append("D:\\Python Project\\DATN")
from depth_estimation_from_image import DepthYOLOEstimator

app = Flask(__name__)

# Khởi tạo mô hình DepthYOLOEstimator
depth_estimator = DepthYOLOEstimator(yolo_model="yolo11n.pt")

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
    """
    Thực hiện beam search để tạo caption cho ảnh
    
    Args:
        photo: Vector đặc trưng của ảnh (2048 chiều)
        beam_width: Số lượng chuỗi tốt nhất được giữ lại ở mỗi bước (mặc định: 3)
    
    Returns:
        Chuỗi caption được tạo ra
    """
    # Bước 1: Khởi tạo
    # sequences chứa các cặp [chuỗi, điểm số]
    # Bắt đầu với chuỗi rỗng và điểm số 0.0
    sequences = [[list(), 0.0]]

    # Bước 2: Lặp qua từng vị trí trong chuỗi
    for _ in range(42):  # Độ dài tối đa của chuỗi là 42
        # Danh sách chứa tất cả các ứng viên có thể
        all_candidates = []
        
        # Bước 3: Xử lý từng chuỗi hiện tại
        for seq, score in sequences:
            # Kiểm tra nếu chuỗi đã kết thúc bằng 'endseq'
            if seq and seq[-1] == 'endseq':
                all_candidates.append([seq, score])
                continue
                
            # Chuyển đổi các từ thành số (index) bằng từ điển w2i
            sequence = [w2i[w] for w in seq if w in w2i]
            # Padding chuỗi để có độ dài 42
            sequence = pad_sequences([sequence], maxlen=42, padding="post")
            # Dự đoán xác suất của từ tiếp theo
            yhat = MyModel.predict([photo, sequence], verbose=0)[0]
            
            # Bước 4: Chọn beam_width từ có xác suất cao nhất
            top_indices = np.argsort(yhat)[-beam_width:]
            
            # Bước 5: Tạo các ứng viên mới
            for idx in top_indices:
                # Tạo chuỗi mới bằng cách thêm từ mới
                candidate = seq + [i2w[idx]]
                # Tính điểm mới (sử dụng log để tránh tràn số)
                candidate_score = score - np.log(yhat[idx])
                # Thêm vào danh sách ứng viên
                all_candidates.append([candidate, candidate_score])
        # Bước 6: Chọn beam_width chuỗi tốt nhất
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]
        
        # Kiểm tra nếu tất cả các chuỗi đã kết thúc bằng 'endseq'
        if all(seq[-1] == 'endseq' for seq, _ in sequences):
            break

    # Bước 7: Xử lý kết quả cuối cùng
    # Chọn chuỗi có điểm số tốt nhất
    final_sequence = sequences[0][0]
    # Loại bỏ các token đặc biệt
    final_sequence = [word for word in final_sequence if word not in ['startseq', 'endseq']]
    # Nối các từ thành câu
    result = ' '.join(final_sequence)
    
    # Bước 8: Xử lý từ đồng nghĩa
    # Thay thế một số từ cụ thể bằng từ đồng nghĩa
    result = result.replace("yên tĩnh", "trống trải")
    result = result.replace("rõ ràng", "trống trải")
    
    return result

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

def process_depth_estimation(image_array):
    """
    Xử lý ảnh với mô hình DepthYOLOEstimator
    
    Args:
        image_array: Numpy array của ảnh
        
    Returns:
        advice_text: Chuỗi lời khuyên từ phương thức advice
        image_base64: Ảnh kết quả dạng base64
    """
    try:
        # Tạo tệp ảnh tạm thời
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, image_array)
        
        # Xử lý ảnh bằng DepthYOLOEstimator
        combined_image, detected_objects = depth_estimator.process_image(
            image_path=temp_image_path,
            save_path=None,
            show_result=False
        )
        
        # Lấy lời khuyên
        advice_text = depth_estimator.advice(detected_objects)
        
        # Chuyển đổi ảnh kết quả sang base64
        _, buffer = cv2.imencode('.jpg', combined_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Xóa tệp tạm
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        return advice_text, image_base64
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh với DepthYOLOEstimator: {e}")
        return f"Lỗi: {str(e)}", None

def generate_advice_audio(advice_text, target_language='vi'):
    """
    Tạo audio từ văn bản lời khuyên
    
    Args:
        advice_text: Chuỗi lời khuyên
        target_language: Ngôn ngữ đích
        
    Returns:
        audio_base64: Chuỗi base64 của audio
    """
    try:
        tts = gTTS(text=advice_text, lang=target_language)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Encode audio to base64 for sending to frontend
        audio_base64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"Lỗi khi tạo audio lời khuyên: {e}")
        return None

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
        # Khôi phục phần gợi ý từ classification
        suggestion=classification(photo=np.array(img))
        
        # Xử lý ảnh cho mô tả
        image = encode(np.array(img))
        image = image.reshape((1, 2048))
        
        # Dự đoán caption
        prediction = TestingBeamSearch(photo=image)
        
        # Tạo audio cho caption
        audio_data = generate_audio(prediction, target_language='vi')
        
        # Khôi phục audio cho suggestion
        suggestion_audio=audio_suggestion(suggestion, target_language='vi')
        
        # Xử lý phân tích độ sâu và đưa ra lời khuyên
        img_array = np.array(img)
        # Chuyển đổi từ RGB sang BGR cho OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        depth_advice, depth_image = process_depth_estimation(img_array)
        depth_advice_audio = generate_advice_audio(depth_advice, target_language='vi')
        
        return jsonify({
            'prediction': prediction,
            'suggestion': suggestion,
            'translation': audio_data.get('translated_text', ''),
            'audio_suggestion': suggestion_audio.get('audio_base64', ''),
            'audio': audio_data.get('audio_base64', ''),
            'depth_advice': depth_advice,
            'depth_image': depth_image,
            'depth_advice_audio': depth_advice_audio
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)