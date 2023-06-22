from flask import send_file, Flask, render_template, request, make_response, send_from_directory
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('mod_new.h5')


# 웹 페이지 렌더링
@app.route('/')
def home():
    return render_template('index.html')


# 이미지 변환 및 다운로드
@app.route('/convert', methods=['POST'])
def convert():
    # 이미지 파일 받기
    image_file = request.files['image']
    image_data = image_file.read()
    image_array = np.frombuffer(image_data, np.uint8)

    # 이미지 열기
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # 이미지 전처리
    # 예시: 이미지 크기 조정 및 배열 변환
    image = cv2.resize(image, (512, 512))
    image = np.expand_dims(image, axis=0)
    image = image.reshape(-1, 512, 512, 1)

    # 예측 수행
    prediction = model.predict(image)
    prediction = prediction.reshape(512, 512, 1)

    # 변환된 이미지를 PIL 이미지 객체로 변환
    transformed_img = np.squeeze(prediction)  # 배치 차원 제거
    transformed_img = Image.fromarray(np.uint8(transformed_img))

    # 변환된 이미지를 다운로드할 수 있도록 바이트 스트림으로 변환
    output = io.BytesIO()
    transformed_img.save(output, format='PNG')
    output.seek(0)

    # 변환된 이미지 다운로드
    return send_file(output, mimetype='image/png', as_attachment=True, download_name='transformed_image.png')


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9000)