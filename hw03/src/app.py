import io
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

from face_recog import annotate_image, detect_faces, encode_faces, load_known_faces, match_faces


SAMPLE_IMAGES = {
    '示例1': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg',
    '示例2': 'https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg',
}


def load_image_from_upload(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert('RGB')
        return image
    except Exception as exc:
        st.error('无法解析上传图片: %s' % exc)
        return None


def load_image_from_url(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert('RGB')


def main():
    st.title('hw03：face_recognition 人脸检测与识别演示')

    st.sidebar.header('输入（上传 / 示例）')
    mode = st.sidebar.radio('选择输入方式', ['上传本地图片', '示例图片'])

    image = None
    if mode == '上传本地图片':
        uploaded_file = st.sidebar.file_uploader('选择图片文件', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = load_image_from_upload(uploaded_file)
    else:
        sample_name = st.sidebar.selectbox('示例图片', list(SAMPLE_IMAGES.keys()))
        if st.sidebar.button('加载示例图片'):
            image = load_image_from_url(SAMPLE_IMAGES[sample_name])

    st.sidebar.markdown('---')
    tolerance = st.sidebar.slider('识别容差 (tolerance)', 0.3, 0.8, 0.5, 0.01)

    st.sidebar.markdown('已知人脸库（known_faces 文件夹，文件名为标签）')
    known_encodings, known_names = load_known_faces()
    st.sidebar.write('已加载：%d个' % len(known_names))

    if image is not None:
        st.subheader('原图')
        st.image(image, use_column_width=True)

        img_array = np.array(image)
        face_locations = detect_faces(img_array)
        face_encodings = encode_faces(img_array, face_locations)

        st.write('检测到人脸数量：%d' % len(face_locations))

        if len(face_encodings) > 0 and len(known_encodings) > 0:
            matches = match_faces(known_encodings, known_names, face_encodings, tolerance=tolerance)
            labels = [f"{m['name']} ({m['confidence']:.2f})" if m['confidence'] is not None else m['name'] for m in matches]
        else:
            labels = ['Unknown'] * len(face_locations)

        annotated = annotate_image(image.copy(), face_locations, labels)
        st.subheader('检测结果')
        st.image(annotated, use_column_width=True)

        if len(face_locations) > 0:
            st.table([
                {'序号': i + 1, '位置': str(face_locations[i]), '标签': labels[i]} for i in range(len(face_locations))
            ])

    else:
        st.info('请先上传图片或加载示例图片。')


if __name__ == '__main__':
    main()
