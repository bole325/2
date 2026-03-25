# HW03: 人脸检测与识别（Streamlit）

## 1. 目录结构

```
hw03/
  requirements.txt
  README.md
  src/
    app.py
    face_recog.py
  tests/
    test_face_recog.py
  known_faces/
    Alice.jpg
    Bob.jpg
```

> `known_faces/` 是可选人脸库，若存在会被自动加载。

## 2. 环境准备

```bash
cd e:/python/three/hw03
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

> Windows: `\.venv\Scripts\activate`

## 3. 启动 Web 界面

```bash
streamlit run src/app.py
```

## 4. 功能

1. 上传图片或选择示例图片（从网络下载）
2. 调用 `face_recognition` 进行人脸检测、128维编码、与已知人脸对比
3. 在图像上框选人脸，展示识别结果与置信度

## 5. 测试

```bash
pytest -q
```


## 6. 已知人脸库说明

- 程序会自动加载 `known_faces/*.jpg`，文件名作为标签。可自行加入更多图片。

