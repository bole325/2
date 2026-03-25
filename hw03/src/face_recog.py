import io
import os
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np
from PIL import Image, ImageDraw


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    return image


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """返回人脸位置：top, right, bottom, left"""
    return face_recognition.face_locations(image, model='hog')


def encode_faces(image: np.ndarray, locations: List[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
    if locations is None:
        locations = detect_faces(image)
    return face_recognition.face_encodings(image, known_face_locations=locations)


def match_faces(known_encodings: List[np.ndarray], known_names: List[str], target_encodings: List[np.ndarray], tolerance=0.5):
    """返回每张目标人脸最可能名字（或 Unknown）"""
    results = []
    for enc in target_encodings:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, enc) if len(known_encodings) > 0 else []
        name = 'Unknown'
        confidence = None
        if True in matches:
            best_index = np.argmin(distances)
            name = known_names[best_index]
            confidence = float(1 - distances[best_index])
        results.append({'name': name, 'confidence': confidence})
    return results


def annotate_image(image_pil: Image.Image, face_locations: List[Tuple[int, int, int, int]], labels: List[str]):
    draw = ImageDraw.Draw(image_pil)
    for (top, right, bottom, left), label in zip(face_locations, labels):
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=4)
        text_height = 15
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 255, 0, 128))
        draw.text((left + 5, bottom - text_height - 5), label, fill='black')
    return image_pil


def load_known_faces(folder: str = None):
    known_encodings = []
    known_names = []
    if folder is None:
        folder = Path(__file__).resolve().parents[1] / 'known_faces'
    else:
        folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        return known_encodings, known_names

    for path in sorted(folder.glob('*.*')):
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            image = face_recognition.load_image_file(str(path))
            locations = detect_faces(image)
            encodings = encode_faces(image, locations)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(path.stem)
        except Exception:
            continue

    return known_encodings, known_names
