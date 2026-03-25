import numpy as np
from PIL import Image

from src.face_recog import detect_faces, encode_faces, load_known_faces


def test_no_face_blank_image():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    locs = detect_faces(blank)
    assert isinstance(locs, list)
    assert len(locs) == 0


def test_load_known_faces_returns_lists():
    encodings, names = load_known_faces('nonexistent_dir')
    assert isinstance(encodings, list)
    assert isinstance(names, list)
    assert len(encodings) == 0
    assert len(names) == 0
