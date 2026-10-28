import numpy as np
import pytest
from sensor_msgs.msg import Image
from wall_inspection_perception.inspector_node import image_to_array


def make_image(pixels, padding=0):
    height, width, _ = pixels.shape
    rows = np.zeros((height, width * 3 + padding), dtype=np.uint8)
    rows[:, :width * 3] = pixels.reshape(height, width * 3)
    msg = Image()
    msg.height = height
    msg.width = width
    msg.encoding = 'rgb8'
    msg.step = width * 3 + padding
    msg.data = rows.tobytes()
    return msg


def test_converts_rgb8_image():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    assert np.array_equal(image_to_array(make_image(pixels)), pixels)


def test_ignores_row_padding():
    # 行の末尾の詰め物は画素ではないので取り除かれる
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    assert np.array_equal(image_to_array(make_image(pixels, padding=4)), pixels)


def test_rejects_other_encodings():
    msg = make_image(np.zeros((2, 3, 3), dtype=np.uint8))
    msg.encoding = 'bgr8'

    with pytest.raises(ValueError):
        image_to_array(msg)
