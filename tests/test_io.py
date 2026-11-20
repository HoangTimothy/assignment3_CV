import numpy as np
import pytest
from PIL import Image

from cvdoc.io import read_image, write_image


@pytest.mark.parametrize("extension", ["png", "jpeg"])
def test_can_read_png_jpeg(tmp_path, extension):
    image = np.full((30, 40, 3), (20, 100, 240), np.uint8)
    path = tmp_path / f"color.{extension}"
    write_image(path, image)
    restored = read_image(path)
    assert restored.shape == image.shape
    assert np.abs(restored.astype(float) - image).max() <= 2


def test_orientation_transform(tmp_path):
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:10, :10] = [255, 0, 0]
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif, quality=100, subsampling=0)
    result = read_image(path)
    assert result.shape == (40, 20, 3)
    # EXIF 6 rotates clockwise: original top-left marker moves to top-right.
    assert result[4, 15, 2] > 240 and result[4, 15, 0] < 10
    assert result[35, 4].max() < 10


def test_corrupt_image(tmp_path):
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="image_read_error"):
        read_image(path)
