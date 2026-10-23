"""Oxford-IIIT Pet trimap conversion on synthetic local arrays (ADR-149)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.workflow_benchmark.trimap_labels import (
    IGNORE_INDEX,
    TRIMAP_TO_MASK,
    convert_trimap_file,
    trimap_to_mask,
)


def test_mapping_is_pet_foreground_background_and_ignored_border() -> None:
    assert TRIMAP_TO_MASK == {1: 1, 2: 0, 3: IGNORE_INDEX}
    assert IGNORE_INDEX == 255


def test_trimap_values_become_class_ids_and_ignore() -> None:
    trimap = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.uint8)

    mask = trimap_to_mask(trimap)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[1, 0, 255], [255, 0, 1]]
    assert trimap.tolist() == [[1, 2, 3], [3, 2, 1]]


@pytest.mark.parametrize("value", [0, 4, 255])
def test_unknown_trimap_value_is_refused(value: int) -> None:
    trimap = np.array([[1, value]], dtype=np.uint8)
    with pytest.raises(ValueError, match=str(value)):
        trimap_to_mask(trimap)


@pytest.mark.parametrize("mode", ["L", "P"])
def test_file_conversion_writes_single_channel_mask_once(
    tmp_path: Path, mode: str
) -> None:
    source = tmp_path / "pet.png"
    Image.fromarray(np.array([[1, 2], [3, 1]], dtype=np.uint8)).convert(mode).save(
        source
    )
    target = tmp_path / "masks" / "pet.png"
    target.parent.mkdir()

    convert_trimap_file(source, target)

    with Image.open(target) as written:
        assert written.mode == "L"
        assert np.array(written).tolist() == [[1, 0], [255, 1]]
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        convert_trimap_file(source, target)
    assert target.read_bytes() == before


def test_rgb_trimap_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "pet.png"
    Image.new("RGB", (2, 2), (1, 1, 1)).save(source)
    with pytest.raises(ValueError, match="RGB"):
        convert_trimap_file(source, tmp_path / "mask.png")
    assert not (tmp_path / "mask.png").exists()
