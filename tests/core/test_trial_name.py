from visionforge.core.trial_name import trial_name
from visionforge.utils.paths import validate_path_component


def test_learning_rate_is_shortened() -> None:
    assert trial_name("cifar10_resnet18_e20", {"training.learning_rate": 0.002}) == (
        "cifar10_resnet18_e20_lr0.002"
    )


def test_several_overrides_keep_their_order() -> None:
    name = trial_name("voc", {"training.lr0": 0.01, "training.batch_size": 16})
    assert name == "voc_lr0.01_bs16"


def test_model_name_contributes_only_its_value() -> None:
    assert trial_name("pet", {"model.name": "unet_resnet34"}) == "pet_unet_resnet34"


def test_unknown_key_uses_its_last_segment_and_unsafe_chars_are_replaced() -> None:
    assert trial_name("x", {"training.scheduler.kind": "cosine/warm"}) == (
        "x_kindcosine-warm"
    )


def test_booleans_and_small_floats() -> None:
    assert trial_name("x", {"training.cos_lr": True}) == "x_cos_lron"
    assert trial_name("x", {"training.learning_rate": 1e-4}) == "x_lr0.0001"


def test_no_overrides_returns_the_base() -> None:
    assert trial_name("x", {}) == "x"


def test_names_stay_valid_folder_names() -> None:
    name = trial_name("x", {"training.scheduler.kind": "a/b.", "model.name": "c:d"})
    assert validate_path_component(name) == name == "x_kinda-b_c-d"
