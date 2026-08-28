from meddetect.synthetic import generate_labeled_images, render_negative, render_positive


def test_synthetic_images_are_rgb() -> None:
    negative = render_negative(size=64, seed=1)
    positive = render_positive(size=64, seed=1)
    assert negative.mode == "RGB"
    assert positive.mode == "RGB"
    assert negative.size == (64, 64)
    assert positive.size == (64, 64)


def test_labeled_pairs_cover_both_classes() -> None:
    pairs = generate_labeled_images(n_per_class=3, size=48, seed=0)
    labels = sorted(label for _, label in pairs)
    assert labels == [0, 0, 0, 1, 1, 1]


def test_positive_is_darker_on_average() -> None:
    import numpy as np

    negative = np.asarray(render_negative(size=128, seed=7), dtype=np.float32)
    positive = np.asarray(render_positive(size=128, seed=7), dtype=np.float32)
    assert positive.mean() < negative.mean()
