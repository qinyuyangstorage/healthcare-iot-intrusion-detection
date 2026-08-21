import numpy as np

from iot_ids.features import PaperFeaturePipeline, augment_components
from iot_ids.synthetic import make_synthetic_iomt


def test_nonlinear_augmentation_matches_paper() -> None:
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    result = augment_components(values)
    np.testing.assert_allclose(result[0], [1.0, 2.0, 1.0, 4.0, np.sin(1.0), np.sin(2.0)])


def test_train_only_pipeline_produces_six_features() -> None:
    frame = make_synthetic_iomt(300, seed=7)
    train = frame.iloc[:200]
    test = frame.iloc[200:]
    pipeline = PaperFeaturePipeline()
    x_train, y_train = pipeline.fit_transform(train)
    x_test, y_test = pipeline.transform(test)
    assert x_train.shape == (200, 6)
    assert x_test.shape == (100, 6)
    assert y_train.shape == (200,)
    assert y_test.shape == (100,)
