import pytest
import tensorflow as tf

from ag_news import AGNewsClassifier

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize("num_classes, expected_shape", [(4, (1, 4)), (10, (1, 10))])
def test_AGNewsClassifier_output_shape(num_classes, expected_shape):
    classifier = AGNewsClassifier(model_name=MODEL_NAME, num_classes=num_classes)
    input_data = tf.ones((1, 256), dtype=tf.float32)
    output = classifier(input_data)
    tf.debugging.assert_shapes([(output, expected_shape)])


@pytest.mark.parametrize("num_classes", [4, 10])
def test_AGNewsClassifier_non_additivity(num_classes):
    classifier = AGNewsClassifier(model_name=MODEL_NAME, num_classes=num_classes)
    a = tf.ones((1, 256), dtype=tf.int32)
    b = tf.ones((1, 256), dtype=tf.int32)

    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(classifier(a + b) - (classifier(a) + classifier(b)))),
        0.0,
        summarize=2,
        message="AGNewsClassifier is acting linearly",
    )


@pytest.mark.parametrize("num_classes", [4, 10])
def test_AGNewsClassifier_non_homogeneity(num_classes):
    classifier = AGNewsClassifier(model_name=MODEL_NAME, num_classes=num_classes)
    a = tf.ones((1, 256), dtype=tf.int32)
    b = 2 * a
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(2 * classifier(a) - classifier(b))),
        0.0,
        summarize=2,
        message="AGNewsClassifier is acting linearly",
    )
