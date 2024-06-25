
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import keras_tuner
from keras_tuner.engine import base_tuner
from keras_tuner import RandomSearch
import os
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import NamedTuple, Dict, Text, Any

LABEL_KEY = "GradeClass"

def transformed_key(key):
    """Rename transformed key"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64)->tf.data.Dataset:
    """Get post_transform feature and create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_key(LABEL_KEY)
    )

    return dataset
# BUG
def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # serialized tf example shape [none, 13]

        feature_spec = tf_transform_output.raw_feature_spec()
        print(len(feature_spec))

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get prediction using transformed_features
        return model(transformed_features)
    
    return serve_tf_examples_fn

def get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building model"""
    hp = keras_tuner.HyperParameters()
    hp.Choice('units', [16, 32, 64], default=32)
    hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-3)

    return hp

def model_builder(hparams):
    """Build machine learning model with hyperparameter tuning"""
    
    
    inputs = {
        transformed_key('Age'): tf.keras.layers.Input(shape=[1], name=transformed_key('Age')),
        transformed_key('Absences'): tf.keras.layers.Input(shape=[1], name=transformed_key('Absences')),
        transformed_key('GPA'): tf.keras.layers.Input(shape=[1], name=transformed_key('GPA')),
        transformed_key('StudyTimeWeekly'): tf.keras.layers.Input(shape=[1], name=transformed_key('StudyTimeWeekly')),
        'Ethnicity': tf.keras.layers.Input(shape=[1], name='Ethnicity'),
        'ParentalEducation': tf.keras.layers.Input(shape=[1], name='ParentalEducation'),
        'ParentalSupport': tf.keras.layers.Input(shape=[1], name='ParentalSupport'),
        'Extracurricular': tf.keras.layers.Input(shape=[1], name='Extracurricular'),
        'Gender': tf.keras.layers.Input(shape=[1], name='Gender'),
        'Music': tf.keras.layers.Input(shape=[1], name='Music'),
        'Sports': tf.keras.layers.Input(shape=[1], name='Sports'),
        'Tutoring': tf.keras.layers.Input(shape=[1], name='Tutoring'),
        'Volunteering': tf.keras.layers.Input(shape=[1], name='Volunteering')
    }

    # Concatenate all input features
    concat = tf.keras.layers.Concatenate()(list(inputs.values()))

    x = layers.Dense(hparams.get('units'), activation='relu')(concat)
    x = layers.Dense(hparams.get('units'), activation='relu')(x)
    x = layers.Dense(hparams.get('units'), activation='relu')(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Tuner component will run this function

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model , e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
    """

    # Define tuner
    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        hyperparameters=get_hyperparameters(),
        directory=fn_args.working_dir,
        project_name='student_performance_classification'
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set
        }
    )

# Trainer component will run this function
def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)

    # Build the model with tuned hyperparameters
    model = model_builder(hparams)

    # Train the model
    model.fit(
        x = train_set,
        validation_data = val_set,
        callbacks = [tensorboard_callback, es, mc],
        epochs=50,
        verbose=2
    )

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None, 13],
                dtype=tf.string,
                name='examples'
            )
        )
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
