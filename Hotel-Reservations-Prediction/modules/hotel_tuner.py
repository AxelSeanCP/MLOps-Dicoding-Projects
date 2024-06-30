import keras_tuner
from keras_tuner.engine import base_tuner
from keras_tuner import RandomSearch
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any

from hotel_trainer import (
    get_model,
    input_fn
)

def get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building model"""
    hp = keras_tuner.HyperParameters()
    hp.Choice('units', [16, 64, 256], default=64)
    hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-3)
    hp.Choice('num_layers', [1, 2, 3], default=2)

    return hp

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args):
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
        get_model,
        objective='val_binary_accuracy',
        max_trials=10,
        hyperparameters=get_hyperparameters(),
        directory=fn_args.working_dir,
        project_name='hotel-reservations-prediction'
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': eval_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )