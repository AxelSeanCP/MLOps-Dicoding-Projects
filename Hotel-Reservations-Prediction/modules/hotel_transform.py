import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "type_of_meal_plan": 4,
    "room_type_reserved": 5,
    "market_segment_type": 5
}

NUMERICAL_FEATURES = {
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "required_car_parking_space",
    "lead_time",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests"
}

LABEL_KEY = "booking_status"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.
    
    Additional Info:
        - one hot encode categorical features
        - standardize numerical features
    """

    outputs = {}
    
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1, vocab_filename=key + '_vocab'
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )
    
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
