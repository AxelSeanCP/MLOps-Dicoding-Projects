
import tensorflow as tf
import tensorflow_transform as tft

def transformed_key(key):
    """Rename transformed key"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features
    
    Return:
        outputs: map from feature keys to transformed features
    
    Description:
        - apply one hot encoded to categorical features
        - apply standardization to float features and int features that isn't binary
        - apply rename of transformed features except for one hot encoded features
    """

    outputs = {}

    # Standardize features
    outputs[transformed_key("Absences")] = tft.scale_to_z_score(inputs["Absences"])
    outputs[transformed_key("Age")] = tft.scale_to_z_score(inputs["Age"])
    outputs[transformed_key("GPA")] = tft.scale_to_z_score(inputs["GPA"])
    outputs[transformed_key("StudyTimeWeekly")] = tft.scale_to_z_score(inputs["StudyTimeWeekly"])
    outputs[transformed_key("GradeClass")] = tf.cast(inputs["GradeClass"], tf.int64) #Target Feature
    
    # normal features
    outputs["Ethnicity"] = inputs["Ethnicity"]
    outputs["ParentalEducation"] = inputs["ParentalEducation"]
    outputs["ParentalSupport"] = inputs["ParentalSupport"]
    outputs["Extracurricular"] = inputs["Extracurricular"]
    outputs["Gender"] = inputs["Gender"]
    outputs["Music"] = inputs["Music"]
    outputs["Sports"] = inputs["Sports"]
    outputs["Tutoring"] = inputs["Tutoring"]
    outputs["Volunteering"] = inputs["Volunteering"]

    return outputs
