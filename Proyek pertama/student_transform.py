
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

    # One hot encode Ethnicity
    ethnicity_one_hot = tf.one_hot(tf.squeeze(inputs["Ethnicity"], axis=1), depth=4)
    outputs['Ethnicity_caucasian'] = ethnicity_one_hot[:, 0]
    outputs['Ethnicity_african_american'] = ethnicity_one_hot[:, 1]
    outputs['Ethnicity_asian'] = ethnicity_one_hot[:, 2]
    outputs['Ethnicity_other'] = ethnicity_one_hot[:, 3]

    # One hot encode Parental Education
    parent_edu_one_hot = tf.one_hot(tf.squeeze(inputs["ParentalEducation"], axis=1), depth=5)
    outputs['ParentalEducation_None'] = parent_edu_one_hot[:, 0]
    outputs['ParentalEducation_HighSchool'] = parent_edu_one_hot[:, 1]
    outputs['ParentalEducation_SomeCollege'] = parent_edu_one_hot[:, 2]
    outputs['ParentalEducation_Bachelor'] = parent_edu_one_hot[:, 3]
    outputs['ParentalEducation_Higher'] = parent_edu_one_hot[:, 4]

    # One hot encode Parental Support
    parent_support_one_hot = tf.one_hot(tf.squeeze(inputs["ParentalSupport"], axis=1), depth=5)
    outputs['ParentalSupport_None'] = parent_support_one_hot[:, 0]
    outputs['ParentalSupport_Low'] = parent_support_one_hot[:, 1]
    outputs['ParentalSupport_Moderate'] = parent_support_one_hot[:, 2]
    outputs['ParentalSupport_High'] = parent_support_one_hot[:, 3]
    outputs['ParentalSupport_VeryHigh'] = parent_support_one_hot[:, 4]

    # Standardize features
    outputs[transformed_key("Absences")] = tft.scale_to_z_score(inputs["Absences"])
    outputs[transformed_key("Age")] = tft.scale_to_z_score(inputs["Age"])
    outputs[transformed_key("GPA")] = tft.scale_to_z_score(inputs["GPA"])
    outputs[transformed_key("StudyTimeWeekly")] = tft.scale_to_z_score(inputs["StudyTimeWeekly"])
    outputs[transformed_key("GradeClass")] = tf.cast(inputs["GradeClass"], tf.int64) #Target Feature
    

    # normal features
    outputs["Extracurricular"] = inputs["Extracurricular"]
    outputs["Gender"] = inputs["Gender"]
    outputs["Music"] = inputs["Music"]
    outputs["Sports"] = inputs["Sports"]
    outputs["Tutoring"] = inputs["Tutoring"]
    outputs["Volunteering"] = inputs["Volunteering"]

    return outputs
