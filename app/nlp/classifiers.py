from transformers import pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)

zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

def zero_shot_classification(text, labels, multi_label=False):
    """
    Zero shot classification models are able to provide confidence values for a given text belonging to a given label.
    This function implements the generic call to the model, and can be used to classify any text with any labels.

    :param text: The text to be classified
    :param labels: The labels to classify the text with
    :param multi_label: Whether the labels are multi-label or not
    """
    logging.info("Zero-shot classification on text: " + str(text) + " with labels: " + str(labels))
    start_time = time.time()
    output = zero_shot_classifier(text, labels, multi_label=multi_label)
    stop_time = time.time()
    logging.info("Zero-shot classification took " + str(stop_time - start_time) + " seconds")
    logging.info("Zero-shot classification: " + str(output))
    return output

def is_recommendation_request(text, confidence_threshold=0.9):
    """
    This function sends a text to the zero-shot classifier and returns True if the text is a recommendation request
    and False if it is not.

    :param text: The text to be classified
    :return: True if the text is a recommendation request (> confidence thresholde to maintain high precision) and False if it is not
    """
    labels = ["location_recommendations_request", "no_location_recommendations_request"]
    output = zero_shot_classification(text, labels)
    if output['labels'][0] == "recommendations_request" and output['scores'][0] > confidence_threshold:
        return True
    else:
        return False