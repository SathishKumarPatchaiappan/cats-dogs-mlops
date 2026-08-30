def classify_prediction(probability):
    if probability >= 0.5:
        return "Dog"
    return "Cat"


def test_dog_prediction():
    result = classify_prediction(0.8)

    assert result == "Dog"


def test_cat_prediction():
    result = classify_prediction(0.2)

    assert result == "Cat"