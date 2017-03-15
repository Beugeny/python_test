from utils.scoring import rmlse_scoring


def predict_model(clf, x_test, y_test):
    result = rmlse_scoring(clf, x_test, y_test)
    print("Test score={0}".format(result))
