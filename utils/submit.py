import pandas as pd


def submit(ids, res, f):
    submission = pd.DataFrame({
        "PassengerId": ids,
        "Survived": res
    })
    submission.to_csv(f + 'submission.csv', index=False)