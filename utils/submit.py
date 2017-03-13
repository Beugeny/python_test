import pandas as pd


def submit(ids, res, f, id_name, result_name):
    submission = pd.DataFrame({
        id_name: ids,
        result_name: res
    })
    submission.to_csv(f + 'submission.csv', index=False)
