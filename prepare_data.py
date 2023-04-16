import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data() -> pd.DataFrame:
    """
    prepare data from  dataset.txt
    return: pd.DataFrame
    """
    with open('data/raw/dataset.txt', 'r', encoding="utf-8") as file:
        comments = file.readlines()
    comment = [" ".join(txt.split(' ')[1:]) for txt in comments]
    toxic = [0 if txt.split(' ')[0] == '__label__NORMAL' else 1
             for txt in comments]
    return pd.DataFrame({'comment': comment, 'toxic': toxic})


def split_data() -> None:
    """
    split and save dataset
    return: None
    """
    ds1 = pd.read_csv('data/raw/labeled.csv')
    ds2 = prepare_data()
    ds = pd.concat([ds1, ds2])
    train, test = train_test_split(ds, test_size=0.2,
                                   stratify=ds['toxic'],
                                   random_state=42)
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)


if __name__ == "__main__":
    split_data()
