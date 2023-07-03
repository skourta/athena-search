import pickle


def load_dataset(path_to_dataset: str):
    with open(path_to_dataset, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def save_dataset(dataset, path_to_save: str):
    with open(path_to_save, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
