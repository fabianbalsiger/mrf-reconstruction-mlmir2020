import pickle


def load(file_name: str):
    with open(file_name, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary


def save(file_name: str, dictionary: dict):
    with open(file_name, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
