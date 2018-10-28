"""
    Class that pre-processes
    the raw data in a compa-
    tible way so that it can
    be feeded to the model -
    chosen for the task.

    The process, depending on
    the data of input data that
    are required can provide
    either "word embeddings "or
    "vectorized" transformations.

    author: Ilyass Taouil <itaouil95@gmail.com>
"""

# Useful import
import os
from keras.preprocessing.text import Tokenizer

class DataPreparation:

    def __init__(self):
        # Relative path to
        # folder containing
        # the data
        self.base_dir = "../data"

    # Loads the books to be processed
    # and stores them in a list
    def load_data(self, language, num_of_books=6):
        """
            Loads the books to be processed
            and returns them as a list of strings.

            Input:
                string: Language whose books to load
                number: Number of books to process

            Ouput:
                list: List of strings of the loaded books
        """
        # Output
        books = []

        # Working path
        wrk_path = "{}/{}".format(self.base_dir, language)

        # Books' filenames
        book_files = os.listdir(wrk_path)

        # Read each book and append
        for book_file in book_files:
            input_file = "{}/{}".format(wrk_path, book_file)
            with open(input_file) as f:
                book = f.read()
                books.append(book)

        # Useful logs about books
        print("Loaded {} books for {}.".format(num_of_books, language))
        for i in range(len(books)):
            print("There are {} words in {}.".format(len(books[i].split()), book_files[i]))

        return books

    def vectorize_data(self, data):
        """
            The functions processes the
            data by cleaning them from
            special characters and other
            punctuations. Afterwards the
            keras tokenizer class is used
            to vectorize the data to a tensor
            to be feeded to out model later on.

            Input:
                list: List of data (i.e. books)

            Output:
                tensor: Float32 numpy array (one hot encoded)
        """
        # Tokenizer
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(data)

        sequences = tokenizer.texts_to_sequences(data)
        print("Seq: ", sequences)

        one_hot_results = tokenizer.texts_to_matrix(data, mode='binary')
        print("One hot: ", one_hot_results)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        return one_hot_results

    

d = DataPreparation()
books = d.load_data("english")
vec_data = d.vectorize_data(books)
