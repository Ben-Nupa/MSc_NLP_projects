import glob
import re


def display_data(data, number_of_lines=5):
    """Display the dataset line by line to check the read_dataset function."""
    i = 0
    while i < number_of_lines:
        print(data[i])
        i += 1


def read_dataset(
        path_to_file='1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100',
        number_lines=10 ** 6):
    """
    Read the given file.

    Parameters
    ----------
    path_to_file : str
        Relative path.
    number_lines : int
        Lines to be extracted (10**6 by default)

    Returns
    ----------
    out : List[List[str]]
        List of sentences where sentences are list of words.
    """

    # Init regex to remove non alphanumerical characters, still we keep spaces and '
    regex = re.compile("[^a-z ']+")

    data = []
    with open(path_to_file, encoding='utf-8') as current_file:
        # print("Opening file n°" + str(id_file) + ". Wrote " + str(len(data)) + "/" + str(number_lines))

        # Init line
        line = current_file.readline()

        while line and len(data) < number_lines:
            # While we are not in the end of the file and we haven't filled the data list

            # Removing unwanted characters
            processed_line = regex.sub('', line.lower())
            final_line = processed_line.replace("  ", " ").split(" ")
            final_line.pop()
            data.append(final_line)

            # refresh line value
            line = current_file.readline()
    # print("Finished reading with " + str(len(data)) + " lines.")
    return data
