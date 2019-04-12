import numpy as np


def extract_dataset_as_text(path: str, is_training_set: bool, nb_dialogues=-1) -> tuple:
    """
    Extract the dataset as text lists. If it is a training dataset, the first answer will be the correct one, others are
    distractors.
    To access an element of a dialogue i in one of the following list (e.g: output_list), do: output_list[i].

    Parameters
    ----------
    path : str
        Path to training file.
    is_training_set : bool
        Whether the dataset to extract is a training set (correct answer is known).
    nb_dialogues : int
        Number of dialogues to extract. Set -1 for all.

    Returns
    -------
    out : Tuple[ List[List[str]], List[List[str]], List[List[int]], List[List[str]], List[List[List[str]]] ]
        my_personae : List[List[str]]
            My personae of each dialogue.
        other_personae : List[List[str]]
            Other personae of each dialogue.
        line_indices : List[List[int]]
            Indices of lines, except those describing the persona, of each dialogue.
        utterances : List[List[str]]
            Utterances (question-like) of each dialogue.
        answers : List[List[List[str]]]
            Answers of each utterance of each dialogue. The correct one (for a training set) for a dialogue i and an
            utterance j is answers[i][j][0], the others answers[i][j][k] for k>0 are wrong answers.
    """
    my_personae = []
    other_personae = []
    line_indices = []
    utterances = []
    answers = []
    idx_dialogue = 0

    with open(path, 'r') as file:
        for line in file:
            words = line.split()
            idx_line = int(words[0])
            if idx_line == 1:
                idx_dialogue += 1
                if idx_dialogue == nb_dialogues + 1:
                    break

            # Get my persona
            if words[1] + ' ' + words[2] == 'your persona:':
                if len(my_personae) != idx_dialogue:
                    my_personae.append([])
                my_personae[-1].append(' '.join(str(word) for word in words[3:]))

            # Get other persona
            elif words[1] + ' ' + words[2] == "partner's persona:":
                if len(other_personae) != idx_dialogue:
                    other_personae.append([])
                other_personae[-1].append(' '.join(str(word) for word in words[3:]))

            # Get dialogue
            else:
                if len(utterances) != idx_dialogue:
                    line_indices.append([])
                    utterances.append([])
                    answers.append([])

                line_indices[-1].append(idx_line)
                exchange = line[len(str(idx_line)) + 1:].split('\t')
                utterances[-1].append(exchange[0])
                # Training set: answer is known
                if is_training_set:
                    answers[-1].append([exchange[1]])  # Correct answers
                    for statement in exchange[2:]:  # Wrong answers
                        if statement == '':
                            continue
                        distractors = statement.split('|')
                        answers[-1][-1] += distractors
                # Testing set: answer is unknown
                else:
                    answers[-1].append([])
                    for statement in exchange[1:]:
                        if statement == '':
                            continue
                        distractors = statement.split('|')
                        answers[-1][-1] += distractors
    print('Loaded data:')
    print(np.shape(my_personae))
    print(np.shape(other_personae))
    print(np.shape(line_indices))
    print(np.shape(utterances))
    print(np.shape(answers))
    return my_personae, other_personae, line_indices, utterances, answers


def print_dialogue(my_personae: list, other_personae: list, line_indices: list, utterances: list, answers: list,
                   idx_dialogue=-1):
    """
    Prints a complete dialogue extracted from a training set. See function 'extract_dataset_as_text' for details of the
    parameters.
    """

    def print_precise_dialogue(idx_dialogue):
        print('\n----------------------')
        print('My persona:')
        for characteristic in my_personae[idx_dialogue]:
            print(characteristic)
        print('############')
        print('Other persona:')
        for characteristic in other_personae[idx_dialogue]:
            print(characteristic)
        print('############')
        for j in range(len(utterances[idx_dialogue])):
            print(str(line_indices[idx_dialogue][j]) + '-U: ' + utterances[idx_dialogue][j])
            print('A:', answers[idx_dialogue][j][0])

    if 0 <= idx_dialogue < len(line_indices):
        print_precise_dialogue(idx_dialogue)
    else:
        for i in range(len(line_indices)):
            print_precise_dialogue(i)


NB_DIALOGUES = -1

my_personae, other_personae, line_indices, utterances, answers = extract_dataset_as_text(
    'data/train_both_original.txt', True, NB_DIALOGUES)

print_dialogue(my_personae, other_personae, line_indices, utterances, answers, 53)
