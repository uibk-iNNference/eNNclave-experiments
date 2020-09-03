import pandas as pd
import os
import json


def reduce(input_file, output_file, samples_per_class, data_dir='datasets/amazon'):
    sample_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def _check_complete():
        for v in sample_counts.values():
            if v < samples_per_class:
                return False
        return True

    ratings = []
    texts = []

    with open(os.path.join(data_dir, input_file), 'r') as input_file:
        for line in input_file:
            json_dict = json.loads(line)
            try:
                text = json_dict['reviewText']
                rating = int(json_dict['overall'])

                if sample_counts[rating] >= samples_per_class:
                    continue

                ratings.append(rating)
                texts.append(text)
                sample_counts[rating] += 1
            except KeyError:
                continue

            if _check_complete():
                break

        data = pd.DataFrame()
        data['rating'] = pd.Series(ratings)
        data['text'] = pd.Series(texts, index=data.index)
        data.to_pickle(os.path.join(data_dir, output_file))


def main():
    print("Reducing books")
    reduce('Books.json', 'books.pkl', samples_per_class=150000)
    print("Reducing CDs")
    reduce('CDs_and_Vinyl.json', 'cds.pkl', samples_per_class=200)


if __name__ == '__main__':
    main()
