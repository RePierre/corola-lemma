from argparse import ArgumentParser
import pandas as pd


def levenshtein_distance(source, target):
    """Computes the Levenshtein distance between source and target.

    Taken from version 6 of the algorithms presented at
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    Parameters
    ----------
    source: string
        The source word.
    target: string
        The target word.

    Returns
    -------
    dist: integer
        The number of edits that need to be made to get from source to target.
    """
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    v0 = [None] * (len(target) + 1)
    v1 = [None] * (len(target) + 1)

    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(source)):
        v1[0] = i + 1
        for j in range(len(target)):
            cost = 0 if source[i] == target[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(target)]


def interpret_results(args):
    df = pd.read_csv(args.results_file,
                     sep=args.separator,
                     encoding=args.encoding,
                     header=0)
    df['Levenshtein'] = df.apply(
        lambda row: levenshtein_distance(row.Prediction, row.Lemma), axis=1)
    print(df)
    dist_series = df['Levenshtein']
    max_val = dist_series.max()
    print("Maximum Levenshtein distance: {}".format(max_val))
    print("Average Levenshtein distance: {}".format(dist_series.mean()))
    print("Median Levenshtein distance: {}".format(dist_series.median()))
    print("Distance groups")
    print(df.groupby(['Levenshtein']).count())
    ax = dist_series.plot.hist(bins=max_val)
    ax.set_xlabel("Levenshtein distance from prediction to lemma")
    ax.figure.savefig(args.output_file)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--results-file',
                        help="Path of the results file in CSV format.",
                        default="predictions.csv")
    parser.add_argument('--separator',
                        help="The separator character of the CSV file.",
                        default=",")
    parser.add_argument('--encoding',
                        help="The encoding of the CSV file.",
                        default="utf-8")
    parser.add_argument('--output-file',
                        help="Path of the output file.",
                        default="results-hist.png")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    interpret_results(args)
