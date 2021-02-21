import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--pickle', type=str,
                        default='data/training_samples.pickle')
    args = parser.parse_args()
    with open(args.pickle, 'rb') as f:
        content = pickle.load(f)

    n_removed = 0
    count = 0
    new_dict = dict()
    for c in content:
        if content[c]['dir'] == args.name:
            n_removed += 1
        else:
            new_dict[count] = content[c]
            count += 1

    print('removed {} items'.format(n_removed))
    print('new index generated with {} items'.format(len(new_dict.keys())))
    out_filename = 'samples.pickle'
    if args.force:
        out_filename = args.pickle
    with open(out_filename, 'wb') as f:
        pickle.dump(new_dict, f)
