import os
import numpy as np
import csv
import copy
from random import randint

class DataManager(object):
    """
    Class to manage data handling. Just CSV import for now.
    """
    def __init__(self, split):
        self.split = split       # train/valid/test fractions, should sum to 1
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.initialised = False

    def init_iris(self, filepath):
        self.filepath = filepath
        self.num_classes = 0

        self.split_data_paths = {'train': './data/iris/train.csv',
                                 'valid': './data/iris/valid.csv',
                                 'test' : './data/iris/test.csv'}
        if os.path.isfile(self.split_data_paths['train']):
            print('Train/Valid/Test data found, loading...')
            self.train_raw, self.valid_raw, self.test_raw = self._load_data()
        else:
            print('Preparing Train/Valid/Test data from {}'.format(filepath))
            self.train_raw, self.valid_raw, self.test_raw = self._import_and_write_data(filepath)

        self.train, self.valid, self.test = self._iris_make_1hot()
        self.initialised = True
        print('Dataset prepared')

        return True

    def prepare_train(self):
        self._initialise_check()
        return self._prepare_split(self.train)

    def prepare_valid(self):
        self._initialise_check()
        return self._prepare_split(self.valid)

    def prepare_test(self):
        self._initialise_check()
        return self._prepare_split(self.valid)

    def _initialise_check(self):
        if not self.initialised:
            print('Init a dataset first (e.g. data_manager.init_iris(filepath))')
            quit()
        return self.initialised

    def _prepare_split(self, split):
        """
        Separate data split into X and Y and convert to numpy array.
        """
        X, Y = zip(*split)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y


    def _import_and_write_data(self, filepath):
        self.iris_raw = self._iris_import(filepath)
        train_raw, valid_raw, test_raw = self._split_data(self.iris_raw)
        self._write_split_files(train_raw, valid_raw, test_raw, 'iris')
        return train_raw, valid_raw, test_raw

    def _load_data(self):
        self.num_classes = 3
        train_raw = self._load_split(self.split_data_paths['train'])
        valid_raw = self._load_split(self.split_data_paths['valid'])
        test_raw  = self._load_split(self.split_data_paths['test'])
        return train_raw, valid_raw, test_raw

    def _load_split(self, split_path):
        split_raw = []
        with open(split_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                iris_properties = row[:4]
                iris_properties = [float(prop) for prop in iris_properties]
                iris_label = row[4]
                label_idx = int(row[-1])
                split_raw.append([iris_properties, iris_label, label_idx])
        return split_raw

    def _iris_import(self, filepath):
        """
        Import from raw Iris data file.
        Returns list of [[iris_properties], [iris_label], [label_index]]
        """
        iris_raw = []
        with open(filepath) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                iris_properties = row[:4]
                iris_properties = [float(prop) for prop in iris_properties]
                iris_label = row[-1]
                label_idx = self._get_idx(iris_label)
                iris_raw.append([iris_properties, iris_label, label_idx])

        return iris_raw

    def _iris_make_1hot(self):
        train_1hot = self._iris_split_to_1hot(self.train_raw)
        valid_1hot = self._iris_split_to_1hot(self.valid_raw)
        test_1hot  = self._iris_split_to_1hot(self.test_raw)
        return train_1hot, valid_1hot, test_1hot

    def _iris_split_to_1hot(self, split):
        iris_1hot = []
        for properties, label, label_idx in split:
            one_hot = [0 for _ in range(self.num_classes)]
            one_hot[label_idx] = 1
            iris_1hot.append([properties, one_hot])
        return iris_1hot

    def _get_idx(self, label):
        if label in self.label_to_idx:
            return self.label_to_idx[label]
        else:
            new_idx = self.num_classes
            self.label_to_idx[label]   = new_idx
            self.idx_to_label[new_idx] = label
            self.num_classes += 1
            return new_idx

    def _split_data(self, original_data):
        train, valid, test = [], [], []
        n_data = len(original_data)
        n_train, n_valid, n_test = (int(n_data*fraction) for fraction in self.split)
        n_valid = n_data - n_train - n_test
        count = {'train': 0, 'valid': 0, 'test': 0}

        # split by class to fill train/valid/test with even class distribution
        data_by_class = self._split_into_classes(original_data)

        while True:
            # fill train/valid/test with a pass through classes
            for data in data_by_class:
                if not data:
                    continue
                pop_idx = randint(0, len(data)-1)
                ex_input, ex_label, ex_idx = data.pop(pop_idx)
                if count['train'] < n_train:
                    train.append([ex_input, ex_label, ex_idx])
                    count['train'] += 1
                elif count['valid'] < n_valid:
                    valid.append([ex_input, ex_label, ex_idx])
                    count['valid'] += 1
                else:
                    test.append([ex_input, ex_label, ex_idx])
                    count['test'] += 1

            # keep taking passes until data_by_class is empty
            remaining = [len(data) for data in data_by_class]
            if sum(remaining) == 0:
                break
        return train, valid, test

    def _split_into_classes(self, original_data):
        data_by_class = [[] for _ in range(self.num_classes)]
        for ex_input, ex_label, ex_idx in original_data:
            data_by_class[ex_idx].append([ex_input, ex_label, ex_idx])
        return copy.deepcopy(data_by_class)

    def _write_split_files(self, train, valid, test, dataset='iris'):
        path = os.path.join('data/', dataset)
        self._write_file(train, 'train', path)
        self._write_file(valid, 'valid', path)
        self._write_file(test, 'test', path)

    def _write_file(self, split, splitname, path):
        filename = os.path.join(path, splitname+'.csv')
        data_strs = self._raw_data_to_str(split)
        with open(filename,'w') as f:
            for row in data_strs:
                writeline = ','.join(row)
                f.write(writeline)
                f.write('\n')

    def _raw_data_to_str(self, data):
        """
        Function to convert data format into a string before writing to csv.
        Dependent on the dataset, so should be adjusted for each dataset.
        """
        newlist = []
        for row in data:
            newrow = []
            properties, label, label_idx = row
            newrow.extend(properties)
            newrow.extend([label, label_idx])
            newrow = [str(item) for item in newrow]
            newlist.append(newrow)
        return newlist
