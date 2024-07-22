#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os


class Path(object):
    @staticmethod
    def db_root_dir(database=''):
        db_root = r'/cs/student/msc/dsml/2023/mdavudov/ADL/MaskContrast/pretrain/DATASET' # VOC will be automatically downloaded
        db_names = ['VOCSegmentation', 'CATSNDOGS', 'CATSNDOGS_PICANET', 'IMAGENET_SAMPLE', 'IMAGENET_SAMPLE_PICANET']

        if database == '':
            return db_root

        if database in db_names:
            return os.path.join(db_root, database)

        else:
            raise ValueError('Invalid database {}'.format(database))
