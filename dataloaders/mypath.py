class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'cihp':
            return './data/datasets/CIHP_OS/'  # folder that contains cihp/.
        elif database == 'lip':
            return './data/datasets/LIP_OS/'  # folder that contains lip/.
        elif database == 'atr':
            return './data/datasets/ATR_OS/'  # folder that contains lip/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
