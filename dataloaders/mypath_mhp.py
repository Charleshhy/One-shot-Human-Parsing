class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'mhp':
            return './data/datasets/LV-MHP-v2/'  # folder that contains atr/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
