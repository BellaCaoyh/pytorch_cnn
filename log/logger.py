import logging
from logging import handlers

# logging.basicConfig(filename='../cache/log/example.log',level=logging.DEBUG)
# logging.debug('debug massage')
# logging.info('info message')
# logging.warning('warning')
# logging.error('error')
# logging.critical('critical')

# logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.DEBUG)
# logging.debug('This message should appear on the console')
# logging.info('So should do it')
# logging.warning('And this, too')
#
logger_file = '../cache/log/example.log'
# logger = logging.getLogger(logger_file)
# logger.setLevel(logging.DEBUG)
#
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
#
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#
# ch.setFormatter(formatter)
#
# logger.addHandler(ch)
#
# logger.debug('debug')

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info' : logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,filename, level='info',
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s : %(message)s'):
        #create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level)) #Logger_level 总开关？？？
        format_str = logging.Formatter(fmt) #set log format

        # create a handler to input
        ch = logging.StreamHandler()
        ch.setLevel(self.level_relations.get(level))
        ch.setFormatter(format_str)

        #create a handler to filer
        fh = logging.FileHandler(filename=filename, mode='a')
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

def main():
    log = Logger(logger_file, level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    log.logger.critical('************************************************')

if __name__=='__main__':
    main()





















