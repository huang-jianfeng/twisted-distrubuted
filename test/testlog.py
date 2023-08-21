import logging
if __name__ == '__main__':
    logger = logging.getLogger('server-logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("this is info.")
    # logging.getLogger().setLevel(logging.INFO)
    # logging.info("dsfds ...")