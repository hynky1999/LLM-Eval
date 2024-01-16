import logging

logger = logging.getLogger(__name__)


def maybe(fc):
    def _inner(x):
        try:
            return fc(x)
        except Exception as e:
            logger.error(e)
            return None

    return _inner
