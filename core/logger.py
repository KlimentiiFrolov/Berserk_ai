import logging

from .config import LoggerSettings, get_settings


def prepare_logger(name: str, log_config: LoggerSettings):
    log = logging.getLogger(name)
    log.setLevel(level=log_config.level)

    if not log.handlers: # защита от дублирования handler'а при повторном вызове
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt=log_config.format,
                datefmt=log_config.datefmt,
            )
        )
        log.addHandler(handler)

    log.propagate = False # Не прокидывать сообщения по цепочке выше (в root логгер, который настраивается определением общим logging.basicConfig)

    return log

log = prepare_logger(__name__, get_settings().log)