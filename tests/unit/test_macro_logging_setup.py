import logging


def test_macro_logging_uses_debug_format_when_requested():
    from mobilerun.macro.utils import configure_macro_logging

    logger = configure_macro_logging(debug=True)
    assert logger.level == logging.DEBUG
    assert (
        logger.handlers[0].formatter.format(
            logging.LogRecord("mobilerun-macro", logging.INFO, "", 0, "hello", (), None)
        )
        == "INFO mobilerun-macro hello"
    )
