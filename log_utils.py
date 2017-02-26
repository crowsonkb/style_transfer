import logging
import os
import sys

try:
    import curses
except ImportError:
    curses = None


def _stderr_supports_color():
    color = False
    if curses and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
        try:
            curses.setupterm()
            if curses.tigetnum("colors") > 0:
                color = True
        except Exception:
            pass
    return color


class LogFormatter(logging.Formatter):
    """Log formatter originally from Tornado and modified."""
    DEFAULT_FORMAT = '%(color)s[%(levelname)1.1s %(asctime)s %(process)d]%(end_color)s %(message)s'
    DEFAULT_DATE_FORMAT = '%y%m%d %H:%M:%S'
    DEFAULT_COLORS = {
        logging.DEBUG: 4,    # Blue
        logging.INFO: 2,     # Green
        logging.WARNING: 3,  # Yellow
        logging.ERROR: 1,    # Red
    }

    def __init__(self, color=True, fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT,
                 colors=DEFAULT_COLORS, precision=3):
        r"""
        :arg bool color: Enables color support.
        :arg string fmt: Log message format.
          It will be applied to the attributes dict of log records. The
          text between ``%(color)s`` and ``%(end_color)s`` will be colored
          depending on the level if color support is on.
        :arg dict colors: color mappings from logging level to terminal color
          code
        :arg string datefmt: Datetime format.
          Used for formatting ``(asctime)`` placeholder in ``prefix_fmt``.
        .. versionchanged:: 3.2
           Added ``fmt`` and ``datefmt`` arguments.
        """
        super().__init__()
        self.default_time_format = datefmt
        self.precision = precision
        self.default_msec_format = ''
        self._fmt = fmt

        self._colors = {}
        if color and _stderr_supports_color():
            fg_color = (curses.tigetstr('setaf') or
                        curses.tigetstr('setf') or '')

            for levelno, code in colors.items():
                self._colors[levelno] = curses.tparm(fg_color, code).decode()
            self._normal = curses.tigetstr('sgr0').decode()
        else:
            self._normal = ''

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record)

        if record.levelno in self._colors:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        else:
            record.color = record.end_color = ''

        formatted = self._fmt % record.__dict__

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            lines = [formatted.rstrip()]
            lines.extend(ln for ln in record.exc_text.split('\n'))
            formatted = '\n'.join(lines)
        return formatted.replace('\n', '\n    ')

    def formatTime(self, record, datefmt=None):
        if not datefmt:
            datefmt = self.default_time_format
        fmttime = super().formatTime(record, datefmt)
        if self.precision >= 4:
            return '%s.%06d' % (fmttime, record.msecs*1000)
        if self.precision >= 1:
            return '%s.%03d' % (fmttime, record.msecs)
        return fmttime


def setup_logger(name=None, level=None, formatter_opts=None):
    """Sets up pretty logging using LogFormatter."""
    if formatter_opts is None:
        formatter_opts = {}
    logging.captureWarnings(True)
    logger = logging.getLogger(name)
    if 'ST_DEBUG' in os.environ:
        level = logging.DEBUG
    elif level is None:
        level = logging.INFO
    logger.setLevel(level)
    channel = logging.StreamHandler()
    formatter = LogFormatter(**formatter_opts)
    channel.setFormatter(formatter)
    logger.addHandler(channel)
    return logger
