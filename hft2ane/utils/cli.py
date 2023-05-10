from argparse import ArgumentParser, Namespace
from contextlib import contextmanager

from beaupy.spinners import Spinner, DOTS


@contextmanager
def spinner(msg, style=DOTS):
    spinner = Spinner(style, msg)
    spinner.start()
    yield
    spinner.stop()


def argument(*name_or_flags, **kwargs):
    """
    Convenience function to properly format arguments to pass to the
    subcommand decorator.
    """
    return (name_or_flags, kwargs)


def group(*args):
    """
    Convenience function to properly format conceptual group of arguments to
    pass to the subcommand decorator.
    """
    return args


def mutex_group(*args):
    """
    Convenience function to properly format mutually exclusive arguments to
    pass to the subcommand decorator.
    """
    return MutexGroup(args)


class MutexGroup(list):
    pass


class Cli:
    """
    Adapted from:
    https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f
    """

    def __init__(self, parser: ArgumentParser | None = None):
        self._parser = parser or ArgumentParser()
        self.parent = self._parser.add_subparsers(title="commands", dest="subcommand")

    def parse_args(self, args=None, namespace=None) -> Namespace:
        return self._parser.parse_args(args, namespace)

    def subcommand(self, *args):
        """
        Decorator to define a new subcommand in a sanity-preserving way.
        The function will be stored in the ``func`` variable when the parser
        parses arguments so that it can be called directly like so::
            args = cli.parse_args()
            args.func(args)
        Usage example::
            @subcommand(
                argument("-d", help="Enable debug mode", action="store_true"),
                mutex_group(
                    argument("-a", help="Option A", action="store_true"),
                    argument("-b", help="Option B", action="store_true"),
                ),
            )
            def subcommand(args):
                print(args)
        Then on the command line::
            $ python cli.py subcommand -d
        """

        def decorator(func):
            parser = self.parent.add_parser(func.__name__, description=func.__doc__)
            for argspec in args:
                if isinstance(argspec, (list, tuple)):
                    if isinstance(argspec, MutexGroup):
                        group = parser.add_mutually_exclusive_group()
                    else:
                        group = parser.add_argument_group()
                    for argspec_ in args:
                        group.add_argument(*argspec_[0], **argspec_[1])
                else:
                    parser.add_argument(*argspec[0], **argspec[1])
            parser.set_defaults(func=func)

        return decorator


def is_format_string(s):
    try:
        s.format()
    except (ValueError, KeyError, IndexError):
        return True
    return False
