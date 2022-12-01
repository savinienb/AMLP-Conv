
from datetime import datetime


class Timer(object):
    """
    Timer class.
    """
    def __init__(self, name, print_on_start=True, print_on_stop=True):
        """
        Initializer.
        :param name: The name of the timer.
        :param print_on_start: If True, print when starting the timer.
        :param print_on_stop: If True, print when stopping the timer.
        """
        self.name = name
        self.print_on_start = print_on_start
        self.print_on_stop = print_on_stop
        self.start_time = None
        self.stop_time = None

    def __enter__(self):
        """
        Start the timer when entering a scope.
        """
        self.start()

    def __exit__(self, *args, **kwargs):
        """
        Stop the timer when exiting a scope.
        """
        self.stop()

    def start(self):
        """
        Start the timer.
        """
        self.start_time = datetime.now()
        if self.print_on_start:
            self.print_start()

    def stop(self):
        """
        Stop the timer.
        """
        self.stop_time = datetime.now()
        if self.print_on_stop:
            self.print_stop()

    def reset(self):
        """
        Stop and then start the timer.
        """
        self.stop()
        self.start()

    def elapsed_time(self):
        """
        Return the elapsed time between start and stop time.
        :return: The elapsed time as a datetime object.
        """
        return self.stop_time - self.start_time

    def time_string(self, time):
        """
        Return a string for the given time.
        :param time: The time.
        :return: The time string.
        """
        return '%02d:%02d:%02d' % (time.hour, time.minute, time.second)

    def seconds_string(self, duration):
        """
        Return a seconds string for the given duration.
        :param duration: The duration.
        :return: The duration string.
        """
        return '%d.%03d' % (duration.seconds, duration.microseconds // 1000)

    def print_start(self):
        """
        Print the start string.
        """
        print(self.name, 'starting at', self.time_string(self.start_time))

    def print_stop(self):
        """
        Print the stop string.
        """
        print(self.name, 'finished at', self.time_string(self.stop_time), 'elapsed time', self.seconds_string(self.elapsed_time()))
