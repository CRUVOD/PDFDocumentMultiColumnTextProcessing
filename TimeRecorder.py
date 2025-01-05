import time

class TimeRecorder(object):
    last_recorded_time: float
    start_time: float

    def __init__(self):
        self.last_recorded_time = 0
        self.start_time = 0

    def start(self):
        self.last_recorded_time = time.time()
        self.start_time = time.time()

    def record(self):
        elapsed_time = time.time() - self.last_recorded_time
        self.last_recorded_time = time.time()
        return str(elapsed_time) + "s"

    def total(self):
        return "Total time: " + str(time.time() - self.start_time) + "s"

