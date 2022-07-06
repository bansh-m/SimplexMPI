import time

class Timer():
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.elapsed = 0

    def start(self):
        self.start_time = time.perf_counter()

    def start_seconds(self):
        self.start_time = time.time()

    def stop_seconds(self):
        self.stop_time = time.time()
        self.elapsed = self.stop_time - self.start_time

    def stop(self):
        self.stop_time = time.perf_counter()
        self.elapsed = self.stop_time - self.start_time