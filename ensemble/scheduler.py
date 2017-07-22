from threading import Thread

class Scheduler:
	def __init__(self):
		self.threads = []

	def schedule(self, function, args):
		thread = Thread(target = function, args = args)
		self.threads.append(thread)
		thread.start()

	def joinAll(self):
		for thread in self.threads:
			thread.join()

