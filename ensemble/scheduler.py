class Scheduler:
	def __init__(self):
		self threads = []

	def schedule(function, args):
		thread = Thread(target = function, args = args)
    	self.threads.append(thread)
    	thread.start()

    def joinAll():
    	for thread in threads:
    		thread.join()

