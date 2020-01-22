from colorlog import ColoredFormatter
import logging



def Logger(log_level = logging.DEBUG):
	global log

	try:
		log
	except:
		LOGFORMAT = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
		logging.root.setLevel(log_level)

		formatter = ColoredFormatter(LOGFORMAT)
		
		stream = logging.StreamHandler()
		stream.setLevel(log_level)
		stream.setFormatter(formatter)
		
		log = logging.getLogger('pythonConfig')
		log.setLevel(log_level)
		log.addHandler(stream)
		log.log_level = log_level


	return log
