import sys
import time


def get_cur_time():
	return time.strftime('[%Y-%m-%d %H:%M:%S] ')


def std_log(text):
	cur_time = get_cur_time()
	print(cur_time + text)


def err_log(text):
	cur_time = get_cur_time()
	text = cur_time + text
	print(text, file=sys.stderr)


if __name__ == '__main__':
	err_log('hello world')
