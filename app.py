from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
import logging
from numberDetector import *
from datetime import date

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

today = date.today()
date = today.strftime("%d-%m-%Y")

start = time.perf_counter()
number_detector = NumberDetector()
init_image = np.zeros((320, 320, 3), np.uint8)
number_detector.infer(init_image)
end = time.perf_counter()

formatter = logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
def setup_logger(name, log_file,level=logging.INFO):
	os.makedirs(os.path.dirname(log_file), exist_ok=True)
	handler = logging.FileHandler(log_file, mode='a')
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger


app = Flask(__name__)
# CORS(app)
app.config.from_object("config.ProductionConfig")
# logging.basicConfig(filename='record.log', level=logging.INFO, format=f'%(asctime)s %(levelname)s : %(message)s')

logger = setup_logger('record', 'logs/record/{}_record.log'.format(date))
bus_logger = setup_logger('bus', 'logs/bus/{}_bus.log'.format(date))
logger.info('Models loaded in {}s'.format(round(end - start, 2)))


@app.before_first_request
def _run_on_init():
	log = logging.getLogger('werkzeug')
	log.setLevel(logging.ERROR)


@app.route('/')
def hello_world():  # put application's code here
	return 'Hello World!'


@app.route('/images', methods=['POST'])
# @cross_origin()
def upload_file():
	start = time.time()
	if not request.form or len(request.form) != 2:
		logger.error('No threshold values in request | status_code: {}'.format(400))
		return ""
	box_threshold = float(request.form['box_threshold'])
	number_threshold = float(request.form['number_threshold'])
	if 'file' not in request.files:
		logger.error('No file in request | status_code: {}'.format(200))
		return ""

	file = request.files['file']
	if file.filename == '':
		app.logger.error('No selected file | status_code: {}'.format(404))
		return ""

	if file:
		start = time.time()
		image = cv2.imdecode(np.frombuffer(request.files['file'].stream.read(), np.uint8), cv2.IMREAD_UNCHANGED)
		# numbers_list, confidences_list = [], []
		grzyb = ''
		confidence = ''
		number_detector.infer(image)
		a, b = number_detector.filter_numbers(number_threshold)
		if all((a, b)):
			grzyb = a
			confidence = b
		response = {
			# 'boxes': len(boxes),
			# 'numbers': len(numbers_list),
			'grzyb': grzyb,
			'confidence': confidence
		}
		end = time.time()
		# print(end - start)
		if file.filename == 'now.jpg':
			bus_logger.info('File \'{}\' processed in {}s | status_code: {}'.format(file.filename, round((end - start), 2), 200))
		else:
			logger.info('File \'{}\' processed in {}s | status_code: {}'.format(file.filename, round((end - start), 2), 200))
		return jsonify(response)
		# return 'test'
	return ""


if __name__ == '__main__':
	app.run()