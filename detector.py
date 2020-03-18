import cv2, pickle
import numpy as np
#import pkg_resources
from os import listdir, path
from mtcnn.mtcnn import MTCNN
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input

#weights_file=pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy')
#steps_threshold=[0.6, 0.7, 0.7]
detector = MTCNN(min_face_size=20, scale_factor=0.709)
network = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

class RTAS():

	def __init__(self):
		self.path='train/'
		self.size=(224, 224)
		self.unknown='?'
		self.mindist=0.4
		self.rectcolor=(0, 0, 255)
		self.textcolor=(255, 255, 255)
		self.circlecolor=(255, 255, 255)
		self.cricleradius=2
		self.textthickness=2
		self.rectthickness=5
		self.circlethickness=-1
		self.metadata={}
		self.traindata()
		# if not path.exists("metadata.pickle"):
		# 	self.traindata()
		# else:
		# 	with open("metadata.pickle", "rb") as inp:
		# 		self.metadata=pickle.load(inp)

	def getimg(self, img=None, path=None):
		if path:
			img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
			return image.img_to_array(img)
			#return image.img_to_array(image.load_img(path))
		if img.any():
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return image.img_to_array(img)

	def boundingbox(self, img, results, landmarks=False, labels=None):
		if results:
			for index, result in enumerate(results):
				x, y, w, h = result['box']
				he, wi, cs = img.shape
				a, b = max(0, x), min(x+w, wi)
				c, d = max(0, y), min(y+h, he)
				img = cv2.rectangle(img, (a, c), (b, d), self.rectcolor, self.rectthickness)
				if landmarks:
					for key, point in result['keypoints'].items():
						img = cv2.circle(img, point, self.cricleradius, self.circlecolor, self.circlethickness)
				if labels:
					img = cv2.putText(img, labels[index][0], (a, c), cv2.FONT_HERSHEY_SIMPLEX, 1, self.textcolor, self.textthickness, cv2.LINE_AA)
		return img

	def detectfaces(self, img=None, path=None):
		try:
			if path and not img: img = self.getimg(path)
			return detector.detect_faces(img)
		except Exception as error:
			print(error)

	def getfeatures(self, img):
		resized = []
		results = self.detectfaces(img=img)
		for result in results:
			x, y, w, h = result['box']
			he, wi, cs = img.shape
			a, b = max(0, x), min(x+w, wi)
			c, d = max(0, y), min(y+h, he)
			scaledimg = cv2.resize(img[c:d, a:b], dsize=self.size)
			resized.append(scaledimg)
		if resized:
			return network.predict(preprocess_input(resized)), results
		return resized, results

	def traindata(self):
		trained = {}
		for label in listdir(self.path):
			filepath = self.path+label+'/'
			for file in listdir(filepath):
				imagepath = filepath+file
				try:
					feature, results = self.getfeatures(self.getimg(path=imagepath))
					if feature.any():
						if not label in trained:
							trained[label] = []
						trained[label].extend(feature)
				except Exception as error:
					print(error)
		with open("metadata.pickle", "wb") as out:
			pickle.dump(trained, out)
		self.metadata = trained

	def recognizefaces(self, img):
		labels = []
		testfeatures, results = self.getfeatures(img)
		if results:
			for testfeature in testfeatures:
				label, mindist = self.unknown, 1.0
				for key in self.metadata.keys():
					for trainedfeature in self.metadata[key]:
						dist = cosine(trainedfeature, testfeature)
						if mindist > dist: label, mindist = key, dist
				labels.append((self.unknown, 99.9)) if mindist >= self.mindist else labels.append((label, 99.9-mindist*100))
		return self.boundingbox(img, results, landmarks=False, labels=labels)

	def upload(self, path=None, live=False, recognizefaces=False):
		if path and not live:
			img = self.getimg(path=path)
			if recognizefaces:
		 		img = self.recognizefaces(img)
			img = cv2.cvtColor(np.array(image.array_to_img(img)), cv2.COLOR_RGB2BGR)
			cv2.imshow('image', img)
#			cv2.imwrite('output.png', img)
			key = cv2.waitKey(0)
			if key == 27:
				cv2.destroyAllWindows()
		if live and not path:
			cap = cv2.VideoCapture(-1)
			while(True):
				ret, frame = cap.read()
				if ret:
					img = self.getimg(img=frame)
					if recognizefaces:
						img = self.recognizefaces(img)
					cv2.imshow('livecapture', cv2.cvtColor(np.array(image.array_to_img(img)), cv2.COLOR_RGB2BGR))
					key = cv2.waitKey(1) & 0xFF
					if key == 27: break
			cap.release()
			cv2.destroyAllWindows()

if __name__ == "__main__":

	recognizer = RTAS()
	recognizer.upload(path='test/test.jpg', recognizefaces=True)
	#recognizer.upload(live=True, recognizefaces=True)