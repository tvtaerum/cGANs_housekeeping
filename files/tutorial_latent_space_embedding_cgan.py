# example of a gan for generating faces
import numpy as np
import pandas as pd
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import append
from numpy.random import random
from numpy.random import randint
from numpy.random import shuffle
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from matplotlib import patheffects as path_effects
import collections
from tensorflow.keras.models import load_model
from tensorflow import get_logger as log

#  SET YOUR FLAGS
qErrorHide = False
if qErrorHide:
    print("\n***REMEMBER:  WARNINGS turned OFF***\n***REMEMBER:  WARNINGS turned OFF***\n")
    log().setLevel('ERROR')

#    INDICATE IF STARTING FRESH OR CONTINUING FROM PREVIOUS RUN
qRestart = False
if qRestart:
    epochs_done = 105
    epochs_goal = 115
else:
    epochs_done = 0
    epochs_goal = 100 

# define the standalone discriminator model
def define_discriminator(in_shape=(80,80,3), n_classes=4):
	print("**********  ENTERED discriminator  *****************")
	##### foundation for labels
	in_label = Input(shape=(1,))
	embedding_layer = Embedding(n_classes, 8)
	# embedding_layer.trainable = False
	li = embedding_layer (in_label)
	n_nodes = in_shape[0] * in_shape[1]
	print(">>embedding>> in_shape[0], in_shape[1], n_nodes: ", in_shape[0], in_shape[1], n_nodes)
	li = Dense(n_nodes)(li)
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	dropout = 0.1
	in_image = Input(shape=in_shape)
	print("\nin_image: ", in_image)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	print("\nmerge.shape: ", merge.shape)
	# sample to 80x80
	fe = Conv2D(128, (5,5), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(dropout)(fe)
	print("fe.shape: ", fe.shape)
	# downsample to 40x40
	fe = Conv2D(128, (5,5), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# fe = Dropout(dropout)(fe)
	print("fe.shape: ", fe.shape)
	# downsample to 20x20
	fe = Conv2D(128, (5,5), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# fe = Dropout(dropout)(fe)
	print("fe.shape: ", fe.shape)
	# downsample to 10x10
	fe = Conv2D(128, (5,5), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# fe = Dropout(dropout)(fe)
	print("fe.shape: ", fe.shape)
	# downsample to 5x5
	fe = Conv2D(128, (5,5), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# fe = Dropout(dropout)(fe)
	print("fe.shape: ", fe.shape)
	# flatten feature maps
	fe = Flatten()(fe)
	# fe = Dropout(dropout)(fe)
	print("fe flatten shape: ", fe.shape)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	print("out_layer.shape: ", out_layer.shape)
	# define model
	model = Model([in_image, in_label], out_layer)
	print("\nmodel: ", model)
	# compile model
	opt = Adamax(lr=0.00007, beta_1=0.08, beta_2=0.999, epsilon=10e-8)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	print("\nembedding_layer.get_weights(): \n",embedding_layer.get_weights())
	model.summary()
	plot_model(model, to_file='cgan/discriminator_model.png')
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=4):
	print("**********  ENTERED generator  *****************")
	##### foundation for labels
	in_label = Input(shape=(1,))
	embedding_layer = Embedding(n_classes, 8)
	embedding_layer.trainable = True
	li = embedding_layer (in_label)
	n_nodes = 5 * 5
	li = Dense(n_nodes)(li)
	li = Reshape((5 , 5, 1))(li)
	print("generator...  n_nodes, li.shape: ", n_nodes, li.shape)
	##### foundation for 5x5 image
	in_lat = Input(shape=(latent_dim,))
	n_nodes = 128 * 5 * 5
	genX = Dense(n_nodes)(in_lat)
	genX = LeakyReLU(alpha=0.2)(genX)
	genX = Reshape((5, 5, 128))(genX)
	dropout = 0.1
	print("genX.shape: ", genX.shape)
	##### merge image gen and label input
	merge = Concatenate()([genX, li])
	print("merge.shape: ", merge.shape)
	##### create merged model
	# upsample to 10x10
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	print("gen after CV2DT.shape: ", gen.shape)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Dropout(dropout)(gen)
	print("gen.shape: ", gen.shape)
	# upsample to 20x20
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	print("gen.shape: ", gen.shape)
	# upsample to 40x40
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	print("gen.shape: ", gen.shape)
	# upsample to 80x80
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	print("gen.shape: ", gen.shape)
	# output layer 80x80x3
	out_layer = Conv2D(3, (5,5), activation='tanh', padding='same')(gen)
	print("out_layer.shape: ", out_layer.shape)
	# define model
	model = Model(inputs=[in_lat, in_label], outputs=out_layer)
	opt = Adamax(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
	model.compile(loss=['binary_crossentropy'], optimizer=opt)
	print("\nembedding_layer.get_weights(): \n",embedding_layer.get_weights())
	model.summary()
	plot_model(model, to_file='cgan/generator_model.png')
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	print("**********  ENTERED gan  *****************")
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adamax(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	model.summary()
	plot_model(model, to_file='cgan/gan_model.png')
	return model

# assign categories
def assign_categs(df, lenrows):
	print("\n*****  ATTRIBUTES: \n", df.mean())

	face_male = df['Male']
	print("face_male: ", face_male.mean())
	face_male = np.where(face_male > 0, 1, face_male)
	print("face_male: ", face_male.mean())

	face_high_cheekbones = df['High_Cheekbones']
	print("face_high_cheekbones: ", face_high_cheekbones.mean())
	face_high_cheekbones = np.where(face_high_cheekbones > 0, 1, face_high_cheekbones)
	print("face_high_cheekbones: ", face_high_cheekbones.mean())

	face_big_lips = df['Big_Lips']
	print("face_big_lips: ", face_big_lips.mean())
	face_big_lips = np.where(face_big_lips > 0, 1, face_big_lips)
	print("face_big_lips: ", face_big_lips.mean())

        # replace vectors with category value
	categs = np.zeros(lenrows, dtype=int)
	print("categ.mean()): ", categs.mean())
	categs = np.where(face_male > 0, 1, categs)
	print("add face_male: categs.mean()): ", categs.mean())
	categs = np.where((face_high_cheekbones > 0)&(categs==0), 2, categs)
	print("add high_cheekbones: categs.mean()): ", categs.mean())
	categs = np.where((face_big_lips > 0)&(categs==0), 3, categs)
	print("add big lips: categs.mean()): ", categs.mean())
	print("\ncategs: \n", categs)
	return categs

def get_cumProbs(freqCategs, categs):
	freqLists = [freqCategs[i][1] for i in range(len(freqCategs))]
	freqListX = asarray(freqLists, dtype=np.float32)
	print("freqListX: ", freqListX)
	print("len(categs): ", len(categs))
	cumProbs = freqListX/len(categs)
	print("cumProbs: ", cumProbs)
	cumProbs = append((0.0),cumProbs)
	for i in range(len(cumProbs)-1):
		cumProbs[i+1]=cumProbs[i]+cumProbs[i+1]
	print("cumProbs: ", cumProbs)
	return cumProbs


 
# load and prepare training images
def load_real_samples():
	# load the face dataset
	data = load('celeb/img_align_celeba_attractive.npz')
	X = data['arr_0']
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	data = pd.read_csv('celeb/list_attr_celeba.csv', nrows=100000)
	data = data.drop("image_id",axis=1)
	ids  = load('celeb/ids_align_celeba_attractive.npz')
	idsX = ids['arr_0']
	dataX = list()
	for i,id in enumerate(idsX):
		dataVal = data[id:id+1].values
		dataVal = np.where(dataVal==-1, 0, dataVal)
		dataX.append(dataVal)
	cols = data.columns
	lencols = len(cols)
	print("cols: ", cols)
	lenrows = len(dataX)
	dataVals = asarray(dataX[0:]).reshape((lenrows,lencols),)
	df = pd.DataFrame(data=dataVals,columns=cols)
	pd.options.display.float_format = '{:,.3f}'.format
	categs = assign_categs(df, lenrows)
	freqCategs = list(collections.Counter(sorted(categs)).items())
	print("freqCategs: ", freqCategs)
	cumProbs = get_cumProbs(freqCategs, categs)
	return [X, categs], cumProbs
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# retrieve selected images
	X, labels = images[ix], labels[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, cumProbs, n_classes=4):
	# print("generate_latent_points: ", latent_dim, n_samples)
	initX = -3.0
	rangeX = 2.0*abs(initX)
	stepX = rangeX / (latent_dim * n_samples)
	x_input = asarray([initX + stepX*(float(i)) for i in range(0,latent_dim * n_samples)])
	shuffle(x_input)
	# generate points in the latent space
	z_input = x_input.reshape(n_samples, latent_dim)
	randx = random(n_samples)
	labels = np.zeros(n_samples, dtype=int)
	for i in range(n_classes):
		labels = np.where((randx >= cumProbs[i]) & (randx < cumProbs[i+1]), i, labels)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, cumProbs):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples, cumProbs)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y
 
# create and save a plot of generated images
def save_plot(examples, labels, epoch, n=10):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		fig = plt.subplot(n, n, 1 + i)
		strLabel = str(labels[i])
		# turn off axis
		fig.axis('off')
		fig.text(8.0,20.0,strLabel, fontsize=6, color='white')
		# plot raw pixel data
		fig.imshow(examples[i])
	# save plot to file
	filename = 'celeb/results/generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
	
def save_real_plots(dataset, nRealPlots = 5, n=10, n_samples=100):
	# plot images
	for epoch in range(nRealPlots):
		if epoch%5==0:
			print("real_plots: ", epoch)
		# prepare real samples
		[X_real, labels], y_real = generate_real_samples(dataset, n_samples)
		# scale from [-1,1] to [0,1]
		X_real = (X_real + 1) / 2.0
		for i in range(n * n):
			# define subplot
			fig = plt.subplot(n, n, 1 + i)
			strLabel = str(labels[i])
			# fig.title = strLabel
			# turn off axis
			fig.axis('off')
			fig.text(8.0,20.0,strLabel, fontsize=6, color='white')
			# plot raw pixel data
			fig.imshow(X_real[i])
		# save plot to file
		filename = 'celeb/real_plots/real_plot_e%03d.png' % (epoch+1)
		plt.savefig(filename)
		plt.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, gan_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	[X_real, labels_real], y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate([X_real, labels_real], y_real, verbose=0)
	# prepare fake examples
	[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, n_samples, cumProbs)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate([X_fake, labels], y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(X_fake, labels, epoch)
	# save the generator model tile file
	filename = 'celeb/results/generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
	filename = 'celeb/results/generator_model_gan%03d.h5' % (epoch+1)
	gan_model.save(filename)
	filename = 'celeb/results/generator_model_dis%03d.h5' % (epoch+1)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	d_model.save(filename)
	d_model.trainable = False
	for layer in d_model.layers:
		layer.trainable = False


def restart(epochs_done):
	# gen_weights = array(model.get_weights())
	print("****  PULLING IN EPOCH: ", epochs_done)
	filename = 'celeb/results/generator_model_dis%03d.h5' % (epochs_done)
	d_model = load_model(filename, compile=True)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	d_model.summary()
	filename = 'celeb/results/generator_model_%03d.h5' % (epochs_done)
	g_model = load_model(filename, compile=True)
	g_model.summary()
	gan_model = define_gan(g_model, d_model)
	gan_model.summary()
	return d_model, g_model, gan_model

 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, epochs_goal=100, n_batch=128, epochs_done=0):
	nTryAgains = 0
	nTripsOnSameSavedWts = 0
	nSaves = 0
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	d_trainable_weights = np.array(d_model.get_weights())
	g_trainable_weights = np.array(g_model.get_weights())
	gan_trainable_weights = np.array(gan_model.get_weights())
	now = time.time()
	ij = 0
	ijSave = -100
	# manually enumerate epochs
	for i in range(epochs_done, epochs_goal):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			ij+=1
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			qDebug=False
			# update discriminator model weights
			dis_loss, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, cumProbs)
			gen_loss, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch, cumProbs)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			gan_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			if (j+1) % 5==0 or dis_loss > 1.10 or gen_loss > 1.10 or gan_loss > 1.80:
				diff = int(time.time()-now)
				print('>%d/%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f, secs=%d, tryAgain=%d, nTripsOnSameSavedWts=%d, nSaves=%d' %
					(i+1, epochs_goal, j+1, bat_per_epo, dis_loss, gen_loss, gan_loss, diff, nTryAgains, nTripsOnSameSavedWts, nSaves))
			if dis_loss > 0.30 and dis_loss < 0.95 and gen_loss > 0.25 and gen_loss < 0.95 and gan_loss > 0.40 and gan_loss < 1.50:
				nTripsOnSameSavedWts = 0
				if ij - ijSave > 8:
					nSaves+=1
					ijSave = ij
					d_trainable_weights = np.array(d_model.get_weights())
					g_trainable_weights = np.array(g_model.get_weights())
					gan_trainable_weights = np.array(gan_model.get_weights())
			if (dis_loss < 0.001 or dis_loss > 2.0) and ijSave > 0:
				nTryAgains+=1
				nTripsOnSameSavedWts+=1
				print("LOADING d_model",j+1," from ",ijSave)
				d_model.set_weights(d_trainable_weights)
			if (gen_loss < 0.001 or gen_loss > 2.0) and ijSave > 0:
				nTryAgains+=1
				nTripsOnSameSavedWts+=1
				print("LOADING g_model",j+1," from ",ijSave)
				g_model.set_weights(g_trainable_weights)
			if (gan_loss < 0.010 or gan_loss > 4.50) and ijSave > 0:
				nTryAgains+=1
				nTripsOnSameSavedWts+=1
				print("LOADING gan_models",j+1," from ",ijSave)
				gan_model.set_weights(gan_trainable_weights)
			# if (j+1) % 10 == 0:
				# summarize_performance(i, g_model, d_model, dataset, latent_dim)
			if nTripsOnSameSavedWts > 20:
				print("**********  Too many rebuilds  **************")
				summarize_performance(i, g_model, d_model, dataset, latent_dim)
				import sys
				sys.exit(0)
		# evaluate the model performance, sometimes
		if (i+1) % 1 == 0:
			summarize_performance(i, g_model, d_model, gan_model, dataset, latent_dim)

# size of the latent space
latent_dim = 100

if qRestart:
        d_model, g_model, gan_model = restart(epochs_done = epochs_done)
else:
        # create the discriminator
        d_model = define_discriminator()
        # create the generator
        g_model = define_generator(latent_dim)
        # create the gan
        gan_model = define_gan(g_model, d_model)
        
# load image data
dataset, cumProbs = load_real_samples()
save_real_plots(dataset, nRealPlots=2)
train(g_model, d_model, gan_model,  dataset, latent_dim, epochs_goal=epochs_goal, n_batch=64, epochs_done=epochs_done)
