## cGANs_with_embedding - housekeeping
### Housekeeping python code for training and utilizing cGans with embedding.  

In particular I thank Jason Brownlee for his brilliant work and tutorials at https://machinelearningmastery.com (plain text citations follow), Jeff Heaton for his insights on embedding https://www.youtube.com/user/HeatonResearch, and Iván de Paz Centeno for his work on face detection https://github.com/ipazc/mtcnn.  I found their code work to be complete, it runs 'out of the box' and they do a wonderful job of explaining why their streams work.  The test of great tutorials is whether they motivate a person to venture beyond the safe haven of working code - these tutorials delivered as promised.  

#### Motivation for housekeeping:
However, even the best tutorials can leave a person scratching their head wondering why their "minor" changes resulted in various forms of mode collapse.  In particular, the user might discover there are no obvious solutions to bad initial randomized values, no obvious way to start a stream from where it left off, no apparent explanation for generated images which are fuzzy and obscure, warning messages that suddenly show up and cannot be turned off, and no obvious ways to vectorize generated outcomes when embedding is employed.   
<!-- ![Cliff Dweller Huts](images/CliffDwellerHuts.png)  -->
<p align="center">
<img src="https://github.com/tvtaerum/cGANs_with_embedding---housekeeping/blob/master/images/CliffDwellerHuts.png" width="650" height="280">
</p>
In particular, the user may not have enough memory to use the code 'out of the box', the user may have to run the stream 20 or 30 times before it avoids mode collapse, attempts to debug Tensorflow or Keras are hindered by never ending warning messages, matching dimensions of generator and discriminator models may be difficult, the stream may be unable to start from where it left off, the user may run into issues with dated, or system specific code... there are so many obstacles that get in the way of operationalizing what ought to be a trivial process.
<p>
As such, while good tutorials make coding as bare bones as possible so that it's easy to explain and understand the concepts being taught, the code delivered here goes somewhat in the opposite direction.  These Python programs spend much of their time housekeeping so that they're able to carry on.    
</p>
#### Citations:
    Jason Brownlee, How to Develop a Conditional GAN (cGAN) From Scratch, Available from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch, accessed January 4th, 2020.
    Jason Brownlee, How to Explore the GAN Latent Space When Generating Faces, Available from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network, accessed January 13th, 2020.
    Jeff Heaton, Jeff Heaton's Deep Learning Course, Available from https://www.heatonresearch.com/course/, accessed February, 2020.
    Iván de Paz Centeno, MTCNN face detection implementation for TensorFlow, as a PIP package, Available from https://github.com/ipazc/mtcnn, accessed February, 2020.

#### Deliverables:
  1.  description of issues identified and resolved
  2.  code fragments illustrating the core of how the issue was resolved
  3.  a cGan Python program with embedding

#### Cautions:
There are a numbers of definitions and perspective which I use coming out of my background in analytics.  
  1.  stream:  the process of moving data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence occurs when there are no apparent improvements in a subjective evaluation of clarity of images being generated.   
  3.  limited applicability:  the methods described work for a limited set of data and cGan problems.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss.  
  
#### Software and hardware requirements:
    - Python
        - Numpy
        - Tensorflow with Keras
        - Matplotlib
    - GPU is highly recommended
    - Operating system used for development and testing:  Windows 10

##### The process:

 Using a cGAN as illustration, I provide limited working solutions to the following problems:

  1.  is there an automatic way to recover before "mode collapse"?
  2.  is there a way to restart a cGAN which has not completed convergence?
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I use embedding when I have descriptions of pictures?
  6.  how can I vectorize from generated face to generated face when using embedding?
  7.  what other adjustments might be applied?
        - selecting only faces with certain features (e.g. attractiveness)
	- adjusting for memory requirements
        - changing optimization from Adam to Adamax for embedding
        - changing number of iterations due to memory issues
	- shutting off Tensorflow warning messages
        - adding label to pictures

### 1.  is there an automatic way to recover from some "mode collapse"?:
Even with reasonable learning rates, convergence can slide to "mode collapse" and require a manual restart.  The stream provides one way of giving intial estimates multiple but limited opportunities to halt it's slide towards mode collapse.  The process also allows the stream to retain whatever progress it has made towards convergence.  
```Python
		if (d_loss1 < 0.001 or d_loss1 > 2.0) and ijSave > 0:
			print("RELOADING d_model weights",j+1," from ",ijSave)
			d_model.set_weights(d_trainable_weights)
		if (d_loss2 < 0.001 or d_loss2 > 2.0) and ijSave > 0:
			print("RELOADING g_model weights",j+1," from ",ijSave)
			g_model.set_weights(g_trainable_weights)
		if (g_loss < 0.010 or g_loss > 4.50) and ijSave > 0:
			print("RELOADING gan_models weights",j+1," from ",ijSave)
			gan_model.set_weights(gan_trainable_weights)
```
It is apparent there is a relationship between model loss and mode collapse.  The previous programming fragment illustrates an approach which often prevents a stream from mode collapse.  It depends on having captured disciminator weights, generator weights, and gan weights either during initialization or later in the process when all model losses are within bounds.  The definition of model loss bounds are arbitrary but reflect expert opinion about when losses are what might be expected and when they are clearly much too high or much too low.  Reasonable discriminator and generator losses are between 0.1 and 1.0, and their arbitrary bounds are set to between 0.001 and 2.0.  Reasonable gan losses are between 0.2 and 2.0 and their arbitrary bounds are set to 0.01 and 4.5.  

What happens then is discriminator, generator, and gan weights are collected when all three losses are "reasonable".  When an individual model's loss goes out of bounds, then the last collected weights for that particular model (and only that model) are replaced, leaving the other model weights are they are, and the process moves forward.  The process stops when mode collapse appears to be unavoidable even when model weights are replaced.  This is identified as when a particular set of model weights continue to result in out of bound model losses.    

### 2.  is there a way to restart a cGAN which has not completed convergence:
There is nothing quite as upsetting as running a stream and six days later the process collapses when it appears to be 90% complete.  Needless to say, your steam needs to be prepared for such an event.  Even with preparation, attempts to restart can result in endless warnings about parameters being not trainable, dimensions of weights being wrong for discriminate, generative, and gan models, and optimizations that collapse.  There is a lot of helpful advice if you just want to inspect weights but after six days, you want to start where you left off - how do you do it?  It's important to note that cGAN will not properly restart unless you resolve the issues of what is trainable, what are the correct dimensions, and what are viable models.    

Once issues with dimensions and what is trainable are resolved, there are then problems where models suffer from model collapse when attempts are made to restart the cGAN.  What happened?  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  As such, if you wish to continue executing the stream, rather than simply inspect weights, you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there only to make the discriminator and generator work together.  

Restarting a cGAN requires saving models in case they are required, and loading models.  When saving a model, the layers that get saved are those which are trainable.  It's worth noting that when running a cGAN model, the discriminator model is set to trainable=False.   Needless to say, there are places in a stream where layers or models need to be fixed (set to trainable=False).  The following code fragment is required when saving the discriminator model:  
```Python
	filename = 'celeb/results/generator_model_dis%03d.h5' % (epoch+1)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	opt = Adamax(lr=0.00007, beta_1=0.08, beta_2=0.999, epsilon=10e-8)
	d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	d_model.save(filename)
	d_model.trainable = False
	for layer in d_model.layers:
		layer.trainable = False
```
And when loading:
```Python
	filename = 'celeb/results/generator_model_dis%03d.h5' % (ist_epochs)
	d_model = load_model(filename, compile=True)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	d_model.summary()
```
Matters are made slightly more complicated if I want to be able to make the embedding layers fixed once training is complete but add other pictures to the training.    

### 3.  are there non-random initialization values that can be useful?
I have found no reason to believe that normal like distributions of random values are better than uniform distributions of random values.  I did a little bit of work on that issue and found that leptokurtic distributions were poorest in generating good images.  A virtue of normal-like distributions is the values further away from the centroid provide more information than those close to the centroid.  Do we really believe this when generating images?  For most of the results discussed here, we are in a bounded 100-dimensional space and there is no reason that I am aware of for fine tuning central values as opposed to values at the upper and lower extremes.   
```Python
def generate_latent_points(latent_dim, n_samples, cumProbs, n_classes=4):
	# print("generate_latent_points: ", latent_dim, n_samples)
	initX = -3.0
	rangeX = 2.0*abs(initX)
	stepX = rangeX / (latent_dim * n_samples)
	x_input = asarray([initX + stepX*(float(i)) for i in range(0,latent_dim * n_samples)])
	shuffle(x_input)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	randx = random(n_samples)
	labels = np.zeros(n_samples, dtype=int)
	for i in range(n_classes):
		labels = np.where((randx >= cumProbs[i]) & (randx < cumProbs[i+1]), i, labels)
	return [z_input, labels]
```
 
### 4.  how important is the source material (original pictures of faces)?
There is a well known acronym GIGO (garbage in, garbage out), and no one is surprised by words of advice to examine the data going into the stream.  When the data going into a stream is a derivative of another process, as in this case, it is important to examine the quality of the input data before declaring a process to be useful or invalid.  
```Python
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
			fig.axis('off')
			fig.text(8.0,20.0,strLabel, fontsize=6, color='white')
			# plot raw pixel data
			fig.imshow(X_real[i])
		# save plot to file
		filename = 'celeb/real_plots/real_plot_e%03d.png' % (epoch+1)
		plt.savefig(filename)
		plt.close()
```
Even after working for 50 years in data/predictive analytics, it's easy to forget to look at the transformed data that goes into an analysis, no matter what the subject matter of the analysis is.  
### 5.  how can I use embedding when I have descriptions of pictures?
There are circumstances where we want to insure that the predicted output has particular characteristics, such as whether the face is attractive, if they are male or female, if they have high cheek bones, large lips, and other features.  At some point, it will be possible to create realistic GAN generated pictures of models wearing particular clothing, with specific expressions, and poses for catalogues.  
There were three parts to this process:  
1. selecting a subset of faces (only those identified as being "attractive"):
Details are discussed in section 7. 
2. identifying the characteristics or attributes to be used and their probabilities in the population of objects:
    a. 0 = featured as being attractive and not male and not high cheek bone and not large lips
    b. 1 = featured as being attractive and male
    c. 2 = featured as being attractive and not male and high cheek bone
    d. 3 = featured as being attractive and not male and not high cheek bone and large lips 
3. setting up the cGAN so that it will generate and save faces based on the attributes (embeddings) associated with a picture.  
![random generated faces](images/4X10RandomlyGeneratedFaces.png)
As you can see, there are four kinds of embedding and the identity of the embedding (from 0 to 4) is included in the generated face. In many ways, those faces identified as being 0 are "featureless".  Those faces identified as 1 (male), are clearly male.  Those faces identifed as 2 are female with high cheek bones.  Feature 3 identifies those faces which supposedly have large lips.  Explanations for what we found is discussed in section 6.  
### 6.  how can I vectorize from generated face to generated face when using embedding?
Jeff Brownlee provides what I believe is a brilliant example of how to vectorize from one face to another face.  We vectorize two generated faces and, for the same 100-dimensional space, add embedding with four attributes which were described in section 5.      

![vectorized range of faces](images/4X10VectorizedRangeOfFaces.png)
Going from left to right, we see Here each row reflects a feature 

### 7.  other changes that can be applied?
- only selecting faces with certain characteristics - such as attractiveness
- adjusting for memory requirements
- changing optimization from Adam to Adamax for embedding
```Python
	opt = Adamax(lr=0.00007, beta_1=0.08, beta_2=0.999, epsilon=10e-8)
```
- changing number of iterations due to memory issues
- shutting off Tensorflow warnings
- adding label to the pictures 
  
