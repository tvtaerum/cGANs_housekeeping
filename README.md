## cGANs: embedding in images - housekeeping
### Housekeeping python code for training and utilizing cGans with embedding.  

In particular I thank Jason Brownlee for his brilliant work and tutorials at https://machinelearningmastery.com (citations below in project), Iván de Paz Centeno for his work on face detection at https://github.com/ipazc/mtcnn, and  Jeff Heaton for his insights on embedding at https://www.youtube.com/user/HeatonResearch.  It took me more than a year digging into GANs on the Internet to find programmers and instructors whose code work was complete and ran 'out of the box' (except for system related issues) and they also do a wonderful job of explaining why their streams work.  Jason Brownlee and Jeff Heaton are not the only instructors I found who are very impressive but they are the source of insights for this presentation. The description which follows can be considered a branch/fork of Jason Brownlee's tutorial on "vector arithmetic with faces".  
<p align="center">
<img src="/images/CliffDwellerHuts.png" width="650" height="270">
</p>  

### Motivation for housekeeping:
Major issues with cGANs include mode collapse and unscheduled interruptions of long running programs.  Even the best GAN program can leave a person scratching their head wondering why their "minor" changes resulted in various forms of mode collapse.  In particular, the user might discover there are no obvious solutions to bad initial randomized values, no obvious way to start a stream from where it left off, no apparent explanation for generated images which are fuzzy and obscure, warning messages that cannot be turned off, and no obvious ways to vectorize generated images when embedding is employed.   
<p align="center">
<img src="/images/attractiveFaces.png" width="650" height="135">
</p>
In particular, the user may not have enough memory to use the code 'out of the box', it may require 20 or 30 attempts before it avoids mode collapse, attempts to debug Tensorflow or Keras may be hindered by never ending warning messages, matching dimensions of generator and discriminator models can be difficult, the suggested learning rates may not be appropriate given small modifications, the user may run into issues with dated, or system specific code... there are so many obstacles that get in the way of operationalizing what ought to be a straight forward but complex process.
</p>

In the face of so many constraints and the ease with which cGANs slide into mode collapse, it can be particularly difficult for the novice (like myself) to make tutorial material work.  While good tutorials make coding as bare bones as possible and adhere as closely as possible to industrial standards so that it's easy to explain and understand the concepts being taught, the code delivered here goes in a different direction.  The Python programs included here invest a greater amount of coding in housekeeping so that the novice, after they've made the essential changes required by their limited environment, will have a better chance of replicating the work done by those who are expert in the field.      

### Citations:
<dl>
<dt> Jason Brownlee, How to Develop a Conditional GAN (cGAN) From Scratch,</dt><dd> Available from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch, accessed January 4th, 2020. </dd>
<dt>Jason Brownlee, How to Explore the GAN Latent Space When Generating Faces, </dt><dd>Available from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network, accessed January 13th, 2020. </dd>
<dt>Iván de Paz Centeno, MTCNN face detection implementation for TensorFlow, as a PIP package,</dt><dd> Available from https://github.com/ipazc/mtcnn, accessed February, 2020. </dd>
<dt>Jeff Heaton, Jeff Heaton's Deep Learning Course,</dt><dd> Available from https://www.heatonresearch.com/course/, accessed February, 2020. </dd>
</dl>

### Deliverables:
  1.  description of issues identified and resolved within specified limitations
  2.  code fragments illustrating the core of how an issue was resolved
  3.  a Python program to prepare images for selection and training
  4.  a cGan Python program with embedding
  5.  a Python program which vectorizes images generated with embedding

### Limitations and caveates:

  1.  stream:  refers to the overall process of streaming/moving data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence is sufficient when there are no apparent improvements in a subjective evaluation of clarity of images being generated.   
  3.  limited applicability:  the methods described work for a limited set of data and cGan problems.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss - when model loss is extreme (too high or too low) then there is mode collapse.  
  
### Software and hardware requirements:
    - Python version 3.7.3
        - Numpy version 1.17.3
        - Tensorflow with Keras version 2.0.0
        - Matplotlib version 3.0.3
    - GPU is highly recommended
    - Operating system used for development and testing:  Windows 10

#### The process:

 Creating a cGAN as illustration, I provide limited working solutions to the following problems:

<ol type="1">
  <li>is there an automatic way to recover before "mode collapse"?</li>
  <li>is there a way to restart a cGAN which is interrupted or has not completed convergence?</li>
  <li>are there different kinds of random initialization values that can be useful?</li>
  <li>how important is the source material (original images of faces)?</li>
  <li>how can I use embedding when I have descriptions of images?</li>
  <li>how can I vectorize from generated face to generated face when using embedding?</li>
  <li>what other adjustments might be applied?</li>
<ol type="a">
	<li>selecting only faces with certain features (e.g. attractiveness)</li>
        <li>adjusting for memory requirements</li>
        <li>changing optimization from Adam to Adamax for embedding</li>
        <li>shutting off Tensorflow warning messages</li>
        <li>stamping labels on images</li>
</ol>
  <li>cGan stream:
<ol type="a">        
	<li>download celebrity images from https://www.kaggle.com/jessicali9530/celeba-dataset</li>
        <li>select out subset of images with attractive faces</li>
        <li>cGan stream</li>
        <li>vectorize images</li>
</ol>
</ol>

### 1.  is there an automatic way to recover from some "mode collapse"?:

Even with reasonable learning rates, convergence can slide into "mode collapse" and require a manual restart.  The stream provides one way of giving intial estimates multiple but limited opportunities to halt it's slide towards mode collapse.  The process also allows the stream to retain whatever progress it has made towards convergence while recovering from mode collapse.     

There are three critical measures of loss:
<ol>
	<li>dis_loss, _ = d_model.train_on_batch([X_real, labels_real], y_real)</li>
	<li>gen_loss, _ = d_model.train_on_batch([X_fake, labels], y_fake)</li>
	<li>gan_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)</li>
</ol>
Before examining the screen shot which comes below, I define the measures used to determine when mode collapse is imminent and recovery is necessary:
<table style="width:100%">
  <tr> <th> Column </th>    <th> measure </th>      <th> example </th>  </tr>
  <tr> <td> 1 </td>  <td> epoch/max_epochs </td>    <td> 1/100 </td>  </tr>
  <tr> <td> 2 </td>  <td> iteration/max_iterations  </td>    <td> 125/781 </td>  </tr>
  <tr> <td> 3 </td>  <td> discriminator loss </td>    <td> d1(dis)=0.020 </td>  </tr>
  <tr> <td> 4 </td>  <td> generator loss </td>    <td> d2(gen)=0.114 </td>  </tr>
  <tr> <td> 5 </td>  <td> gan loss </td>    <td> g(gan)=2.368 </td>  </tr>
  <tr> <td> 6 </td>  <td> run time (seconds) </td>   <td> secs=142 </td>  </tr>
  <tr> <td> 7 </td>  <td> number of restarts </td>    <td> tryAgain=0 </td>  </tr>
  <tr> <td> 8 </td>  <td> number of restarts using same base </td>    <td> nTripsOnSameSavedWts=0 </td>  </tr>
  <tr> <td> 9 </td>  <td> number of weight saves </td>    <td> nSaves=2 </td>  </tr>
</table>
There are three parts in the screen shots below: 
<p align="center">
<img src="/images/escapingModeCollapse.png" width="850" height="225">
</p>

In layer 1 of the screen shot above, we can see at epoch 1/100 and iteration 126/781, the discriminator loss has dropped to near zero and the gan loss is beginning to escalate.  Left to itself, the discriminator loss would drop to zero and we would see mode collapse.  In this case, the saved discriminator weights (d_weights) are loaded back in and the stream recovers.  

In layer 2, we see proof of recovery at the end of epoch 1 with discriminator loss at 0.459 and gan loss at 1.280.  At this point, the accuracy for "real" is 77% and fake is 93%.  These values may not sound impessive until we look at the generated faces from epoch 1.

In layer 3, we see a screen shot of the generated faces from epoch 1 out of 100 epoches.  

So how can we recover from a mode collapse?  The syntax below illustrates the core of the process:  

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
The previous programming fragment illustrates an approach which often prevents a stream from mode collapse.  It depends on having captured disciminator weights, generator weights, and gan weights either during initialization or later in the process when all model losses are within bounds.  The definition of model loss bounds are arbitrary but reflect expert opinion about when losses are what might be expected and when they are clearly much too high or much too low.  Reasonable discriminator and generator losses are between 0.1 and 1.0, and their arbitrary bounds are set to between 0.001 and 2.0.  Reasonable gan losses are between 0.2 and 2.0 and their arbitrary bounds are set to 0.01 and 4.5.  

What happens then is discriminator, generator, and gan weights are collected when all three losses are "reasonable".  When an individual model's loss goes out of bounds, then the last collected weights for that particular model are replaced, leaving the other model weights are they are, and the process moves forward.  The process stops when mode collapse appears to be unavoidable even when model weights are replaced.  This is identified when a particular set of model weights continue to be reused but repeatedly result in out of bound model losses.   

The programming fragment for saving the weights are:

```Python
	if d_loss1 > 0.30 and d_loss1 < 0.95 and d_loss2 > 0.25 and d_loss2 < 0.95 and g_loss > 0.40 and g_loss < 1.50:
		d_trainable_weights = np.array(d_model.get_weights())
		g_trainable_weights = np.array(g_model.get_weights())
		gan_trainable_weights = np.array(gan_model.get_weights())
```
Needless to say, there are a few additional requirements which can be found in the Python program available at the end of this README document.  For instance, if your stream goes into mode collapse just after saving your trainable weights, there is little likelihood that the most recently saved weights will save the recovery.  

It's important to note that a critical aspect of this stream is to help the novice get over the difficult challenge of making the first GAN program work.  As such, its focus is not simply on automatic ways to recover from mode collapse and methods of restarting streams, but on the debugging process that may be required.  To do this, we need constant reporting.  As we observe in the screen shot below, not every execution results in a requirement to load in most recent working trainable weights.  However, we do see information which may be helpful in understanding what is going on.  
<p align="center">
<img src="/images/nonEscapingModeCollapse.png" width="500" height="150">
</p>
Typically, the situation for loss is reported every five iterations.  As illustrated in the area in the red blocked area, when the program appears to be drifting into mode collapse, losses are reported on every iteration.  In the blue blocked area, we can see the generative loss beginning to incease beyond reasonable limits.  In the green blocked area, we see the tendency for when the discriminator or generator losses move beyond reasonable limits, the gans losses move out of range.  And finally, in the brown blocked area, we see a counter of the number of times weights have been saved to be used later in recovery.  

### 2.  is there a way to restart a cGAN which has not completed convergence:
There is nothing quite as problematic as running a program and six days later the process is interrupted when it appears to be 90% complete.  Like many others, I have run streams for over 21 days using my GPU before something goes wrong and I am unable to restart the process.  Progress is measured in "epochs".  There is no guarantee but with a bit of good fortune and cGAN steams which are properly set up, every epoch brings an improvement in clarity.  The images which follow illustrate observed improvements over epochs.  
<p align="center">
<img src="/images/improvedImagesOverEpochs.png" width="650" height="500">
</p>
  
The numbers on the left side are epochs which produced the observed results.  We can see the faint images of faces by epoch 5, good impressions of faces by epoch 45, details of faces by epoch 165 and small improvements by epoch 205.  We want to do better than being stuck at epoch 45 and we want to be able to continue from epoch 45 if the process is interrupted.  We are, in a sense, mapping from a 100-dimensional space to images of faces and it takes time to complete the mapping from representative parts of the 100-dimensional space.      
    
Needless to say, the steam needs to be prepared for interruptions.  Even with preparation, attempts to restart can result in warnings about model and/or layers being trainable=False, dimensions of weights being incompatable for discriminate, generative, and gan models, and optimizations that collapse.  It's important to note that cGAN will not properly restart unless you resolve the issues of what is trainable, what are the correct dimensions, and what are viable models. If your only interest is in examining weights and optimization, then warning messages can often be ignored.  If you wish to restart from where you left off, then you ignore warning messages at considerable risk.   
 
Once issues with dimensions and what is trainable are resolved, there are then problems where models suffer from mode collapse when attempts are made to restart the cGAN.  What happened?  If you wish to continue executing the program, my experience is you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there only to constrain and make the discriminator and generator work together.  
 
Restarting a cGAN requires saving models and their optimizations in case they are required after each epoch.  When saving a model, the layers that get saved are those which are trainable.  It's worth recalling that the discriminator model is set to trainable=False within the gan model.  Depending on the requirements, there may also be layers which are set to trainable=False.  In order to save the models, and recover the fixed weights, the weights must temporarily be set to trainable=True.  The following code fragment is required when saving the discriminator model:  
```Python
	filename = 'celeb/results/generator_model_dis%03d.h5' % (epoch+1)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
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
Setting the layers on an individual basis may seem overly detailed but it is a reminder that, in some circumstances, there are layers which may need to be set to trainable-False. 

Three parameters need to be changed in order to restart the process:  qRestart, epochs_done, epochs_goal.  These parameters are found near the beginning of the Python program.  
```Python
#    INDICATE IF STARTING OR CONTINUING FROM PREVIOUS RUN
qRestart = False
if qRestart:
    epochs_done = 105
    epochs_goal = 115
else:
    epochs_done = 0
    epochs_goal = 100
```
qRestart is set to True indicating the program needs to start from where it left off.
"epochs_done" refers to the number of epochs already completed.  
"epochs_goal" refers to how many epochs you think you'd like to complete.  


### 3.  are there different kinds of random initialization processes that can be helpful in accelerating convergence?
While the use of normal like distributions may be useful, there is no reason to believe that other distributions will not work.  A small investigation on my part suggested that leptokurtic distributions were poorest in generating good images.  For most of the results discussed here, I use a uniform distribution in a bounded 100-dimensional space.   
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
Substantially, the routine divides the range of values from -3.0 to +3.0 into equal intervals and then randomizes the values by a shuffle.  The process works - I'm still examining whether it accelerates convergence with images.  
 
### 4.  how important is the source material (original images of faces)?
In my attempts to improve the results of the generations, I initially overlooked a critical factor - what does the transformed data going into the cGAN look like.  When the data going into a stream is a derivative of another process, as in this case, it is critical to examine the quality of the input data before declaring the results to be useful or invalid.  

The code to examine the data going into the cGAN is trivial and is included in the final stream.  

![real faces rows](images/sampleRealImagesRows.png)

It's worth remembering that the GAN process sees the images at the convoluted pixel level - it sees every spot and wrinkle, every imperfection.   
![real faces](images/sampleRealImages.png)

In spite of all the imperfections in individual images, my belief is the final results are impressive.  Selecting out only faces featured as attractive helped in obtaining results which had considerable clarity.  

### 5.  how can I use embedding when I have descriptions of images?
There are circumstances where we want to insure that a generated image has particular characteristics, such as a face being attractive, selecting a particular gender, and having facial features such as high cheek bones and large lips.  Looking into the near future, it will be possible to create realistic GAN generated images of models wearing fashionable clothing, with specific expressions, and poses for catalogues.  In this example, we could enter in the features:  attractive, female, high cheek bones, and large lips in order to get many faces for fashion models.    

There were three parts to this process:  
1. selecting a subset of faces (only those identified as being "attractive"):
Details of the process are discussed in section 7. 
2. identifying the characteristics or attributes to be used and their probabilities in the population of images:
<ol type="a">
      <li>..... 0 = featured as attractive and female and not high cheek bone and not large lips</li>
	<li>..... 1 = featured as attractive and male</li>
	<li>..... 2 = featured as attractive and female and high cheek bone</li>
      <li>..... 3 = featured as attractive and female and not high cheek bone and large lips</li>
</ol>

3. setting up the cGAN so that it will generate and save faces based on the features (embeddings/labels) associated with an image.  
![random generated faces](images/4X10RandomlyGeneratedFaces.png)
There are four kinds of embedding and the identity of the embedding (0 thru 3) is included in the generated face. In many ways, those faces identified as being 0 are "female without high cheeck bones and without large lips".  Those faces identified as 1 (male), are clearly male.  Those faces identifed as 2 are female with high cheek bones.  Feature 3 identifies those faces which supposedly have large lips.  The labels (0 thru 3) are added when creating the image.  Explanations for what we found is discussed in section 6.  

### 6.  how can I vectorize from generated face to generated face when using embedding?
Jeff Brownlee provides a brilliant example of how to vectorize from one face to another face.  In addition to what Brownlee had done, we vectorize two generated faces and then, for the same 100-dimensional space, "add" the predictive value of the features through embedding as described in section 5. 

![vectorized range of faces](images/4X10VectorizedRangeOfFaces.png)
Going from left to right, we see the face on the left morphing into the face on the right.  When we compare each row, we see the four features described in section 5.  The only difference between each row are due to the predictive power of the embeddings/labels.  Of particular interest is comparing the second row (embedded value 1: attractive male) with the third row (embedded value 2: attractive female with high cheek bones). Everything except the embedding/label is identical.  

From an analytical perspective, comparing rows 3 and 4 (embedded value 2: attractive female with high cheek bones versus embedded value 3: attractive female with large lips) may provide insight into what a feature actually means.  While the persons identifying features may believe they are only looking at a feature, such as the size of lips, the analytical process of cGans identifies what is uniquely different in comparing rows three and four.  

```Python
            n_classes = 4     
            latent_dim = 100                  # 100 dimensional space
            pts, labels_input = generate_latent_points(latent_dim, n_samples, cumProbs)
            results = None
            for i in range(n_samples):        # interpolate points in latent space
                interpolated = interpolate_points(pts[2*i], pts[2*i+1])
                for j in range(n_classes):    # run each class (embedding label)
                    labels = np.ones(10,dtype=int)*j
                    X = model.predict([interpolated, labels])  # predict image based on latent points & label
                    X = (X + 1) / 2.0         # scale from [-1,1] to [0,1]
                    if results is None:
                        results = X
                    else:
                        results = vstack((results, X))   # stack the images for display
            plot_generated(filename, results, labels_input, 10, n_samples, n_classes)   #generate plot
```
The programming fragment illustrates the effect of embedding, where the generated latent points are identical but the embedded labels are different - resulting in generated images which are marketly different.  The effect of label information is most clearly illustrated when we compare row 2 (males) and row 3 (females with high cheek dones).  

### 7.  other changes that can be applied?

There are a number of other adjustments which were made in order to improve outcomes.  

#### a. select faces with certain characteristics - such as attractiveness - for analysis
 
Only faces identified as being attractive were included.  Given the attributes associated with attractiveness, such as symmetry, clarity and visibility, it appeared to be a good way to select out those faces which were complete.   
```Python
	# enumerate files
	for idx, filename in enumerate(listdir(directory)):
		# load the image
		pixels = load_image(directory + filename)
		# get face
		face = extract_face(model, pixels)
		if face is None:
			continue
		if data_attractive[idx] == -1.0:
			continue
```
#### b. adjust for memory requirements

Based on my own experiences, I'd surmise that one of the most frequent modifications novices have to make is making adustments so that the problem will fit on their GPU resources.  In many circumstances, this is done by adjusting batch sizes.  In this particular case, the fork required a change of n_batch from 128 to 64.  
```Python
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128, ist_epochs=0):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
...
train(g_model, d_model, gan_model,  dataset, latent_dim, n_epochs=n_epochs, n_batch=64, ist_epochs=ist_epochs)
```
While the changing the size of batch is trivial, it often has unexpected outcomes which result in mode collapse.  

#### c. change optimization from Adam to Adamax for embedding

While Adam optimizers are generally used, Adamax is recommended when there are embeddings.  
```Python
	opt = Adamax(lr=0.00007, beta_1=0.08, beta_2=0.999, epsilon=10e-8)
```
#### d. turn off Tensorflow warnings for debugging purposes
Tensorflow and Keras are both very good at giving warnings when syntax being used is out of date, dimensions do not match, or features (such as trainable=True) are not used as required.  The problem is you sometimes have to run through many warnings before seeing the impact of the issue.  In debugging circumstances, being able to shut off warnings can be helpful.  
```Python
qErrorHide = True
if qErrorHide:
    print("\n***REMEMBER:  WARNINGS turned OFF***\n***REMEMBER:  WARNINGS turned OFF***\n")
    log().setLevel('ERROR')
```
#### e. stamp labels on images 
Finally, it's helpful if the image has a label stamped on it so you can see, at a glance, whether or not the embedding matches what you believe ought to be features of the generated image.  
```Python
def save_plot(examples, labels, epoch, n=10):
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		fig = plt.subplot(n, n, 1 + i)
		strLabel = str(labels[i])
		fig.axis('off')
		fig.text(8.0,20.0,strLabel, fontsize=6, color='white')
		fig.imshow(examples[i])
	filename = 'celeb/results/generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
```
###  8.  cGan streams and data sources:
The following is an outline of the programming steps and Python code used to create the results observed in this repository.  There are three Python programs which are unique to this repository.  The purpose of the code is to assist those who struggled like I struggled to understand the fundamentals of Generative Adversarial Networks and to generate interesting and useful results beyond number and fashion generation.  My edits are not elegant... it purports to do nothing more than resolve a few issues which I imagine many novices to the field of Generative Adversarial Networks face.  If you know of better ways to do something, feel free to demonstrate it.  If you know of others who have found better ways to resolve these issues, feel free to point us to them.  

The recommended folder structure looks as follows:
<ul>
    <li>cGANs_housekeeping-master (or any folder name)</li>
	<ul>
       <li> files (also contains Python programs)</li>
	<ul>
		<li> <b>celeb</b></li>
		<ul>
			<li> <b>img_align_celeba</b> (contains about 202,599 images for data input)</li>
			<li> <b>real_plots</b> (contains arrays of real images for inspection)</li>
			<li> <b>results</b> (contains generated png images of faces and and h5 files for models saved by program)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
       <li> images (contains images for README file)</li>
	</ul>
</ul>
Those folders which are in <b>BOLD</b> need to be created. 

#### a. download celebrity images from https://www.kaggle.com/jessicali9530/celeba-dataset
#### b. select out subset of images with attractive faces and compress <a href="/files/images_convert_mtcnn_attractive_faces.py">MTCNN convert attractive faces</a>

When executing, you will get the following output:  
<p align="left">
<img src="/images/LoadingAndCompressing50000Images.png" width="200" height="100">
</p>  

It will create two files:
    ids_align_celeba_attractive.npz
    image_align_celeba_attractive.npz
    
#### c. cGan stream <a href="/files/tutorial_latent_space_embedding_cgan.py">cGan embedding</a>

Refer back to Python coding fragments for explanation on restarting program.

#### d. vectorize images <a href="/files/images_run_thru_models_1_restart_cgan.py">run thru faces using embedding</a> 

The list of images for visual examination depends on the lstEpochs variable below.  
```Python
directory = 'celeb/results/'
iFile = 0
for idx, filename in enumerate(listdir(directory)):
    if ".h5" in filename and not("_gan" in filename) and not("_dis" in filename):
        iFile += 1
        lstEpochs = [5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,150]
        if iFile in lstEpochs: 
            model = load_model(directory + filename)
            gen_weights = array(model.get_weights())
```
#### LICENSE  <a href="/LICENSE">MIT license</a>
