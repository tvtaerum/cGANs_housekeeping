## cGans_with_embedding - housekeeping
### Housekeeping python code for training and utilizing cGans with embedding.  
In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  Their code works 'out of the box' and they deliver what they promise.  In particular, the stream (Python program and process) is a derivative of tutorials by Jason Brownlee and insights on embeddings by Jeff Heaton.  The subject matter are faces derived by a process using the ipazc/mtcnn project by Iván de Paz Centeno. 

#### Motivation for housekeeping:
Even the best tutorials can leave a person scratching their head wondering if there are ways to make "minor" changes to the stream.  In particular, the user might discover there are no obvious solutions to bad initial randomized values, no obvious way to restart streams when convergence is not complete, no obvious way for figuring out why outcomes appear dirty or not clear, warning messages that suddenly show up and cannot be turned off, and no obvious ways to vectorize generated outcomes when embedding is employed.   

In particular, the user may not have enough memory to use the code 'out of the box', the user may have to run the stream 20 or 30 times before it avoids mode collapse, the user may be attempting to debug Tensorflow or Keras and is hindered by the never ending warning messages, the user may want to add embedding to a model and is unable to match dimensions, the stream may be interrupted six days into the process and be unable to start it from where it left off, the user may be using bad learning rates or slopes and be unable to recover from them, the user may stumble on to some bad Gan code 

#### Deliverables:
  1.  description of issues identified and resolved
  2.  code fragments illustrating the core of how the issue was resolved
  3.  complete cGan Python program

#### Cautions:
There are a numbers of perspective which I use coming from my background in analytics.  
  1.  stream:  the moving process of data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence occurs when there are no apparent improvements in clarity of images being generated.  It's worth noting that in some circumstances, we might define a valueable stream as one where continued processing always results in improved clarity of images.  
  3.  limited applicability:  the methods which I describe work for the limited set of data sets and cGan problems I have investigated.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss.  
  
#### Software and hardware requirements:
    - Python
        - Numpy
        - Tensorflow with Keras
        - Matplotlib
    - GPU is highly recommended
    - Operating system used:  Windows 10

##### The process:

 Using a cGAN as illustration, I provide partial solutions to the following questions:

  1.  is there an automatic way to recover from "mode collapse" when learning rates or slopes are reasonable?
  2.  is there a way to restart a cGAN which has not completed convergence?
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I use embedding when I have descriptions of pictures?
  6.  how can I vectorize from generated face to generated face?
  7.  what other changes can be applied?
        - adjusting optimization from Adam to Adamax for embedding
        - changing number of iterations due to memory issues
	- shutting off Tensorflow warning messages
        - adding label to the pictures

### 1.  is there an automatic way to recover from "mode collapse" when learning rates or slopes are reasonable?:
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
It is apparent there is a relationship between model loss and mode collapse.  The previous programming fragment illustrates an approach which often prevents a stream from mode collapse.  It depends on having captured disciminator weights, generator weights, and gan weights either during initialization or later in the process when all model losses are within bounds.  

### 2.  is there a way to restart a cGAN which has not completed convergence:
There is nothing quite as upsetting as running a stream using your GPUs and six days later the program bombs when it appears to be 90% complete.  Needless to say, your steam needs to be prepared for such an event.  Even with preparation, attempts to restart can result in endless warnings about parameters being not trainable, dimensions of weights being wrong for discriminate, generative, and gan models, and optimization values that make no sense.  There is a lot of helpful advice if you just want to inspect weights and optimization but after six days, you want to start where you left off - how do you do it.  It's important to note that cGAN will not properly restart unless you resolve the issues of what is trainable, what are the correct dimensions, and reasonable values  

Once issues with dimensions and what is trainable are resolved, there are then problems where models suffer from model collapse when attempts are made to restart the cGAN.  What happened?  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  As such, if you wish to continue executing the stream, rather than simply inspect weights, you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there only to make the discriminator and generator work together.  

Matters are made slightly more complicated if I want to be able to make the embedding layers fixed once training is complete but add other pictures to the training.    

### 3.  are there non-random initialization values that can be useful?
I have found no reason to believe that normal like distributions of random values are better than uniform distributions of random values.  I did a little bit of work on that issue and found that leptokurtic distributions were poorest in generating good images.  A supposed virtue of normal-like distributions is the values further away from the centroid are supposed to provide more information than those close to the centroid but do we really believe this when generating images?  For most of the results discussed here, we are in a bounded 100-dimensional space and there is no strong reason for fine tuning central values as opposed to values at the upper and lower tail.   
 
### 4.  how important is the source material (original pictures of faces)?
There is a well known acronym GIGO (garbage in, garbage out), and no one is surprised by words of advice to examine the data going into the stream.  When the data going into a stream is a derivative of another process, as in this case, it is important to examine the quality of the input data before declaring a process to be useful or invalid.  
 
### 5.  how can I use embedding when I have descriptions of pictures?
There are circumstances where we want to insure that the predicted output has particular characteristics, such as whether the face is attractive, what their gender is, and if they have high cheek bones, large lips, lots of hair, and other features.  At some point, it will be possible to create realistic GAN generated pictures of models wearing particular clothing, with specific expressions, and poses for catalogues.   
 
### 6.  how can I vectorize from generated face to generated face when using embedding?
Jeff Brownlee provides what I believe is a brilliant example of how to vectorize from one face to another face.  We vectorize two generated faces and, for the same 100-dimensional space, add embedding with four attributes:  0, no descriptor; 1 male; 2 high cheek bones; 3 large lips.    
 
### 7.  what other changes can be applied?
        - changing optimization from Adam to Adamax for embedding
        - changing number of iterations due to memory issues
	- shutting off Tensorflow warnings
        - adding label to the pictures
  
