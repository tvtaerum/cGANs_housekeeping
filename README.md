## cGans_with_embedding - housekeeping
### Housekeeping python code for training and utilizing cGans with embedding.  
In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  In contrast to many other projects, their code works 'out of the box' and they deliver what they promise.  In particular, the stream (Python program and process) is a derivative of tutorials by Jason Brownlee and insights on embeddings by Jeff Heaton.  The subject matter are faces derived by a process from...  

#### Motivation for housekeeping:

Even the best tutorials can leave a person scratching their head wondering if there are ways to make "minor" changes to the stream.  In particular, the user might discover there are no obvious solutions to bad initial randomized values, no obvious way to restart streams when convergence is not complete, no obvious way for figuring out why outcomes appear dirty or not clear, warning messages that suddenly show up and cannot be turned off, and no obvious ways to vectorize generated outcomes when embedding is employed.   

#### Cautions:

I define a couple of terms which reflect my background in analytics.  
  1.  stream:  the moving process including input of data, algorithms used, and the output of data and its evaluation.
  2.  convergence:  Since there are no unique solutions in GAN, convergence occurs when there are no apparent improvements in clarity of images being generated.  In some circumstances, good models might be characteristic of streams for which continued processing always results in improved clarity of images.  

##### The process:

 Using a cGAN, I provide some partial solutions to the following questions:

  1.  is there an automated way to recover from bad starts when learning rates or slopes are reasonable?
  2.  is there a way to restart a cGAN which has not completed convergence?
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I override warning messages from tensorflow?
  6.  how can I use embedding when I have descriptions of pictures?
  7.  how can I vectorize from generated face to generated face?
  8.  how can I add additional information to a generated face?

### 1.  what is one way to recover from poor learning rates and/or slopes:

As many experts in GAN will point out, setting learning rates and slopes are an art as much as a science.  The stream provides one way of giving intial estimates second and third opportunities in its slide towards convergence.  While I can provide no theoretical basis for the process, it works more often than not.  
 
### 2.  is there a way to restart a cGAN which has not completed convergence:

There is nothing quite as upsetting as running a stream using your GPUs and two days later the program bombs when it appears to be 90% complete.  Attempts to restart end in tragedy as there are endless warnings about parameters being not trainable and dimensions of weights being different for discriminate, generative, and gan models.  There is lots of helpful advice available if you just want to inspect weights and optimization but you want to start where you left off.  As such, the cGAN will not properly restart unless you actually resolve the issues of what is trainable and insure the dimensions of your model are correct.

Once issues with dimensions and what is trainable are resolved, there are then problems where models suffer from model collapse when attempts are made to restart the cGAN.  What happened?  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  As such, if you wish to continue executing the stream, rather than simply inspect weights, you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there only to make the discriminator and generator work together.  

Matters are made slightly more complicated if I want to be able to make the embedding layers fixed once training is complete but add other pictures to the training.    

### 3.  are there non-random initialization values that can be useful?
I have found no reason to believe that normal like distributions of random values are better than uniform distributions of random values.  I did a little bit of work on that issue and found that leptokurtic distributions were poorest in generating good images.  A supposed virtue of normal-like distributions is the values further away from the centroid are supposed to provide more information than those close to the centroid but do we really believe this when generating images?  For most of the results discussed here, we are in a bounded 100-dimensional space and there is no strong reason for fine tuning central values as opposed to values at the upper and lower tail.   
 
### 4.  how important is the source material (original pictures of faces)?
There is a well known acronym GIGO (garbage in, garbage out), and no one is surprised by words of advice to examine the data going into the stream.  When the data going into a stream is a derivative of another process, as in this case, it is important to examine the quality of the input data before declaring a process to be useful or invalid.  
 
### 5.  how can I override warning messages from tensorflow?
When debugging keras in tensorflow, it is occasionally not helpful to have tensorflow warning messages repeatedly occur.  Turning off the warning messages turned out, for me, to be surprisingly difficult.   
 
### 6.  how can I use embedding when I have descriptions of pictures?
There are circumstances where we want to insure that the predicted output has particular characteristics, such as whether the face is attractive, what their gender is, and if they have high cheek bones, large lips, lots of hair, and other features.  At some point, it will be possible to create realistic GAN generated pictures of models wearing particular clothing, with specific expressions, and poses for catalogues.   
 
### 7.  how can I vectorize from generated face to generated face when using embedding?
Jeff Brownlee provides what I believe is a brilliant example of how to vectorize from one face to another face.  We vectorize two generated faces and, for the same 100-dimensional space, add embedding with four attributes:  0, no descriptor; 1 male; 2 high cheek bones; 3 large lips.    
 
### 8.  how can I add descriptive information to a generated face when using embedding?
All we are doing is taking a generated face and putting a label on the picture.    
