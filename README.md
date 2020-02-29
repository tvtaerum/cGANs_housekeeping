## cGans_with_embedding - housekeeping
### Housekeeping python code for training and utilizing cGans with embedding.  
In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  In contrast to many other projects, their code works 'out of the box' and they deliver what they promise.  

#### Motivation:

Even the best tutorials can leave a person scratching their head wondering if there are ways to make "minor" changes to the stream.  In particular, the user might discover there are no obvious solutions to bad initial randomized values, no obvious way to restart streams when convergence is not complete, figuring out why outcomes appear dirty or messy, warning messages that suddenly show up and cannot be turned off, and no obvious ways to vectorize generated outcomes when embedding is employed.   

 Using a cGAN, I provide some partial solutions to the following questions:

  1.  is there an automated way to recover from bad starts when learning rates or slopes are reasonable?
  2.  is there a way to restart a cGAN which has not completed convergence?
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I override warning messages from tensorflow?
  6.  how can I use embedding when I have descriptions of pictures?
  7.  how can I vectorize from generated face to generated face?

### 1.  what is one way to recover from poor learning rates and/or slopes:

As many experts in GAN will point out, setting learning rates and slopes are an art as much as a science.  
 
### 2.  is there a way to restart a cGAN which has not completed convergence:

There is nothing quite as upsetting as running a stream on your GPUs and two days later the program bombs when it appears to be 90% complete.  Attempts to restart end in tragedy as there are endless warnings about parameters being not trainable and dimensions of weights being different for discriminate, generative, and gan models.  There is lots of helpful advice available if you just want to inspect weights and optimization but you want to start where you left off.  As such, the cGAN will not properly restart unless you actually resolve the issues of what is trainable when and insure the dimensions of your model are correct.

Once issues with dimensions and what is trainable is resolved, there are then problems where models which were happily moving towards convergence show losses going to zero or ridiculously high values.  What happened?  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  As such, if you wish to continue executing the stream, rather than simply inspect weights, you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there simply to make the discriminator and generator work together.  

Matters are made slightly more complicated if I want to be able to make the embedding layers fixed once training is complete but add other pictures to the training.    

### 3.  are there non-random initialization values that can be useful?
I have found no reason to believe that normal like distributions of random values are better than, for instance, uniform distributions of random values...  We can imagine we are in a bounded 100-dimensional space and there is no strong reason for fine tuning central values as opposed to values at the upper and lower tail.   
 
### 4.  how important is the source material (original pictures of faces)?
There is
 
### 5.  how can I override warning messages from tensorflow?
There is
 
### 6.  how can I use embedding when I have descriptions of pictures?
There is
 
### 7.  how can I vectorize from generated face to generated face when using embedding?
There is 
 
