# cGans_with_embedding - housekeeping
Housekeeping python code for training cGans.  In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  In contrast to many other projects, their code works 'out of the box' and they deliver what they promise.  

Motivation:
Even the best tutorials can leave a person scratching their head wondering if there are ways to 
Are you tired of publications about GAN where authors present code and, with a wave of a hand, declare it to be complete and working, when clearly it is not?  Closely related is the situation where the code is obviously wrong and could not have produced the output shown in the published results.  In contrast, I found the tutorials by Jason Brownlee and also Jeff Heaton to be particularly helpful because they adhere to the recommendations of other GAN experts and, if they don't work 'out of the box', a few minor tweeks will deliver wonderful solutions.  

Occasionally, even the best tutorials can leave a person scratching their head wondering if there is a solution to a number of housekeeping issues.  Using a cGAN, I provide some partial solutions to the following questions:

  1.  is there an automated way to recover from bad starts when learning rates or slopes are reasonable?
  2.  is there a way to restart a cGAN which has not completed convergence?
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I override warning messages from tensorflow?
  6.  how can I use embedding when I have descriptions of pictures?
  7.  how can I vectorize from generated face to generated face?

Even the best tutorials 

1.  what is one way to recover from poor learning rates and/or slopes:

2.  is there a way to restart a cGAN whiich has not completed convergence:
      There is nothing quite as upsetting as running a stream on your GPUs and having the program bomb when it is 90% of the way through.  Attempts to restart end in tragedy as there are warnings about 
      
      a. getting rid of the ???  warning
      
      b. getting rid of the ????  warning
      
      c. getting rid of the lack of convergence problem after restarting.  
With respect to restarting cGAN streams when there are interruptions, there are differences between loading optimization for the purpose of inspection and loading optimizations for the purpose of continuing an optimizattion.  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  Matters are made slightly more complicated if I want the embedding layer to be fixed in the discriminator model.  My experience is, if I want to continue training, I load in the discriminator and the generator models and then create a new instance of the gans model using the loaded discriminator and generator models rather than loading in the saved gan model.  At this point, everything works marvellously well.  If I attempt to use a loaded gan model, then the optimization quickly reaches a point where loss on the gan model goes to zero (g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)) and no improvements are made.    

3.  are there non-random initialization values that can be useful?
4.  how important is the source material (original pictures of faces)?
5.  how can I override warning messages from tensorflow?
6.  how can I use embedding when I have descriptions of pictures?
7.  how can I vectorize from generated face to generated face?
