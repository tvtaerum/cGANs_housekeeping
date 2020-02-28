# cGans_with_embedding
Python code for training cGans.  In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  Their code works and they deliver what they promise.  

Using a cGAN, I provide some answers to the following questions:
  1.  what is one way to recover from poor learning rates or slopes
  2.  is there a way to restart a cGAN which has not completed convergence
            - requirements for restarting
  3.  are there non-random initialization values that can be useful?
  4.  how important is the source material (original pictures of faces)?
  5.  how can I override warning messages from tensorflow?
  6.  how can I use embedding when I have descriptions of pictures?
  7.  how can I vectorize from generated face to generated face?

Motivation:
In my efforts to learn Generative Adversarial Networks, I got tired of so many publications about GAN where the author would present code and, with a wave of a hand, would declare it to be complete and working.  The code, as I discovered, was neither complete nor working and appeared to be in completion of class assignments.  These authors would, with considerable fanfare, present themselves as newly discovered experts in the field.  

Closely related is the situation where the code is obviously wrong and could not have produced the output shown in the published results.  Did they run out of time?   

The tutorials by Jason Brownlee are particularly helpful because they adhere to the recommendations of other GAN experts.  

With respect to restarting cGAN streams when there are interruptions, there are differences between loading optimization for the purpose of inspection and loading optimizations for the purpose of continuing an optimizattion.  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  Matters are made slightly more complicated if I want the embedding layer to be fixed in the discriminator model.  My experience is, if I want to continue training, I load in the discriminator and the generator models and then create a new instance of the gans model using the loaded discriminator and generator models rather than loading in the saved gan model.  At this point, everything works marvellously well.  If I attempt to use a loaded gan model, then the optimization quickly reaches a point where loss on the gan model goes to zero (g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)) and no improvements are made.    
