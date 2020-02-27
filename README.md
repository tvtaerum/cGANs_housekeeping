# cGans_with_embedding
Python code for training cGans.  In particular I thank Jason Brownlee and also Jeff Heaton - their tutorials on the Internet are brilliant.  Their code works and they deliver what they promise.  

I demonstrate a cGAN with the following characteristics:
  1.  a method to automatically recover from poor initialization
  2.  ways to recover from incomplete convergence
  3.  the use of non-random initialization values
  4.  testing source material (original pictures of faces)
  5.  how to override warning messages
  6.  changes to setting learning rates and slopes

Motivation:
In my efforts to learn Generative Adversarial Networks, I got tired of so many publications about GAN where the author would present code and, with a wave of a hand, would declare it to be complete and working.  The code, as I discovered, was neither complete nor working and appeared to be in completion of class assignments.  These authors would, with considerable fanfare, present themselves as newly discovered experts in the field.  

Closely related is the situation where the code is obviously wrong and could not have produced the output shown in the published results.  Did they run out of time?   

The tutorials by Jason Brownlee are particularly helpful because they adhere to the recommendations of other GAN experts.  

I'd just like to add an additional comment that an issue I've had is loading in the optimization information for cGAN models with embedding for the purpose of improving the results by carrying out more iterations.  This is slightly different than loading in the optimization information for inspection or prediction.  As was pointed out, the discriminator and generator models are components of the gans model and trainable flags have to be reset when loading and saving the discriminator model.  Matters are made slightly more complicated if I want the embedding layer to be fixed in the discriminator model.  My experience is, if I want to continue training, I load in the discriminator and the generator models and then create a new instance of the gans model using the loaded discriminator and generator models rather than loading in the saved gan model.  At this point, everything works marvellously well.  If I attempt to use a loaded gan model, then the optimization quickly reaches a point where loss on the gan model goes to zero (g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)) and no improvements are made.    
