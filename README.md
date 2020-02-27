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
