import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
import matplotlib.pyplot as plt
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.models import load_model
from numpy.random import randn
import numpy as np



class DCGAN:
    
    def __init__(self,images,labels):
        self.latent_dim = 100
        self.images =images
        self.labels=labels
   
        
   
    def class_images(self):
        image0=[]
        image1=[]

        for i in self.labels:
           if i==0:
              image0.append(self.images[i])
           
           elif i==1:
               image1.append(self.images[i])
           
    
        return np.array(image0),np.array(image1)
    
    
    def define_discriminator(self,in_shape=(256,256,3)):
        model = Sequential()
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape)) 
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, (4,4), strides=(2,2), padding='same')) #32x32x256
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, (4,4), strides=(2,2), padding='same')) #32x32x128
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(512, (4,4), strides=(2,2), padding='same')) #16x16x128
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Conv2D(1024, (4,4), strides=(2,2), padding='same')) #16x16x128
        model.add(LeakyReLU(alpha=0.2))
    
        model.add(Conv2D(2048, (4,4), strides=(2,2), padding='same')) #16x16x128
        model.add(LeakyReLU(alpha=0.2))


        model.add(Flatten()) #shape of 8192
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid')) #shape of 1
	  # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
        
        
    
    
    def define_generator(self):
        
        model = Sequential()
        n_nodes = 2048*4*4#4096 nodes
        model.add(Dense(n_nodes, input_dim=self.latent_dim)) #Dense layer so we can work with 1D latent vector
   
        model.add(Reshape((4, 4, 2048)))  #8x8x128 dataset from the latent vector.

        model.add(Conv2DTranspose(1024, (4,4), strides=(2,2), padding='same')) #8x8
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same')) #16x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')) #32x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) #32x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')) #32x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')) #32x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (3,3), activation='tanh', padding='same')) #32x32x3
   
        return model
    
    
    def define_gan(self):
        d_model=self.define_discriminator(in_shape=(256,256,3))
        g_model=self.define_generator()
       
        d_model.trainable = False  #Discriminator is trained separately. So set to not trainable.
	     # connect generator and discriminator
        model = Sequential()
        model.add(g_model)
        model.add(d_model)
	  # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
    
      
    def generate_latent_points(self, n_samples):
        
        x_input = randn(self.latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input
   
    def generate_fake_samples(self, generator, n_samples):
        # generate points in the latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = generator.predict(x_input)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))  
        return X, y
    
    def generate_real_samples(self,dataset,n_samples):
        ix = randint(0, dataset.shape[0], n_samples)
	
        X = dataset[ix]

        y = ones((n_samples, 1)) ##Label=1 indicating they are real
        return X, y

   
    
    
    def train(self,dataset,n_epochs=1, n_batch=1, half_batch=1):
        bat_per_epo = int(dataset.shape[0] / n_batch)
        #half_batch = int(n_batch / 2)
       

    # Lists to store the losses for plotting
        d_loss_real_list, d_loss_fake_list, g_loss_list = [], [], []
        d_model=self.define_discriminator()
        g_model=self.define_generator()
        gan_model=self.define_gan()

    # manually enumerate epochs and batches
        for i in range(n_epochs):
           for j in range(bat_per_epo):
                X_real, y_real = self.generate_real_samples(dataset, half_batch)
                d_loss_real, _ = d_model.train_on_batch(X_real, y_real)
                X_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
                d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)
                X_gan = self.generate_latent_points(half_batch)
                y_gan = ones((n_batch, 1))
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

           
                d_loss_real_list.append(d_loss_real)
                d_loss_fake_list.append(d_loss_fake)
                g_loss_list.append(g_loss)

    # Plot the losses
        plt.plot(d_loss_real_list, label='Discriminator Real Loss')
        plt.plot(d_loss_fake_list, label='Discriminator Fake Loss')
        plt.plot(g_loss_list, label='Generator Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

    # Save the model after training
        g_model.save('g_image1_33epochs.h5')
        print("training completed succesfully")
        return g_model
        
        
    def generate_images(self,n_samples,model):
        latent_points = self.generate_latent_points(n_samples)  #Latent dim and n_samples
        g_img = model.predict(latent_points)
        return  g_img
        
    
        
        
 