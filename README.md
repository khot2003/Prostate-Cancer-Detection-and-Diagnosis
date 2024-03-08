
# Prostate cancer detection using Deep learning models

The goal of our project was to develop  a computer aided diagnosis system for prostate cancer
detection using multimodal diagnostic data .Since manually diagnosis of prostate
cancer is time consuming, which leads to delay in treatment hence to solve this
problem we developed the end to end system to assist the medical professional.
 
This system helps in detection of prostate cancer at an early stage that can increase
the chances of successful treatment.


## Deployment

To deploy this project we used framework called Flask 

  Flask: Facilitated the creation of a robust web server, enabling seamless communication between the frontend and the backend machine learning model
 


## Screenshots




## Documentation

### Introduction:
Prostate cancer is a malignant tumor that occurs in the male prostate. Prostate
cancer stands as the second most prevalent form of cancer affecting men in the
United States and claims the position of the primary cause of cancer-related fatalities in males aged 85 and older. Prostate cancer depends on age, family history,
and lifestyle. The prostate is found below the bladder, with the urethra passing through it. Manifesting symptoms often involve difficulties in urination, the
presence of blood in urine or semen, and pain experienced in the hips, back, and
pelvis. Early detection and diagnosis can effectively prevent Prostate cancer from
developing into advanced metastatic cancer and can improve the survival rate of
patients.

### Model Architecture
Gan Architecture: Gan consists of a generator and  a discriminator Both models undergo training. The generator takes latent noise vector and produces a batch of samples.This batch of samples along with the real examples
from the dataset, are presented to the discriminator. The discriminator classifies these samples as either real or fake.Subsequently, the discriminator is updated to enhance its ability to discriminate between real and fake samples with each iteration. Simultaneously, the generator keeps on  improving based on the discriminator’s feedback and is able to generate realistic fake samples.

### Working :
The generated Image smaples by Gan with original samples from dataset are give to VGG16 classifier and when Input data is given VGG16 classifier classifies and generate result as Normal or Cancerous 







### Conculsion
In addressing the challenge of improving prostate cancer detection and diagnosis, our solution revolves around the development of a computer-aided diagnostics system using multimodel dataset that encompasses diverse patient information, including medical histories and scans. Employing generative adversarial neural networks (GANs), we introduced synthetic images to improve the model’s learning, enhancing its ability to learn deep representations. This was followed by performing  the task of binary classification by fine tuned VGG16 classifier for detection of Prostate Cancer.


## Demo
![fornt](https://github.com/khot2003/Prostate-cancer-Detection/assets/105428024/b4238b3a-0713-4ecc-aedd-573553b43e93)




