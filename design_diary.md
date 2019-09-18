2019-08-20T194012EST
The next kaggle competition require semantic classification. 

For this, based on recent performance, either Unet or adapted DeepLabV3+ might be a good starting point. 

For this reason, I am going to adapt from the existing keras implementation to start and tweak it to train on my data first before updating or benchmarking other things, etc.

2019-08-20T203424EST
Was reading the excellent illustration of the depthwise separable convolution and just realized maybe that is an area of improvmeent too: https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

using modified Depthwise Separable Convolution instead. 

I already made leakRELU update because I believe that should in theory help gradient flow but I gotta put that input practise to be honest. 

2019-09-10T181849EST
Been a few weeks since I was able to work on this. Time to put the DeepLabV3+ to test by actually putting the network to work running on real world data. 

Today's goal is to get the basic Severstal example working namely in terms of IO, as this will benefit with my interview process as well. 

2019-09-10T202306EST
Training images, there are about 10K images. Pretty good fun time. Going to try to abel them based on the defect type color etc. 
 
 2019-09-10T215137EST
 Had a really good discussion at the meetup meetings and these highlighted the further area actions that I need to explore more. 
 
 