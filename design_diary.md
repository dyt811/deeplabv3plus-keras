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
 
 2019-09-20T013853EST
 Apparently, DeepLabV3+ have undifferentiable layers and hence results in F1 score type score cannot be calculated for some reason. Super weird. 
 
 MAE and SSIM straight all does not produce enough class balance and lead to strictly BLACK images due to class imbalance. Something needs to change. 

2019-09-21T213258EST
Substantially improved the logging and workflow routines include model parameters documentation generation at run time.  

2019-09-24T201057EST
Time to repurpose this for Severstal competition. 
I have updated the CSV parser to export everything into 3D binary mask NPY files instead of 2D gray scale images. We will see how this looks. I am still not 100% convinced it converted properly but whatever. First step to go.   

2019-09-30T203015EST
Rewrite the part to parse ground truth into 4 channel binary masks are now in the process of saving them. Also wrote a function to actually parse the numpy array into the gray scale images. 

2019-10-07T185538EST
After reviewing last year's prediction, I see that that the results are not super optimal. Prediction outcome were actually around 0.4 in F1 score. 

Now that we have a network more or less working, we need to build the reverise IO to actually get the submission part working so we can get some score on the private
Kaggle competition board. 

Two major goals of today: reverse IO, and maybe add a secondary balance weighted binary cross entropy function for better loss alternative than MSE.

2019-10-09T080743EST
Got the additional lossfn working. Time to tackle the reverse IO isssue for CSV submission for ServalStal

2019-10-14T073649EST
Reverse IO function has been adopted. However, it has not been tested to see how well it reconstitute the label CSV. I need to validate it against the existing training images to see if anything changes. 

Another thing I want to clarify is the reuse of the data and whether the shuffling is working. 

2019-10-21T210159EST
I have been away for a while and losing momentum on this. :(

We now have a models at least. Making shitty prediction but I should at least try to submit something.

2019-10-22T214816EST
Just completed the prediction IO to CSV in my current routine.  
Cross check of same images encoding decoding check resulted in different things... Fffffff...T
This is very concerning as this means there is a bug in the encoder...

2019-10-22T235150EST
Further evaluation suggests there was 257 unites of displaycement, i.e. 1 column + 1 cell. Off by one in BOTH dimensions. This likely occurred early as the RLE encoding part is very precise and unlikely to be errored. 