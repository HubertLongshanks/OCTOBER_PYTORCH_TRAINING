# Hello and welcome

## to get started
click this link to download the dataset we will be training on (oxford pets), this is a built in benchmark in torchvision but the server it is served from is slow - so I predownloaded it and made it available in an S3-compatible bucket that is much faster :). 

https://fl-parcels.atl1.digitaloceanspaces.com/data.zip

Once you have the data, just unzip it and we can get started. 


Also, as the prior instructions suggested, please have a python installation and preferably an environment with torch and torchvision installed ready to go. If not you can use the requirmenets file in this repo. run:
```sh
pip install -r requirements.txt
```

to install all of what you will need for this session.

## you may notice
that there is already code in this repo, we will be going over (but not implementing due to time) a training/eval as well as a short test script. This code was largely created by the instructor using the AI coding agent Claude, but has been checked for accuracy and the instructor's intent of a simple demonstration of *how* something like this might look. A real-world pipeline would likely be more complicated, this is an example of a barebones case where we just want to more or less segment *foreground* from an image (in this case the foreground is pets).