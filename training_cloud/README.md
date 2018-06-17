# Training on the Cloud

In order to train your tensor flow model on the cloud
there are a few prerequisites.  

1. Go through the tutorial on [training locally](/training_locally) and complete everything except anything related to TensorFlow i.e don't download TensorFlow Models, don't create the TF records, don't start training and don't export the inference graph. That will all be handled on the cloud. You should have a directory that looks like this.

```
images (folder)
    class1 (folder)
        class1_1.jpg
        class1_2.jpg
        ...
    class2 (folder)
        ...

annotations (folder)
    class1_2.xml
    ...
    class2_1.xml
    ...

annotation.pbtext
```

2. Get a Google account if you don't already have one. Open up google drive and create a folder called `tf_training`. Copy all the folders listed above as well as the [tf_cloud_training.ipynb](/training_cloud/tf_cloud_training.ipynb) into `tf_training`. 

3. Open up Google Drive and click `NEW`->`More`->`+Connect more apps`. 
In the searchbar type `colab` and connect it.

4. Back in your `tf_training` folder on Drive, double click the tf_cloud_training.ipynb file and click `Colaboratory` under Connected Apps.

5. In the top menu, click `Runtime`-> `Change runtime type` and set Runtime type to `Python 2` and Hardware Accelerator to `GPU`. Save these selections. In the top right hand corner of the menu bar will be the button that says `Connect`. Click button and wait till you are connected to a GPU.

6. In each block of code starting from the top, hit the small plat button which will execute that code. Once this has finished proceed on with the next section.

7. When you get to the 2nd or 3rd block, you will need to give the SDK access to your drive. Click the link provided, click accept, copy the large key provided and past that into the field back in Colaboratory.

8. Continue on till training has finished and the model is saved back into your Google drive folder. 
