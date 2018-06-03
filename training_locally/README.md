# Training an Object Detection Model Locally

## Prerequisites

TensorFlow should be installed correctly on your system

You will need Python 2 if you are on linux or Python 3 on windows. PIP is also required.

## Image Gathering

To train a model using TensorFlow's object detection API, we first need to gather enough input data to create the model. 
A good model for our purposes needs > 100 images of medium (300 x 300) quality. The images should be taken at multiple angles and distances and with varying surroundings. It is prefereable to have the images in JPEG format.

Create a folder called `images` and `annotations`. Inside the images folder, for each class, create a new folder and place all of their images in there. The structure of your folder should like like this. It is advisable to place these folders into this current directory

```
-images
    -class_a
        -*.jpg
-annotations
```

## Annotation

Once the images have been collected it is time to annotate the images. clone our version of labelimg [here](https://github.com/uvic-auvic/labelimg) and run it with Python 3. You may need to install a few dependencies such as PyQt5 and lxml. Once that is working make sure that the output format is set to Pascal VOC. 

Make sure that ALL of the xmls are being saved into the `annotations` folder

## Downloading TensorFlow-Models

TensorFlow models is the research project which brings us the object detection API. Clone their repo [here](https://github.com/tensorflow/models). Add its location to the PYTHONPATH. 

On windows that can be done with this command

`set PYTHONPATH="C:\path\to\models;C:\path\to\models\research;C:\path\to\models\research\slim"`

On Linux

`export PYTHONPATH=$PYTHONPATH:/path/to/models:/path/to/models/researc:/path/to/models/research/slim:`

It is important to note that both the models directory, the research directory and slim directory all need to be added to the path

Once that has been added to path, we need to compile the protocol buffer classes.

Install the protocol buffer application from the instructions [here](https://github.com/google/protobuf) and enter the research directory of tensorflow/models/

Assuming, protoc has been added to your path, run the command 
`protoc --python_out=. object_detection/protos/*.proto`

This command may not work on windows, because of globbing, and you may have to list out every .proto file in that directory.

## Creating the TF records

After all the images have been annotated, create a file called `annotation.pbtext`. In this file you should list all the classes you want to build your model around. The structure should look like this

```
item {
    id: 1
    name: 'buoy'
}

item {
    id: 2
    name: 'gate'
}
```

You should also modify the generate_tf_record.py file to return the same values as is listed in the `class_text__to_int` function.

Now all that is left to be done is to run the script `generate_tf_record.py`. This will generate the binary record file that we can work with. If all goes well you should see a `train.record` and `val.record` in your directory.