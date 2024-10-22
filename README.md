# FRC Robot Detector with Tensorflow v1 

![example](https://github.com/Alex-Alexiev/robot_detector/blob/master/images/results/IMG_20191124_124626.jpg)

Using transfer learning on Google's pre-trained inference-graphs for fast FRC robot localization.

Tested for tensorflow-gpu==1.14
pip3 install tensorflow-gpu==1.14

Google image download tool: https://github.com/hardikvasa/google-images-download
labelImg: https://github.com/tzutalin/labelImg

# Setup
make sure you have pip3 install tensorflow-gpu==1.14

First clone https://github.com/tensorflow/models.git into a directory (i used /home/alexiev/dev)

Then install the protoc zip file from https://github.com/protocolbuffers/protobuf/releases/tag/v3.10.1
in downloads folder run:
sudo unzip -o protoc-3.7.1-linux-x86_64.zip -d /usr/local bin/protoc
sudo unzip -o protoc-3.7.1-linux-x86_64.zip -d /usr/local 'include/*'

inside tensorflow/models/research/ run:
sudo protoc --python_out=. object_detection/protos/*.proto

Inside the tensorflow/models/research folder run:
sudo python3 setup.py install

Then add the following to your PYTHONPATH

sudo gedit ~/.bashrc #open bashrc file
then add this new line to the end of your bashrc  file
export
PYTHONPATH=$PYTHONPATH=/home/alexiev/dev/tensorflow/models/research:/home/alexiev/dev/tensorflow/models/research/slim
then restart then bashrc to make it work
source ~/.bashrc
you can change the path to your own!

# Running

go inside the util folder in terminal

python3 get_data.py

then use the labelImg tool to draw bounding boxes on the images

python3 process_data.py

python3 train_model.py --logtostderr --model_dir=../training/ --pipeline_config_path=../training/ssd_mobilenet_v1_robot.config

to log data to tensorboard at localhost:6006 run
tensorboard --logdir=training

export the inference graph
python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path ../training/ssd_mobilenet_v1_robot.config --trained_checkpoint_prefix ../training/model.ckpt-XXXX --output_directory ../inference_graph

then run the detector
python3 detector.py




