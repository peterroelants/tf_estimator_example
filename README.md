# Tensorflow Estimator example

Example scripts on how to use Tensorflow's [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) class.

This repository as an accompanying blogpost at https://medium.com/@peter.roelants/tensorflow-estimator-dataset-apis-caeb71e6e196

The main file of interest will be [`srs/mnist_estimator.py`](https://github.com/peterroelants/tf_estimator_example/blob/master/src/mnist_estimator.py), which defines an example Estimator to train an network on mnist.


## Setup environment

With [Anaconda](https://www.anaconda.com/download/) Python:

```
conda env create -f env.yml
source activate tensorflow
```

## Training

### Training locally

After setting up the environment you can run the training locally with:
```
./src/mnist_estimator.py
```

Training can be monitored with [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard):
```
tensorboard --logdir=./mnist_training
```

After training you can check the inference with:
```
./src/mnist_inference.py
```

### Training on Google Cloud

1. Create a [new project in the [cloud resource manager](https://console.cloud.google.com/cloud-resource-manager) as described [here](https://cloud.google.com/resource-manager/docs/creating-managing-projects). (I named my project `mnist-estimator`)
2. [Install](https://cloud.google.com/sdk/downloads) the Google Cloud SDK
3. [Enable](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com) the ML Engine APIs.
4. Set up a Google Cloud Storage (GCS) bucket as described [here](https://cloud.google.com/storage/docs/creating-buckets). This will be needed to save our model checkpoints. I named my bucket `estimator-data`.

Run the training job on Google Cloud with:
```
gcloud ml-engine jobs submit training mnist_estimator_`date +%s` \
    --project mnist-estimator \
    --runtime-version 1.8 \
    --python-version 3.5 \
    --job-dir gs://estimator-data/train \
    --scale-tier BASIC \
    --region europe-west1 \
    --module-name src.mnist_estimator \
    --package-path src/ \
    -- \
    --train-steps 6000 \
    --batch-size 128
```

Note:
* Replace `gs://estimator-data/` with the link to the bucket you created.
* Latest Python supported on gcloud is 3.5 (although I'm using 3.6 locally)
* The `--project` flag will refer to the gcloud project (`mnist-estimator` in my case). To avoid using this flag you can set the default project in this case with `gcloud config set core/project mnist-estimator`.
* You can feed in arguments to the script by adding an empty `--` after the gcloud parameters and adding your custom arguments after, like `train-steps` and `batch-size` in this case.
* Note that the `job-dir` argument will be fed into the arguments of `mnist_estimator`. This script should thus always accept this parameter.


You can follow the training with tensorboard by:
```
tensorboard --logdir=gs://estimator-data/train
```

However, tensorboard seems to update very slowly when connected to a gcloud bucket. Sometimes it didn't even want to display all data.

After training you can download the checkpoint files from the gcloud bucket.


## More info

There is a Google Cloud [blogpost](https://cloud.google.com/blog/big-data/2018/02/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine) going into more detail on training an estimator in the cloud if you're interested.
