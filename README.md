# Release ML integration #

### Overview ###

This POC has been developed to display possibility to train and deploy models using Release solution where analytics stakeholders can use use config driven architecture to train and deploy models with minimal knowledge of machine learning.

### Demo structure ###

Solution consists of three shell scripts in deployment directory:

1. `build_docker.sh` - will build a local docker image that will be used for later stages. It accepts single argument `container_name`. Docker image can be tested by running the following commands:
* `train_local.sh`: Run this with the name of the image and it will run training on the local tree. For example, you can run `$ ./train_local.sh sagemaker-decision-trees`. It will generate a model under the `/test_dir/model` directory. You'll want to modify the directory `test_dir/input/data/...` to be set up with the correct channels and data for your algorithm. Also, you'll want to modify the file `input/config/hyperparameters.json` to have the hyperparameter settings that you want to test (as strings).
* `serve_local.sh`: Run this with the name of the image once you've trained the model and it should serve the model. For example, you can run  `$ ./serve_local.sh sagemaker-decision-trees`. It will run and wait for requests. Simply use the keyboard interrupt to stop it.,
* `predict.sh`: Run this with the name of a payload file and (optionally) the HTTP content type you want. The content type will default to `text/csv`. For example, you can run `$ ./predict.sh payload.csv text/csv`.

2. build_deploy_sagemaker.sh - will build and push docker image to Amazon ECR and will start a model training job. It accepts two arguments -  `container_name`, `data_dir` - location of train data that is to be used for training job and `out_file` - location of training logs file. Once the training job is completed it will create a training logs file that contains stdout of training job that includes model metrics and name of training job.

3. run_batch_prediction_demo.sh - will run a batch prediction job using the model that was trained in the previous step (will work only if the model was trained) and will store predictions in S3. 


### Demo instructions ###

1. `cd deployment`
2. `./build_docker.sh release-demo`

    1 `cd local_test`

    2 `./train_local release-demo` - will train a model in local docker
    
    3 `./serve_local release-demo` - will start a REST server in localhost:8080
    
    4 run `./predict.sh test_dir/input/data/test/test.csv text/csv` - will predict CSV payload
    
    5 run `./predict.sh test_dir/input/data/test/test.json application/json` - will predict JSON payload 
    
3. `./build_deploy_sagemaker.sh release-demo data training_output.txt` - will deploy Docker image to ECR and initiate training job. Output will be written to `training_output.txt` file. Output will contain model training metrics and name of the training job.
4. `cat training_output.txt` - will print metrics of the model training
5. `./run_batch_prediction_demo.sh output-for-release-demo data training_output.txt` - starts batch prediction job using dataset stored in data directory
6. `aws s3 sync s3://sagemaker-us-west-2-552551502186/output-for-release-demo-out predictions` - will download predictions of batch prediction job


### How do I get set up? ###

* Prerequisites
  * Docker
  * AWS command line tools
  * Python3.7 (catboost package requires python>=3.6) 
  
* Install pip packages from requirements.txt file


### Who do I talk to? ###

* laurynas.stasys@digital.ai
* ben.mackenzie@digital.ai

### Known limitations ###

1. BatchTransform jobs has limitation of file size. Single file cannot be larger than 6MB
2. Deploying model as SageMaker endpoints will not you allow to use curl/wget without AWS authentication header. Read more [here](https://aws.amazon.com/blogs/machine-learning/creating-a-machine-learning-powered-rest-api-with-amazon-api-gateway-mapping-templates-and-amazon-sagemaker/)
3. At this point model training parameters are part of docker image but could be used as model training hyperparameters. However, there's limitation though of how Sagemaker passes hyperparameters to docker (only top level dictionary items are parsed correctly) 
4. Most of the training job configs (i.e. S3 bucket for hosting models, training data and predictions, EC2 instance configs) are hardcoded
