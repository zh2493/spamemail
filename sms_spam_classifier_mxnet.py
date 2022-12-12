#!/usr/bin/env python
# coding: utf-8

# <h1>SMS Spam Classifier</h1>
# <br />
# This notebook shows how to implement a basic spam classifier for SMS messages using Apache MXNet as deep learning framework.
# The idea is to use the SMS spam collection dataset available at <a href="https://archive.ics.uci.edu/ml/datasets/sms+spam+collection">https://archive.ics.uci.edu/ml/datasets/sms+spam+collection</a> to train and deploy a neural network model by leveraging on the built-in open-source container for Apache MXNet available in Amazon SageMaker.

# Let's get started by setting some configuration variables and getting the Amazon SageMaker session and the current execution role, using the Amazon SageMaker high-level SDK for Python.

# In[1]:


from sagemaker import get_execution_role

bucket_name = 'smlambda-workshop-6998lh'

role = get_execution_role()
bucket_key_prefix = 'sms-spam-classifier'
vocabulary_length = 9013

print(role)


# We now download the spam collection dataset, unzip it and read the first 10 rows.

# In[2]:


get_ipython().system('mkdir -p dataset')
get_ipython().system('curl https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip -o dataset/smsspamcollection.zip')
get_ipython().system('unzip -o dataset/smsspamcollection.zip -d dataset')
get_ipython().system('head -10 dataset/SMSSpamCollection')


# We now load the dataset into a Pandas dataframe and execute some data preparation.
# More specifically we have to:
# <ul>
#     <li>replace the target column values (ham/spam) with numeric values (0/1)</li>
#     <li>tokenize the sms messages and encode based on word counts</li>
#     <li>split into train and test sets</li>
#     <li>upload to a S3 bucket for training</li>
# </ul>

# In[3]:


import pandas as pd
import numpy as np
import pickle
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

df = pd.read_csv('dataset/SMSSpamCollection', sep='\t', header=None)
df[df.columns[0]] = df[df.columns[0]].map({'ham': 0, 'spam': 1})

targets = df[df.columns[0]].values
messages = df[df.columns[1]].values

# one hot encoding for each SMS message
one_hot_data = one_hot_encode(messages, vocabulary_length)
encoded_messages = vectorize_sequences(one_hot_data, vocabulary_length)

df2 = pd.DataFrame(encoded_messages)
df2.insert(0, 'spam', targets)

# Split into training and validation sets (80%/20% split)
split_index = int(np.ceil(df.shape[0] * 0.8))
train_set = df2[:split_index]
val_set = df2[split_index:]

train_set.to_csv('dataset/sms_train_set.gz', header=False, index=False, compression='gzip')
val_set.to_csv('dataset/sms_val_set.gz', header=False, index=False, compression='gzip')


# We have to upload the two files back to Amazon S3 in order to be accessed by the Amazon SageMaker training cluster.

# In[4]:


import boto3

s3 = boto3.resource('s3')
target_bucket = s3.Bucket(bucket_name)

with open('dataset/sms_train_set.gz', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/train/sms_train_set.gz'.format(bucket_key_prefix))
    
with open('dataset/sms_val_set.gz', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/val/sms_val_set.gz'.format(bucket_key_prefix))


# <h2>Training the model with MXNet</h2>
# 
# We are now ready to run the training using the Amazon SageMaker MXNet built-in container. First let's have a look at the script defining our neural network.

# In[5]:


get_ipython().system("cat 'sms_spam_classifier_mxnet_script.py'")


# We are now ready to run the training using the MXNet estimator object of the SageMaker Python SDK.

# In[6]:


from sagemaker.mxnet import MXNet

output_path = 's3://{0}/{1}/output'.format(bucket_name, bucket_key_prefix)
code_location = 's3://{0}/{1}/code'.format(bucket_name, bucket_key_prefix)

m = MXNet('sms_spam_classifier_mxnet_script.py',
          role=role,
          train_instance_count=1,
          train_instance_type='ml.c5.2xlarge',
          output_path=output_path,
          base_job_name='sms-spam-classifier-mxnet',
          framework_version='1.2',
          py_version = 'py3',
          code_location = code_location,
          hyperparameters={'batch_size': 100,
                         'epochs': 20,
                         'learning_rate': 0.01})

inputs = {'train': 's3://{0}/{1}/train/'.format(bucket_name, bucket_key_prefix),
 'val': 's3://{0}/{1}/val/'.format(bucket_name, bucket_key_prefix)}

m.fit(inputs)


# <h3><span style="color:red">THE FOLLOWING STEPS ARE NOT MANDATORY IF YOU PLAN TO DEPLOY TO AWS LAMBDA AND ARE INCLUDED IN THIS NOTEBOOK FOR EDUCATIONAL PURPOSES.</span></h3>

# <h2>Deploying the model</h2>
# 
# Let's deploy the trained model to a real-time inference endpoint fully-managed by Amazon SageMaker.

# In[7]:


mxnet_pred = m.deploy(initial_instance_count=1,
                      instance_type='ml.m5.large')

# import boto3

# # create model within sagemaker
# model = m.create_model()
# session = model.sagemaker_session

# container_def = model.prepare_container_def(instance_type='ml.m4.xlarge')
# role = model.role
# model_name = model.name

# session.create_model(model_name, role, container_def)

# # create endpoint config
# endpoint_config_name = session.create_endpoint_config(name=model_name,
#                                                       model_name=model_name,
#                                                       initial_instance_count=1,
#                                                       instance_type='ml.m4.xlarge')

# # check endpoint
# endpoint_name = 'sms-spam-classifier-mxnet'
# endpoint_is_created = False

# client = boto3.client('sagemaker')
# response = client.list_endpoints()

# for i in response['Endpoints']:
#     if endpoint_name == i['EndpointName']:
#         endpoint_is_created = True
#         break

# # if no endpoint has been created, create one
# if not endpoint_is_created:
#     created_endpoint_name = session.create_endpoint(endpoint_name=endpoint_name,
#                                                     config_name=endpoint_config_name)
    
#     print('\nCreated Endpoint name: ', created_endpoint_name)
    
# # otherwise, update the created endpoint with new Endpoint Config
# else:
#     updated_endpoint_name = session.update_endpoint(endpoint_name=endpoint_name,
#                                                     endpoint_config_name=endpoint_config_name)
    
#     print('\nUpdated Endpoint name: ', updated_endpoint_name)


# <h2>Executing Inferences</h2>
# 
# Now, we can invoke the Amazon SageMaker real-time endpoint to execute some inferences, by providing SMS messages and getting the predicted label (SPAM = 1, HAM = 0) and the related probability.

# In[8]:


from sagemaker.mxnet.model import MXNetPredictor
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

# Uncomment the following line to connect to an existing endpoint.
# mxnet_pred = MXNetPredictor('<endpoint_name>')

test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

result = mxnet_pred.predict(encoded_test_messages)
print(result)


# <h2>Cleaning-up</h2>
# 
# When done, we can delete the Amazon SageMaker real-time inference endpoint.

# In[11]:


mxnet_pred.delete_endpoint()


# In[10]:





# In[ ]:





# In[ ]:




