## Full Stack LLM Powered Question and Answer Chatbot.

Sample code repo that forms the base of the following tutorial:
* [Unlock the power of Unstructured Data: From Embeddings to In Context Learning – Build a Full stack Q&A Chatbot with Langchain, and LLM Models on Sagemaker](https://community.aws/tutorials/fullstack-llm-langchain-chatbot-on-aws)
* Detailed setup and steps are provided in the tutorial.
* Explanation of the folders that you see in this repo
    * create-embeddings-save-in-vectordb/ folder
        * This folder has the code for the ingestion and processing pipeline for converting the car manual document into embeddings and storing the embeddings into AWS OpenSearch
        * We are using AWS OpenSearch as a Vector Database for storing the embeddings.
        * startup_script.py contains the code that will invoke the hugging face embeddings model endpoint that is deployed on sagemaker for the car manual document and will insert the emebeddings into the Vector Database.
        * After using the Dockerfile for building the container upload the image into the AWS Elastic container Registry (ECR) in your AWS Account.
    * RAG-langchain-questionanswer-t5-llm/ folder 
        * This folder has the code for building the API endpoint which will respond back to the car related questions sent to it. 
        * This API serves as the backend intelligence to our car savvy AI Assistant and invokes the deployed T5-Flan LLM endpoint.
        * Build the Dockerfile in this folder and push the image to AWS Elastic Container Registry (ECR).
    * homegrown-website-and-bot/ folder
        * Code for Website and chatbot along with Dockerfile 
        * Build the Dockerfile in this folder and push the image to AWS Elastic Container Registry (ECR).
    * Infrastructure
        * Create the Cloudformation stack (opensearch-vectordb.yaml) to build the Amazon OpenSearch Cluster.   
        * After the 3 docker container images from  create-embeddings-save-in-vectordb/ , RAG-langchain-questionanswer-t5-llm/ and homegrown-website-and-bot/  are pushed to Amazon ECR , use the 3 Cloudformation templates mentioned below from this folder.
        * Create the Cloudformation stack (fargate-embeddings-vectordb-save.yaml) to build the Fargate task that will create embeddings and store into the vector database.
        * Create the Cloudformation stack (fargate-api-rag-llm-langchain.yaml) to build the the ECS cluster for the API.
        * Create the Cloudformation stack (fargate-website-chatbot.yaml) to build the the ECS cluster for the website with embedded chatbot.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

