## Full Stack Recommendation Engine 

Sample code repo that forms the base of the following tutorial:
* [Unlock the power of Unstructured Data: From Embeddings to In Context Learning – Build a Full stack Q&A Chatbot with Langchain, and LLM Models on Sagemaker](https://buildon.aws/tutorials/fullstack-llm-langchain-chatbot-on-aws)
* Detailed setup and steps are provided in the tutorial.
* Explanation of the folders that you see in this repo
    * create-embeddings-save-in-vectordb folder
        * This folder contains the code for the ingestion and processing pipeline
        * 
    * data folder
        * Contains  the python notebook, the raw csv movie files , kmeans output from previous iterations to save time and the localui folder has the User Interface for our fancy MyFlix UI
    * sagemaker-migration-toolkit
        * Utility code to make deployment of  custom scaling model on sagemaker easier
    * apis_for_sagemaker_models
        * There are 2 folders containing the snippets of code for creating our REST API using the [Chalice framework](https://github.com/aws/chalice)




## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

