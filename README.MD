# What the project is about?
This project enables you to transform your codebase—whether open source or proprietary—into a vector database. Simply specify the root directory of your codebase and the destination for the generated RAG database.

Once created, the vector database leverages Retrieval-Augmented Generation (RAG) techniques to provide valuable insights into the codebase. This can be particularly useful for new contributors seeking to understand the project's structure, onboarding new team members, or even analyzing bug reports to identify potential root causes efficiently.

# How to start?
To start using the tools, you need OpenAI API, then follow the [Setup](SETUP.MD) guide to get your environment ready.

# Features

### Creating and Loading Embedding
![image_doc](/doc/doc3.png)

If you haven't have an embedding, you can run the embedding process first by specifying the parent directory of the code base, and the directory where the vector embedding will be stored.

If you already have an existing vector embedding, you can load it right away by providing the path to the existing embedding database.

### Querying About the Codebase
![image_doc2](/doc/doc1.png)

Once the embedding is created / loaded, you're ready to query the LLM about the project. For every query, some context will be given to the LLM behind the scene, which will be displayed on the right hand panel.