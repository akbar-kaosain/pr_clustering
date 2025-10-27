# Pull Request Clustering Using Text Similarity

📄 A detailed project report is available at [https://kaosain.com/calgary/](https://kaosain.com/calgary/)

### Overview
This project groups similar pull requests (PRs) from a GitHub repository based on their text content. Each pull request has a title and description written by developers. By using AI-based text embeddings and clustering methods, the project identifies PRs that are semantically related such as those fixing similar bugs or improving the same parts of the system.

### Goal
- Collect merged pull requests from a GitHub repository using the GitHub API.
- Generate text embeddings using the SentenceTransformer model.
- Apply Agglomerative Clustering to group similar pull requests.
- Save and analyze cluster results for better understanding of project activities.

### Dataset
The dataset was collected directly from the GitHub repository **wesnoth/wesnoth** using the GitHub API.
Each PR record includes:
- PR number
- Title and description
- Labels
- URL

The data is saved in two formats:
- `wesnoth_pr_data.json` — full details of each PR
- `wesnoth_index.csv` — summarized dataset for clustering

### Main Steps
1. **Data Collection:** Fetch merged pull requests using the GitHub API.
2. **Data Preparation:** Combine titles and bodies into a single text column.
3. **Embedding Generation:** Convert text into numerical vectors using `all-MiniLM-L6-v2`.
4. **Clustering:** Apply Agglomerative Clustering with a distance threshold of 0.9.
5. **Result Saving:** Store cluster outputs in CSV format for analysis.

### Algorithm Used
The project uses **Agglomerative Clustering**, a hierarchical unsupervised learning method that merges the most similar items step by step until a threshold is reached.

**Why this method:**
- Works well with sentence embeddings.
- No need to predefine the number of clusters.
- Easy to control precision with a threshold value.

### Results
- **Total clusters:** 397  
- **Single-PR clusters:** 394  
- **Multi-PR clusters:** 3  
- **Largest cluster size:** 2  

Most pull requests were unique, showing a wide variety of PR topics in the repository.

### Future Improvements
- Add PCA or t-SNE visualizations to display cluster separation.  
- Allow users to upload GitHub data directly in the Streamlit interface.  
- Test other embedding models like `all-mpnet-base-v2`.  
- Try clustering algorithms such as DBSCAN or HDBSCAN.  

### Problems Faced
- Understanding how GitHub pull requests work.  
- Creating and managing GitHub tokens.  
- Learning the concept of embeddings and clustering thresholds.  
- Handling environment setup issues in Streamlit.  
- Debugging installation and dependency problems.  

### Personal Learning
This was my first project in software engineering using AI. I learned how to collect and process real GitHub data, create embeddings, and apply clustering. It also helped me understand how AI methods can be used for real-world software development analysis.

### Disclaimer
Some helper functions (like GitHub API calls and Streamlit setup) were developed with the help of online large language models to save time.  
All data collection, clustering, and analysis were done manually.
