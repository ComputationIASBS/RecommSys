# RSAttAE: An Information-Aware Attention-based Autoencoder Recommender System

![architecture](plots/RSAttAE_architecture.png)
This repository contains the code and implementations for the paper "RSAttAE: An Information-Aware Attention-based
Autoencoder Recommender System". Both the Vanilla Autoencoder and proposed Attention Autoencder models are available in `models` directory. The main directory involves following files:

| File | Description |
| ---- | ----------- |
| settings.py | global settings such as dataset path.|
| features_stats.py | generating plots related to MovieLens100k data.|
| data.py | functions to lead ratings data and features.|
| helper.py | functions such as `masked_rmse_loss`.|
| train_user_autoencoder.py | training vanilla autoencoder to learn users embeddings.|
| train_movie_autoencoder.py | training vanilla autoencoder to learn movies embeddings.|
| train_user_attention_autoencoder.py | training attention autoencoder to learn users embeddings.|
| train_movie_attention_autoencoder.py | training attention autoencoder to learn movies embeddings.|
| XGboost_params_grid_search.py | Param grid search for XGBoost model to predict ratings in supervised setting using learned embeddings.|

To reproduce the paper results:

1- train the Attention Autoencoder for users which outputs `users_embeddings_attention_autoencoder_64_0.5.pt`.

2- train the Attention Autoencoder for movies which outputs `movies_embeddings_attention_autoencoder_64_0.5.pt`.

3- Use two `.pt` file and run the `XGboost_params_grid_search.py` to tune hyperparameters for XGBoost model.

Feel free to try different parameters and any contribution is welcomed. if you have problem in any section please contact us. 

#### Cite Us
```
....
```