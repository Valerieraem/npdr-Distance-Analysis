# create sim
# npdro regain, network, find best pairs, use on gene expression and masspec

# update ndpro nearestNeighbors?
# update npdro to include non-parallel regain. 
# update nearestNeighbors and npdr in npdro.

# npdr modify: allow distance input, metric = precomputed, external.dist
# npdr baseline feature selection
#      PC distance (also useful for gwas) -- how many PC?
#      rf proximity distance (with and without supervision)
#         evaporative: remove worst 10% npdr variables and repeat distance
#      umap distance?

# ? nbd.method = precomputed? 
# use umap nn.descent? still have to fix k, but could use surf approx fixed k.
# dann adaptive nearest neighbors

# knn network clustering
# PC's (how many?), rf proximity (increase trees, with and w/o class label),
# choice of knn controls number of clusters
# multisurf adaptive radius, results in undirected (Louvain clustering)
# Leiden clustering?
# Transpose the clustering to see if interacting variables cluster together. 

###################################################################
# simulate data
library(npdro)
num.samples <- 300
num.variables <- 100
dataset <- npdro::createSimulation2(num.samples=num.samples,
                                    num.variables=num.variables,
                                    pct.imbalance=0.5,
                                    pct.signals=0.2,
                                    main.bias=0.5,
                                    interaction.bias=1,
                                    hi.cor=0.95,
                                    lo.cor=0.2,
                                    mix.type="main-interactionScalefree",
                                    label="class",
                                    sim.type="mixed",
                                    pct.mixed=0.5,
                                    pct.train=0.5,
                                    pct.holdout=0.5,
                                    pct.validation=0,
                                    plot.graph=F,
                                    verbose=T)
dats <- rbind(dataset$train, dataset$holdout, dataset$validation)
colnames(dats)
dim(dats)
# functional variables: main effects and interaction effects
#dataset$signal.names
#dataset$main.vars
#dataset$int.vars
#functional_mask <- colnames(dats[,-ncol(dats)]) %in% dataset$signal.names
# underlying variable network
#A.graph <- graph_from_adjacency_matrix(dataset$A.mat, mode="undirected")
#isSymmetric(dataset$A.mat)
#plot(A.graph)
#deg.vec <- degree(A.graph)
#hist(deg.vec)

###################################################################
#Deep Learning Autoencoder
library(keras)

input_size = 100
layer1_size = 60
#Smallest dimension (fully compressed)
encoding_dim = 30

#Encoder
encoder_input <- layer_input(shape = input_size)
encoded <- encoder_input %>% 
  layer_dense(layer1_size, activation='sigmoid') %>%
  layer_dense(encoding_dim, activation='relu')

encoder <- keras_model(encoder_input, encoded)
summary(encoder)

#Decoder
decoder_input <- layer_input(shape = encoding_dim)
decoded <- decoder_input %>%
  layer_dense(layer1_size, activation='relu') %>%
  layer_dense(input_size, activation='sigmoid')

decoder <- keras_model(decoder_input, decoded)
summary(decoder)

#Autoencoder
autoencoder_input <- layer_input(shape = input_size)
autoencoder <- autoencoder_input %>%
  encoder() %>%
  decoder()

autoencoded <- keras_model(autoencoder_input, autoencoder)
summary(autoencoded)

#Training Data
class_col <- which(colnames(dataset$train)=="class")
training_to_matrix <- dataset$train[, -class_col]
training_set <- as.matrix(training_to_matrix)

#Validation/Testing Data
class_col <- which(colnames(dataset$holdout)=="class")
testing_to_matrix <- dataset$holdout[, -class_col]
holdout_set <- as.matrix(testing_to_matrix)


#Training
autoencoded %>% compile(optimizer='adam', loss='categorical_crossentropy')
autoencoded %>% fit(training_set, training_set, validation_data=list(holdout_set, holdout_set), metrics=list(tf.keras.metrics.Accuracy()), epochs=50, batch_size =256)

#Testing
encoded_holdout <- encoder %>% predict(holdout_set)
decoded_holdout <- decoder %>% predict(encoded_holdout)
encoded_holdout <- as.data.frame(encoded_holdout)
decoded_holdout <- as.data.frame(decoded_holdout)

#Disance Matrix Creation
class_col <- which(colnames(dats)=="class")
data_to_matrix <- dats[, -class_col]
data_whole_set <- as.matrix(data_to_matrix)
autoencoded_dist_matrix <- encoder %>% predict(data_whole_set)
auto_dist <- npdr::npdrDistances(autoencoded_dist_matrix, metric="euclidean")

npdr_aed_results <- npdr::npdr("class", dats, regression.type="binomial",
                              attr.diff.type="numeric-abs",
                              nbd.method="relieff",
                              nbd.metric = "precomputed",
                              external.dist=auto_dist,
                              #msurf.sd.frac=.5,
                              knn=my.k,
                              neighbor.sampling="none", dopar.nn = F,
                              padj.method="bonferroni", verbose=T)
npdr_aed_results[npdr_aed_results$pval.adj<.05,] # pval.adj, first column
npdr_aed_results[1:20,1] # top 20


#TODO
#Fix a Training and Test Dataset Percentages

#check loss function

#Test accuracy of model

###################################################################
# Random Forrest
library(randomForest)
rf<-randomForest(as.factor(dats$class) ~ .,data=dats, ntree=5000) 
print(rf)  # error
importance(rf)

imp<-sort(importance(rf),decreasing=T,index.return=T)
imp_sorted<-cbind(rownames(importance(rf))[imp$ix],importance(rf)[imp$ix])
imp_sorted[1:20]

###################################################################
# npdr
library(npdr)
# if you have imbalanced data:
min.group.size <- min(as.numeric(table(dats[, "class"])))
my.k <- npdr::knnSURF(2*min.group.size - 1, 0.5)
npdr_results <- npdr::npdr("class", dats, regression.type="binomial", 
                     attr.diff.type="numeric-abs",
                     nbd.method="relieff", 
                     nbd.metric = "manhattan", 
                     msurf.sd.frac=.5,
                     knn=my.k,
                     neighbor.sampling="none", dopar.nn = F,
                     padj.method="bonferroni", verbose=T)
npdr_results[npdr_results$pval.adj<.05,] # pval.adj, first column
npdr_results[1:20,1] # top 20

###################################################################
# Find principal components of observations in variable space
# Choose the number of PCs to use
class_col <- which(colnames(dats)=="class")
predictors <- dats[, -class_col]
PC<-prcomp(predictors)
pc.num <- 10
topPCs <- PC$x[,1:pc.num]  # observations x PCs
# Find k nearest-neighbors in PC space.
#my.k <- 15  # larger k, fewer clusters
pc.dist <- npdr::npdrDistances(topPCs, metric="euclidean")
npdr_pc_results <- npdr::npdr("class", dats, regression.type="binomial", 
                           attr.diff.type="numeric-abs",
                           nbd.method="relieff", 
                           nbd.metric = "precomputed",
                           external.dist=pc.dist,
                           #msurf.sd.frac=.5,
                           knn=my.k, 
                           neighbor.sampling="none", dopar.nn = F,
                           padj.method="bonferroni", verbose=T
                           )
npdr_pc_results[npdr_pc_results$pval.adj<.05,] # pval.adj, first column
npdr_pc_results[1:20,1] # top 20

library(randomForest)
rf<-randomForest(y=as.factor(dats$class), x=predictors,
                  ntree=5000,
                  proximity=T) 
# naive distance from proximity
rf.dist <- 1-rf$proximity
max(rf.dist)
min(rf.dist)
print(rf)  # error rate
npdr_rf_results <- npdr::npdr("class", dats, regression.type="binomial", 
                              attr.diff.type="numeric-abs",
                              nbd.method="relieff", 
                              nbd.metric = "precomputed",
                              external.dist=rf.dist,
                              #msurf.sd.frac=.5,
                              knn=my.k, 
                              neighbor.sampling="none", dopar.nn = F,
                              padj.method="bonferroni", verbose=T
)
npdr_rf_results[npdr_rf_results$pval.adj<.05,] # pval.adj, first column
npdr_rf_results[1:20,1] # top 20
