setwd('~/projects/crispr-phylogeny')
library('Quartet')
library('TreeDist')
truth_tree = ape::read.tree('data/benchmark/SubC2_train_REF/SubC2_train_0002_REF.nw')
for(dm_method in c('l','hamm1','hamm2')){
  print(dm_method)
  for(method in c('weighbor','nj','upgma')){
    tree = ape::read.tree(paste0('output/',method,'-',dm_method,'.txt'))
    statuses <- QuartetStatus(truth_tree, tree)
    print(method)
    
    # uncomment these to see different metrics:
    
    #print(RobinsonFoulds(truth_tree,tree))
    #print(SimilarityMetrics(statuses, similarity = TRUE))
    print(TreeDistance(truth_tree, tree))
    #print(NyeSimilarity(truth_tree, tree))
    #print(MatchingSplitDistance(truth_tree,tree))
  }
  print('------------------------')
}

tree1 = ape::read.tree(paste0('output/','weighbor','-','l','.txt'))
tree2 = ape::read.tree(paste0('output/','nj','-','hamm1','.txt'))
#print(ClusteringInfoDistance(tree1, tree2, reportMatching = TRUE))
VisualizeQuartets(tree1, tree2)
print(TreeDistance(tree1, tree2))