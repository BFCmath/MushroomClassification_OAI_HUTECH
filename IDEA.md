# Idea

## Cross Validation
+ Some tries and I realized that the local score can easily get up to 0.9, but lb score is about 0.7-0.8!
+ It seems that the test set is not randommly sampled from the train set.
+ So a correlation between cv and lb is the key to win this competition.
+ Spend time for manual CV...
+ It indeed made the correlation between cv and lb better, however, the gap is still there (0.85->0.9 for cv and 0.8->0.82 for lb).
+ Its suprising that training on the random cv gives a much better lb score than the manual cv, although the gap is notable!!!
+ Maybe consider splitting a harder cv set...? or just use the random cv...???

## Models
### CNN
+ Residual blocks seem to be the key!
+ Some test and indeed the residual blocks perform better than the normal conv blocks.
+ Trying Resnet and DenseNet, DenseNet seems to be better.

+ Dual branch network seems to be a good idea (global and local features).
+ SPD seems to be another key to win this competition!

+ InceptionFSD give a valid result, maybe FSD is another key to win this competition!
### Transformer
+ Pixel Transformer seems to be a good idea.
+ We can even push that further with multi scale pixel transformer (1x1, 2x2, 4x4, 8x8, 16x16, 32x32).
+ ViT seems to not work well, maybe because of the lack of data.

## Dropout
+ Some experiments and it seems that the dropout is not the best way to go.
+ Need some paper reading to confirm this...

## Data Augmentation
+ Data augmentation is a must because the dataset is too small and the data is taken in many different conditions/aspects.
+ Work on to figure out which data augmentation is the best for this competition.
+ Extensive data augmentation and multiscale transformations work pretty well!
+ multiscale currently seems to be the best data augmentation method, but need to check again for the extensive data augmentation coding...
+ Try to merge the multiscale and extensive data augmentation, but only the low aug is working... So maybe stick with the multiscale.
+ What about randomly erase pixels? Because the images are low resolution, so randomly erase pixels?
+ Actually, instead of erase, we can apply random white or black pixels to the image.
+ The above approach seems to punish the model too much!
+ Mosaic and cutmix didn't work well, however, we can use them to create a fifth class, which is mix, to push the model to further learn global features! Need to do some experiments on this...
+ Some research for data augmentation, and here is some nice one we could try:
  + CutMix
  + MixUp
  + AutoAugment
  + Repeated Augment

## Regularization
+ Some research for regularization, and here is some nice one we could try:
  + Label Smoothing
  + Stochastic depth
  + Random erasing
## Downsampling
+ Some paper reading and it seems to me that the normal downsampling is not the best way to go.
+ Try with other downsampling methods like dilated convolutions, space to depth convolutions, etc...
+ Testing to make sure that the SPD is the key to win this competition...
+ SPD indeed gives a better result than the normal downsampling!
+ FSD (Feature Space Downsampling) seems to work well, but need to check again.

## Robustness
+ Attack the models with high and low aug images
+ TTA
+ Adversarial validation
