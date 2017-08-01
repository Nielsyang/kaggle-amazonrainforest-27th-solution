# kaggle-amazonrainforest-27th-solution
27th solution of [Amazon rainforest](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

## My brief solution
(1) use resnet and densenet

(2) drop random crop and add label smoothing

(3) TTA(Test Time Augmentation)

(4) because the error is mainly concentrated on label water and cultivation, i train two single model on water and cultivation

(5) best f2 threshold search

(6) 5-fold voting
