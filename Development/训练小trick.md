warming up 会通过若干的设定的epoch， 将learning rate从0.0通过线性增加等方式达到正式的base learning rate.
在这个过程中，如果性能出现了不升反降，我们不如找到他的巅峰值对应的learning rate作为最终的base learning rate.
这样反复多次设定 base learning rate ，训练有可能达到比较好的性能。