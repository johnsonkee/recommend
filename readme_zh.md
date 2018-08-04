# 1. 代码说明
这个代码是改编自[mlper's recommendation](https://github.com/mlperf/reference/tree/master/recommendation)，参考的文献是[Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569)。

代码的原先框架是pytorch，在保持代码功能不变的情况下把原先的代码改为了mxnet框架。
# 2. mxnet搭建的步骤
以下的内容是基于mxnet的gluon模块的，gluon模块是mxnet2017年新推出的模块，比原先的module模块更便于使用，包括静态图和动态图互相转换、自动求导等简单功能。

## 参考资料
[Gluon新手资料](https://gluon-crash-course.mxnet.io/)  
[Gluon进阶资料](http://zh.gluon.ai/)  
[Gluon中文论坛](https://discuss.gluon.ai/)

## 搭建网络结构
# 3. mxnet和pytorch在创建网络时的不同点
```flow
st=>start: Start
op=>operation: Your Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
```
