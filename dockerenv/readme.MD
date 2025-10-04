### Docker build image on local host
```shell
 docker build -t dnn-dev-env:latest -f Dockerfile.update .                                             
 ```



### My docker images repos on AliCloud
https://cr.console.aliyun.com/repository/cn-guangzhou/llm_agents/dnn_dev_env/details


```shell
$ sudo docker login --username=13008861045@163.com crpi-8nfy446089zsqqbx.cn-guangzhou.personal.cr.aliyuncs.com
```
#### 将镜像推送到Registry
```shell

$ docker login --username=13008861045@163.com crpi-8nfy446089zsqqbx.cn-guangzhou.personal.cr.aliyuncs.com
$ docker tag [ImageId] crpi-8nfy446089zsqqbx.cn-guangzhou.personal.cr.aliyuncs.com/llm_agents/dnn_dev_env:[镜像版本号]
$ docker push crpi-8nfy446089zsqqbx.cn-guangzhou.personal.cr.aliyuncs.com/llm_agents/dnn_dev_env:[镜像版本号]
```
