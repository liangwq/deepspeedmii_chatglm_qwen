# deepspeedmii_chatglm_qwen
## deepspeed的grpc框架支持chatglm和qwen代码开发
1.deepspeed在inference中增加了适配chatglm的代码，cd 到deepspeed文件夹 pip install -e . <br \>
2.deepspeedmii中做了chatglm适配代码改造，cd 到deepspeedmii文件夹 pip install -e . <br \>
3.启动测试： <br \>
a.下载模型到指定路径 <br \>
b.更换test_chatglm_sever.py中文件路径名，python chatglm_test_sever.py启动服务 <br \>
d.启动客户端测试服务返回结果：python chatglm_test_client.py <br \>
