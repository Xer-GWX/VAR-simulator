1、生成算子图：
python ./generate_model_ir.py
prefill在 ./.output/ir_output/Layer_list_prefill.json

2、
后面用的是basic_list.json做的test，这里面提取了上面prefill过程的1个stage1个block的除了silu，transpose等等操作的例子

python ./main.py[需要打一下断点，只跑一个layer，剩下的layer还没写]

main.py[目前只试了跑一个layer(linear)的情况，第二个layer是matmul1，还没有写MM的这种情况]
    config.py [得到layerconfig,hardware的config，以及layermap后的config]
        hardware.py [得到hardware config]
    compute_block.py [这是执行一个layer计算访存的地方]
        draw_pipeline.py [这里只draw了1个A_block * 1个B_block的]

decode算子图的问题：
1、attnblock 源码将四维matmul reshape成了三维，这里reshape并没有单独列出来
2、一些地方的linear的bias torch.是一维，统一归为[1,1,bias] 但是linear的输入是四维就有bug了
3、hidden = F.interpolate(x, scale_factor=2, mode='nearest') # 这里上采样，没有引入
   result = self.conv(hidden) 
// {
    //     "name": "mat2",
    //     "type": "attention_MM",
    //     "input": [
    //       {
    //         "name": "1",
    //         "shape": [
    //           25,
    //           55
    //         ],
    //         "dtype": "float16"
    //       },
    //       {
    //         "name": "1",
    //         "shape": [
    //           55,
    //           1024
    //         ],
    //         "dtype": "float16"
    //       }
    //     ],
    //     "output": [
    //       {
    //           "name": "0",
    //           "shape": [
    //             25,
    //             1024
    //           ],
    //           "dtype": "float16"
    //         }
    //     ],
    //     "param": {}
    // },