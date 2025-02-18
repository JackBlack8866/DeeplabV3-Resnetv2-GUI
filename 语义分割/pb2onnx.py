import tensorflow as tf
import tf2onnx


with tf.io.gfile.GFile("frozen_inference_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())


input_names = ["sub_7:0"]  # 修改为你的输入节点名称
output_names = ["ResizeBilinear_3:0"]  # 修改为你的输出节点名称


model_proto, _ = tf2onnx.convert.from_graph_def(
    graph_def,
    output_names=output_names,
    input_names=input_names,
    opset=13
)


with open("frozen_inference_graph3.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
