from calflops import calculate_flops
from torchvision import models

model = models.alexnet()
batch_size = 1
input_shape = (batch_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

# in module.py predict_start()
# from calflops import calculate_flops
# x = cond_audio.long()
# x = self.encoder.label_emb(x)


# flops, macs, params = calculate_flops(model=self.encoder, 
#     input_shape=tuple(x.shape),
#     output_as_string=True,
#     output_precision=4)
# print("Encoder FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
# exit()