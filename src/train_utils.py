from .lora import LoRAWrapper

def get_params(model, method="lora"):
    params = []
    if method=="lora":
        for layer in model.modules():
            if isinstance(layer, LoRAWrapper):
                for n, p in layer.named_parameters():
                    if n in ["A", "B"]:
                        params.append(p)
        return params
    elif method=="adapter":
        pass
    else:
        return NotImplementedError

