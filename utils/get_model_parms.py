def get_model_params(net,arg,cfg):
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    with open(cfg.PARA.utils_paths.params_path+arg.net+'_params.txt') as f:
        f.write('total_params:%d\n'%total_params)
        f.write('total_trainable_params: %d\n'%total_trainable_params)
    # return total_params, total_trainable_params






