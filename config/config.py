PARA = dict(
    train=dict(
        epoch = 200,
        batch_size = 128,
        lr = 0.001,
        momentum=0.9,
        wd = 5e-4,
        num_workers = 2,
        divice_ids = [1]
    ),
    test=dict(
        batch_size=100
    ),
    cifar10_paths = dict(
        validation_rate = 0.05,

        root = '/home/caoyh/DATASET/cifar10/',

        original_trainset_path = '/home/caoyh/DATASET/cifar10/cifar-10-batches-py/',#train_batch_path
        original_testset_path = '/home/caoyh/DATASET/cifar10/cifar-10-batches-py/',

        after_trainset_path = '/home/caoyh/DATASET/cifar10/trainset/',
        after_testset_path = '/home/caoyh/DATASET/cifar10/testset/',
        after_validset_path = '/home/caoyh/DATASET/cifar10/validset/',

        train_data_txt = '/home/caoyh/DATASET/cifar10/train.txt',
        test_data_txt = '/home/caoyh/DATASET/cifar10/test.txt',
        valid_data_txt = '/home/caoyh/DATASET/cifar10/valid.txt',
    ),
    utils_paths = dict(
        checkpoint_path = './cache/checkpoint/',
        log_path = './cache/log/',
        visual_path = './cache/visual/',
        params_path = './cache/params/',
    ),
)