import argparse

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='C:/Jyd/test/Cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='shelves',
                        choices=['voc', "shelves"], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    ## aux supervise
    parser.add_argument("--aux-alpha", type=float, default=0.6, help="aux supervise's weights")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warmup', choices=['poly', 'warmup'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='JointEdgeSegLoss',
                        choices=['cross_entropy', 'focal_loss', "hard_mine_loss", 
                                 "ssim_loss", "JointEdgeSegLoss", "OhemCrossEntropy2dTensor"], 
                        help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--edge-width", type=int, default=13, help="for training, extract binary edge's width")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    parser.add_argument("--s", type=float, default=0.0001, help="scale sparse rate (default:0.0001)")

    # ------------------------------------- for inference and onnx -----------------------------------
    parser.add_argument("--weight_path", type=str, default="C:/Users/dmall/Downloads/best_bisenetv3_shelves_os16 (3).pth", 
                            help="weight file")
    parser.add_argument("--onnx_saved_path", type=str, default="./", help="pth weight serialize to onnx weight")
    parser.add_argument("--onnx_name", type=str, default="weights.onnx", help="pth weight serialize to onnx weight")
    return parser