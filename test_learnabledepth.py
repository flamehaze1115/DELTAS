import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from assets.sequence_folders import SequenceFolder
from models import superpoint, triangulation, densedepth
from assets.utils import *
import os
import re
import time

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', default='./assets/sample_data/scannet_sample', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: sequential: sequential folders')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', default=200, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=1, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--ttype2', default='./assets/sample_data/sample_list.txt', type=str,
                    help='Text file indicates input data')
parser.add_argument('--mindepth', type=float, default=0.5, help='minimum depth')
parser.add_argument('--maxdepth', type=float, default=10.0, help='maximum depth')
parser.add_argument('--width', type=int, default=320, help='image width')
parser.add_argument('--height', type=int, default=256, help='image height')
parser.add_argument('--seq_length', default=3, type=int, help='length of sequence')
parser.add_argument('--seq_gap', default=1, type=int, help='gap between frames for ScanNet dataset')
parser.add_argument('--resume', type=bool, default=True, help='Use pretrained network')
parser.add_argument('--pretrained', dest='pretrained', default='./assets/pretrained_checkpoint.pth.tar', metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--do_confidence', type=bool, default=True, help='confidence in triangulation')
parser.add_argument('--dist_orthogonal', type=int, default=1, help='offset distance in pixels')
parser.add_argument('--kernel_size', type=int, default=1, help='kernel size')
parser.add_argument('--out_length', type=int, default=100, help='output length of epipolar patch')
parser.add_argument('--depth_range', type=bool, default=True, help='clamp using range of depth')
parser.add_argument('--num_kps', default=512, type=int, help='number of interest keypoints')
parser.add_argument('--model_type', type=str, default='resnet50', help='network backbone')
parser.add_argument('--align_corners', type=bool, default=False, help='align corners')
parser.add_argument('--descriptor_dim', type=int, default=128, help='dimension of descriptor')
parser.add_argument('--detection_threshold', type=float, default=0.0005, help='threshold for interest point detection')
parser.add_argument('--frac_superpoint', type=float, default=.5, help='fraction of interest points')
parser.add_argument('--nms_radius', type=int, default=9, help='radius for nms')

n_iter = 0


def main():
    global n_iter
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Data loading code
    print("=> fetching scenes in '{}'".format(args.data))
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    test_transform = Compose([
        Scale(),
        ArrayToTensor(),
        normalize])

    test_set = SequenceFolder(
        args.data,
        transform=test_transform,
        seed=args.seed,
        ttype=args.ttype2,
        sequence_length=args.seq_length,
        sequence_gap=args.seq_gap,
        height=args.height,
        width=args.width,
    )

    print('{} samples found in {} valid scenes'.format(len(test_set), len(test_set.scenes)))

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model")

    # step 1 using superpoint
    config_sp = {
        'top_k_keypoints': args.num_kps,
        'height': args.height,
        'width': args.width,
        'align_corners': args.align_corners,
        'detection_threshold': args.detection_threshold,
        'frac_superpoint': args.frac_superpoint,
        'nms_radius': args.nms_radius,
    }

    cudnn.benchmark = True

    supernet = superpoint.Superpoint(config_sp)
    supernet = supernet.cuda() if torch.cuda.is_available() else supernet

    # step 2 using differentiable triangulation
    config_tri = {
        'dist_ortogonal': args.dist_orthogonal,
        'kernel_size': args.kernel_size,
        'out_length': args.out_length,
        'depth_range': args.depth_range,
        'has_confidence': args.do_confidence,
        'align_corners': args.align_corners,
    }

    trinet = triangulation.TriangulationNet(config_tri)
    trinet = trinet.cuda() if torch.cuda.is_available() else trinet

    # step 3 using sparse-to-dense

    config_depth = {
        'min_depth': args.mindepth,
        'max_depth': args.maxdepth,
        'input_shape': (args.height, args.width, 1),
    }

    depthnet = densedepth.SparsetoDenseNet(config_depth)
    depthnet = depthnet.cuda() if torch.cuda.is_available() else depthnet

    # load pre-trained weights

    if args.resume:
        if torch.cuda.is_available():
            weights = torch.load(args.pretrained)
        else:
            weights = torch.load(args.pretrained, map_location=torch.device('cpu'))
        supernet.load_state_dict(weights['state_dict'], strict=True)
        trinet.load_state_dict(weights['state_dict_tri'], strict=True)
        depthnet.load_state_dict(weights['state_dict_depth'], strict=True)
        if torch.cuda.is_available():
            depthnet = torch.nn.DataParallel(depthnet).cuda()
            supernet = torch.nn.DataParallel(supernet).cuda()
            trinet = torch.nn.DataParallel(trinet).cuda()

    errors_depth, error_names = validate_with_gt(args, test_loader, supernet, trinet, depthnet, test_set)


def colorize_depth_np(input, max_depth, color_mode=cv2.COLORMAP_RAINBOW):
    input_tensor = input
    normalized = input_tensor / max_depth * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], color_mode)
        return normalized_color
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, color_mode)
        return normalized


def validate_with_gt(args, val_loader, supernet, trinet, depthnet, val_set=None):
    error_names = ['abs_rel', 'abs_diff', 'sq_rel', 'a1', 'a2', 'a3']
    errors_depth = AverageMeter(i=len(error_names))

    # switch to evaluate mode

    supernet.eval()
    trinet.eval()
    depthnet.eval()

    evaluation_dir = "/mnt/scannet/DELTAS_results"

    all_epls = []
    with torch.no_grad():

        for i, (tgt_img, ref_imgs, poses, intrinsics, tgt_depth, ref_depths, tgt_img_path) in enumerate(val_loader):
            if i % 10!=0:
                continue
            print(tgt_img_path)
            tgt_img_path = str(tgt_img_path[0])
            scene_name = tgt_img_path.split('/')[-3]
            tgt_img_name = tgt_img_path.split('/')[-1]
            tgt_index = [int(s) for s in re.findall(r'\d+', tgt_img_name)][0]
            pred_depth_name = "frame-%06d.pred_depth.npy" % tgt_index

            pred_depth_dir = os.path.join(evaluation_dir, scene_name, "pred_depth")
            if not os.path.exists(pred_depth_dir):
                os.makedirs(pred_depth_dir)

            tgt_img_var = tgt_img
            ref_imgs_var = ref_imgs
            img_var = make_symmetric(tgt_img_var, ref_imgs_var)

            start_time = time.time()
            ##Step 1: Detect and Describe Points
            data_sp = {'img': img_var, 'process_tsp': 'ts'}  # t is detector, s is descriptor
            pred_sp = supernet(data_sp)

            batch_sz = tgt_img_var.shape[0]
            img_var = img_var[:batch_sz]

            ##Pose and intrinsics
            poses_var = [pose for pose in poses]
            intrinsics_var = intrinsics
            seq_val = args.seq_length - 1
            pose = torch.cat(poses_var, 1)
            pose = pose_square(pose)

            ##Depth
            tgt_depth_var = tgt_depth
            ref_depths_var = [ref_depth for ref_depth in ref_depths]
            depth = tgt_depth_var
            depth_ref = torch.stack(ref_depths_var, 1)

            # Keypoints and descriptor logic
            keypoints = pred_sp['keypoints'][:batch_sz]
            features = pred_sp['features'][:batch_sz]
            skip_half = pred_sp['skip_half'][:batch_sz]
            skip_quarter = pred_sp['skip_quarter'][:batch_sz]
            skip_eight = pred_sp['skip_eight'][:batch_sz]
            skip_sixteenth = pred_sp['skip_sixteenth'][:batch_sz]
            scores = pred_sp['scores'][:batch_sz]
            desc = pred_sp['descriptors']
            desc_anc = desc[:batch_sz, :, :, :]
            desc_view = desc[batch_sz:, :, :, :]
            desc_view = reorder_desc(desc_view, batch_sz)

            ## Step 2: Match & Triangulate Points
            data_sd = {'iter': n_iter, 'intrinsics': intrinsics_var, 'pose': pose, 'depth': depth,
                       'ref_depths': depth_ref, 'scores': scores,
                       'keypoints': keypoints, 'descriptors': desc_anc, 'descriptors_views': desc_view,
                       'img_shape': tgt_img_var.shape, 'sequence_length': seq_val}
            pred_sd = trinet(data_sd)

            view_matches = pred_sd['multiview_matches']
            anchor_keypoints = pred_sd['keypoints']
            keypoints3d_gt = pred_sd['keypoints3d_gt']
            range_mask_view = pred_sd['range_kp']
            range_mask = torch.sum(range_mask_view, 1)

            d_shp = tgt_depth_var.shape
            keypoints_3d = pred_sd['keypoints_3d']
            kp3d_val = keypoints_3d[:, :, 2].view(-1, 1).t()
            kp3d_filter = (range_mask > 0).view(-1, 1).t()
            kp3d_filter = (kp3d_filter) & (kp3d_val > args.mindepth) & (kp3d_val < args.maxdepth)

            ## Step 3: Densify using Sparse-to-Dense
            data_dd = {'anchor_keypoints': keypoints, 'keypoints_3d': keypoints_3d, 'sequence_length': args.seq_length,
                       'skip_sixteenth': skip_sixteenth,
                       'range_mask': range_mask, 'features': features, 'skip_half': skip_half,
                       'skip_quarter': skip_quarter, 'skip_eight': skip_eight}
            pred_dd = depthnet(data_dd)
            output = pred_dd['dense_depth']



            elps = time.time() - start_time
            all_epls.append(elps)

            # Calculate metrics
            tgt_depth_tiled = depth
            pred_depth = output.squeeze().cpu().numpy()
            np.save(os.path.join(pred_depth_dir, pred_depth_name), np.float16(pred_depth))
            colored_pred_depth = colorize_depth_np(pred_depth, max_depth=5.0)
            cv2.imwrite(os.path.join(pred_depth_dir, pred_depth_name.replace("npy", "jpg")),
                        cv2.cvtColor(np.uint8(colored_pred_depth), cv2.COLOR_RGB2BGR))

            if output.is_cuda:
                tgt_depth_tiled = tgt_depth_tiled.to(output.device)

            mask = (tgt_depth_tiled <= args.maxdepth) & (tgt_depth_tiled >= args.mindepth) & (
                    tgt_depth_tiled == tgt_depth_tiled)
            mask.detach_()
            output = output.squeeze(1)
            errors_depth.update(compute_errors(tgt_depth_tiled, output, mask, False))
            if i % int(0.5 * args.print_freq) == 0:
                print(' TEST: Depth Error {:.4f} ({:.4f})'.format(errors_depth.avg[1], errors_depth.avg[0]))

    print("inference time:", np.mean(all_epls))
    return errors_depth.avg, error_names


if __name__ == '__main__':
    main()
