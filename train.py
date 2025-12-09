import argparse
import numpy as np
import os
import signal
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='mode-connectivity', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')

# Early stopping parameters
parser.add_argument('--early_stopping', action='store_true', help='enable early stopping based on validation error')
parser.add_argument('--patience', type=int, default=20, metavar='N',
                    help='early stopping patience (default: 20)')
parser.add_argument('--min_delta', type=float, default=0.0, metavar='DELTA',
                    help='minimum improvement to count as better (default: 0.0)')
parser.add_argument('--split_test_from_train', action='store_true',
                    help='split training data into train/val/test (40K/5K/5K)')

# Symmetry plane projection
parser.add_argument('--project_symmetry_plane', action='store_true',
                    help='project middle bend to symmetry plane after each optimizer step (requires num_bends=3, fix_start=True, fix_end=True)')

args = parser.parse_args()

# Initialize wandb if requested
use_wandb = args.wandb and WANDB_AVAILABLE
if use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    split_test_from_train=args.split_test_from_train
)

architecture = getattr(models, args.model)

if args.curve is None:
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
else:
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=architecture.kwargs,
    )
    base_model = None
    if args.resume is None:
        for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
            if path is not None:
                if base_model is None:
                    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                checkpoint = torch.load(path)
                print('Loading %s as point #%d' % (path, k))
                base_model.load_state_dict(checkpoint['model_state'])
                model.import_base_parameters(base_model, k)
        if args.init_linear:
            print('Linear initialization.')
            model.init_linear()
model.to(device)

# Validate symmetry plane projection requirements
if args.project_symmetry_plane:
    if args.curve is None:
        raise ValueError("--project_symmetry_plane requires --curve to be specified")
    if args.num_bends != 3:
        raise ValueError("--project_symmetry_plane requires --num_bends=3 (got {})".format(args.num_bends))
    if not args.fix_start or not args.fix_end:
        raise ValueError("--project_symmetry_plane requires --fix_start and --fix_end")
    if args.init_start is None or args.init_end is None:
        raise ValueError("--project_symmetry_plane requires --init_start and --init_end")
    print("\n" + "=" * 80)
    print("SYMMETRY PLANE PROJECTION ENABLED")
    print("=" * 80)
    print("Middle bend will be projected to symmetry plane after each optimizer step")
    print("Plane constraint: n · (θ - m) = 0")
    print("  where n = w₂ - w₁ (normal vector)")
    print("        m = (w₁ + w₂) / 2 (midpoint)")
    print("=" * 80 + "\n")


def project_to_symmetry_plane(model, midpoint_params, normal_params):
    """
    Project middle bend (index=1) parameters to symmetry plane.

    The plane is defined by: n · (θ - m) = 0
    Projection: θ_new = θ - ((θ - m)·n / ||n||²) * n
    """
    all_params = list(model.net.parameters())
    num_bends = model.num_bends

    with torch.no_grad():
        for i in range(0, len(all_params), num_bends):
            theta_param = all_params[i + 1]  # Middle bend
            midpoint_param = midpoint_params[i // num_bends]
            normal_param = normal_params[i // num_bends]

            # Compute displacement from midpoint
            displacement = theta_param - midpoint_param

            # Compute dot product and scale
            dot_product = torch.sum(displacement * normal_param)
            normal_norm_sq = torch.sum(normal_param ** 2)
            scale = dot_product / (normal_norm_sq + 1e-10)

            # Project: θ_new = θ - scale * n
            theta_param.sub_(scale * normal_param)


def compute_symmetry_plane_params(model):
    """Compute midpoint and normal vector from endpoints."""
    all_params = list(model.net.parameters())
    num_bends = model.num_bends

    midpoint_params = []
    normal_params = []

    for i in range(0, len(all_params), num_bends):
        w1_param = all_params[i]      # First endpoint (index 0)
        w2_param = all_params[i + 2]  # Second endpoint (index 2)

        midpoint = (w1_param + w2_param) / 2.0
        normal = w2_param - w1_param

        midpoint_params.append(midpoint)
        normal_params.append(normal)

    return midpoint_params, normal_params


# Compute symmetry plane parameters if needed (once, stays constant)
if args.project_symmetry_plane:
    midpoint_params, normal_params = compute_symmetry_plane_params(model)


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


criterion = F.cross_entropy
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd if args.curve is None else 0.0
)


start_epoch = 1
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']
if args.split_test_from_train:
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_nll', 'val_acc', 'te_nll', 'te_acc', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

# Early stopping variables
best_val_error = float('inf')
best_epoch = 0
patience_counter = 0
early_stopped = False

# Track L2 norm of middle point for curve models
middle_point_l2_norms = []
interpolated_l2_norms = []
epochs_list = []

has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}
val_res = {'loss': None, 'accuracy': None, 'nll': None}

# Signal handler for graceful shutdown on Ctrl+C
interrupted = False
def signal_handler(sig, frame):
    global interrupted
    print('\n' + '='*70)
    print('INTERRUPT SIGNAL RECEIVED (Ctrl+C)')
    print('='*70)
    print('Saving checkpoint before exit...')
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Evaluate initial model (epoch 0) before training
if start_epoch == 1:  # Only for new training, not resumed
    print("\n" + "=" * 80)
    print("INITIAL EVALUATION (Epoch 0 - Before Training)")
    print("=" * 80)

    if args.curve is None or not has_bn:
        if args.split_test_from_train and 'val' in loaders:
            val_res = utils.test(loaders['val'], model, criterion, regularizer)
            test_res = utils.test(loaders['test'], model, criterion, regularizer)
        else:
            test_res = utils.test(loaders['test'], model, criterion, regularizer)

        # For curve models, evaluate at t=0.5 (midpoint)
        if args.curve is not None:
            t_midpoint = torch.tensor([0.5]).to(device)
            if has_bn:
                utils.update_bn(loaders['train'], model, device=device, t=t_midpoint)
            if args.split_test_from_train and 'val' in loaders:
                val_res = utils.test(loaders['val'], model, criterion, regularizer, device=device, t=t_midpoint)
                test_res = utils.test(loaders['test'], model, criterion, regularizer, device=device, t=t_midpoint)
            else:
                test_res = utils.test(loaders['test'], model, criterion, regularizer, device=device, t=t_midpoint)

        # Print initial metrics
        print(f"Initial test loss: {test_res['loss']:.4f}")
        print(f"Initial test acc:  {test_res['accuracy']:.2f}%")
        print(f"Initial test err:  {100.0 - test_res['accuracy']:.2f}%")
        if args.split_test_from_train and val_res['accuracy'] is not None:
            print(f"Initial val loss:  {val_res['loss']:.4f}")
            print(f"Initial val acc:   {val_res['accuracy']:.2f}%")

        # Log to WandB
        if use_wandb:
            log_dict = {
                'epoch': 0,
                'test/loss': test_res['loss'],
                'test/accuracy': test_res['accuracy'],
                'test/error': 100.0 - test_res['accuracy'],
            }
            if args.split_test_from_train and val_res['accuracy'] is not None:
                log_dict.update({
                    'val/loss': val_res['loss'],
                    'val/accuracy': val_res['accuracy'],
                    'val/error': 100.0 - val_res['accuracy'],
                })
            wandb.log(log_dict, step=0)
            print("✓ Initial metrics logged to WandB")

        print("=" * 80 + "\n")

for epoch in range(start_epoch, args.epochs + 1):
    # Check if interrupted at the start of epoch
    if interrupted:
        print(f'Saving checkpoint at epoch {epoch-1}...')
        utils.save_checkpoint(
            args.dir,
            epoch - 1,
            name='checkpoint-interrupted',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
        print(f'✓ Checkpoint saved to: {args.dir}/checkpoint-interrupted-{epoch-1}.pt')
        print('Exiting gracefully.')
        sys.exit(0)


    time_ep = time.time()

    lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    utils.adjust_learning_rate(optimizer, lr)

    # Prepare projection function if symmetry plane is enabled
    projection_fn = None
    if args.project_symmetry_plane:
        projection_fn = lambda: project_to_symmetry_plane(model, midpoint_params, normal_params)

    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer, projection_fn=projection_fn)

    # Evaluate on validation set if 3-way split is used
    if args.split_test_from_train and 'val' in loaders:
        if args.curve is None or not has_bn:
            val_res = utils.test(loaders['val'], model, criterion, regularizer)
            test_res = utils.test(loaders['test'], model, criterion, regularizer)
    else:
        # Standard 2-way split: test is actually validation
        if args.curve is None or not has_bn:
            test_res = utils.test(loaders['test'], model, criterion, regularizer)

    # Early stopping logic (only for non-curve models)
    if args.early_stopping and args.curve is None:
        # Use validation set if available (3-way split), otherwise use test set
        if args.split_test_from_train and val_res['accuracy'] is not None:
            val_error = 100.0 - val_res['accuracy']
        elif test_res['accuracy'] is not None:
            val_error = 100.0 - test_res['accuracy']
        else:
            val_error = None

        if val_error is not None:
            # Check if validation error improved
            if val_error < (best_val_error - args.min_delta):
                best_val_error = val_error
                best_epoch = epoch
                patience_counter = 0

                # Save best model checkpoint
                utils.save_checkpoint(
                    args.dir,
                    epoch,
                    name='checkpoint-best',
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    val_error=val_error
                )

                # Save best epoch info
                with open(os.path.join(args.dir, 'best_epoch.txt'), 'w') as f:
                    f.write(f'Best epoch: {best_epoch}\n')
                    f.write(f'Best val error: {best_val_error:.4f}%\n')

                print(f'✓ New best val error: {best_val_error:.4f}% (patience reset)')
            else:
                patience_counter += 1
                print(f'  No improvement (patience: {patience_counter}/{args.patience})')

                # Check if patience exceeded
                if patience_counter >= args.patience:
                    print(f'\n{"="*70}')
                    print(f'Early stopping triggered at epoch {epoch}')
                    print(f'Best val error: {best_val_error:.4f}% at epoch {best_epoch}')
                    print(f'{"="*70}\n')
                    early_stopped = True

    # Regular periodic checkpoint saving
    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep

    # Compute L2 norm of middle point for curve models
    middle_point_l2_norm = None
    interpolated_l2_norm = None
    if args.curve is not None:
        # 1. Calculate L2 norm of the middle trainable point (raw parameters)
        l2_sum = 0.0
        for name, param in model.named_parameters():
            # Middle point parameters have suffix '_1' for 3-bend Bezier curves
            if '_1' in name and param.requires_grad:
                l2_sum += torch.sum(param ** 2).item()
        middle_point_l2_norm = torch.sqrt(torch.tensor(l2_sum)).item()

        # 2. Calculate L2 norm at t=0.5 (interpolated model)
        t = torch.FloatTensor([0.5]).to(device)
        weights = model.weights(t)
        interpolated_l2_norm = np.sqrt(np.sum(np.square(weights)))

        # Track for saving later
        epochs_list.append(epoch)
        middle_point_l2_norms.append(middle_point_l2_norm)
        interpolated_l2_norms.append(interpolated_l2_norm)

    # Prepare values for logging
    if args.split_test_from_train:
        values = [epoch, lr, train_res['loss'], train_res['accuracy'],
                  val_res['nll'], val_res['accuracy'],
                  test_res['nll'], test_res['accuracy'], time_ep]
    else:
        values = [epoch, lr, train_res['loss'], train_res['accuracy'],
                  test_res['nll'], test_res['accuracy'], time_ep]

    # Log to wandb if enabled
    if use_wandb:
        log_dict = {
            'epoch': epoch,
            'lr': lr,
            'train/loss': train_res['loss'],
            'train/accuracy': train_res['accuracy'],
            'train/error': 100.0 - train_res['accuracy'],
            'time_per_epoch': time_ep
        }

        # Add L2 norm of middle point for curve models
        if middle_point_l2_norm is not None:
            log_dict['curve/middle_point_l2_norm'] = middle_point_l2_norm
            log_dict['curve/interpolated_l2_norm'] = interpolated_l2_norm

        if args.split_test_from_train and 'val' in loaders:
            log_dict.update({
                'val/nll': val_res['nll'] if val_res['nll'] is not None else 0,
                'val/accuracy': val_res['accuracy'] if val_res['accuracy'] is not None else 0,
                'val/error': 100.0 - val_res['accuracy'] if val_res['accuracy'] is not None else 100,
                'test/nll': test_res['nll'] if test_res['nll'] is not None else 0,
                'test/accuracy': test_res['accuracy'] if test_res['accuracy'] is not None else 0,
                'test/error': 100.0 - test_res['accuracy'] if test_res['accuracy'] is not None else 100,
            })
            if args.early_stopping:
                log_dict['best_val_error'] = best_val_error
                log_dict['best_epoch'] = best_epoch
        else:
            log_dict.update({
                'test/nll': test_res['nll'] if test_res['nll'] is not None else 0,
                'test/accuracy': test_res['accuracy'] if test_res['accuracy'] is not None else 0,
                'test/error': 100.0 - test_res['accuracy'] if test_res['accuracy'] is not None else 100,
            })
            if args.early_stopping:
                log_dict['best_val_error'] = best_val_error
                log_dict['best_epoch'] = best_epoch

        wandb.log(log_dict, step=epoch)

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    # Check if interrupted at the end of epoch
    if interrupted:
        print(f'\n{"="*70}')
        print(f'Saving checkpoint at epoch {epoch}...')
        utils.save_checkpoint(
            args.dir,
            epoch,
            name='checkpoint-interrupted',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
        print(f'✓ Checkpoint saved to: {args.dir}/checkpoint-interrupted-{epoch}.pt')
        print('Exiting gracefully.')
        print('='*70)
        sys.exit(0)

    # Break if early stopping triggered
    if early_stopped:
        break

# Save final checkpoint if not already saved
final_epoch = epoch if early_stopped else args.epochs
if final_epoch % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        final_epoch,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

# Log early stopping info to wandb
if use_wandb and args.early_stopping:
    wandb.run.summary['early_stopped'] = early_stopped
    wandb.run.summary['stopped_at_epoch'] = final_epoch
    wandb.run.summary['best_val_error'] = best_val_error
    wandb.run.summary['best_epoch'] = best_epoch

# Save L2 norm history for curve models
if args.curve is not None and len(middle_point_l2_norms) > 0:
    import numpy as np
    # Save to evaluations directory (create if needed)
    eval_dir = args.dir.replace('/checkpoints', '/evaluations')
    os.makedirs(eval_dir, exist_ok=True)
    l2_norm_file = os.path.join(eval_dir, 'middle_point_l2_norms.npz')
    np.savez(
        l2_norm_file,
        epochs=np.array(epochs_list),
        l2_norms=np.array(middle_point_l2_norms),
        interpolated_l2_norms=np.array(interpolated_l2_norms)
    )
    print(f'\nSaved middle point L2 norm history to: {l2_norm_file}')
