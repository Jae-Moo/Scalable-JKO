import argparse
import torch
import torch.nn.functional as F
import os
import json
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision
from datetime import datetime
from models import get_model
from utils import *
from dataset import build_boundary_distribution, build_prior_test_sampler
from pytorch_fid.fid_score import calculate_fid_given_paths


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:0')
    batch_size = args.batch_size
    nz = args.nz

    # Create log path
    exp = args.exp
    parent_dir = "./train_logs/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    os.makedirs(exp_path, exist_ok=True)
    
    # Make log file
    with open(os.path.join(exp_path, 'log.txt'), 'w') as f:
        f.write("Start Training")
        f.write('\n')


    # Get Q
    if args.model_name == 'otm':
        Q = lambda x: F.interpolate(x.reshape(-1, 3, 8, 8), args.image_size, mode='bicubic').detach()
    else:
        Q = lambda x: x


    # If image, get FID path & statistics
    if args.fid:
        FID_img_path = os.path.join(exp_path, 'generated_samples')
        os.makedirs(FID_img_path, exist_ok=True) 
        if args.dataset == 'cifar10' or args.dataset == 'cifar10+mnist':
            real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
        elif args.dataset == 'celeba64':
            real_img_dir = 'pytorch_fid/celeba_64_jpg_stat.npy'
        elif args.dataset == 'celeba_256':
            real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
        elif args.dataset == 'lsun':
            real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
        else:
            args.fid = False
        fid_prev = 10000

    
    # Get Networks/Optimizer
    netD, netG = get_model(args)
    
    netG = netG.to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    if args.lr_scheduler:
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.init_num_iterations + args.num_iterations * args.num_phase, eta_min=args.eta_min)
    
    netG = nn.DataParallel(netG)

    netD = netD.to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    if args.lr_scheduler:
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.init_num_iterations + args.num_iterations * args.num_phase, eta_min=args.eta_min)
    
    netD = nn.DataParallel(netD)

    # Resume
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_iteration = checkpoint['iteration']
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        if args.lr_scheduler:
            schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        if args.lr_scheduler:
            schedulerD.load_state_dict(checkpoint['schedulerD'])
        print("=> loaded checkpoint (iteration {})".format(checkpoint['iteration']))
    else:
        init_iteration = 0
    
    
    # Get Data
    data_sampler, prior_sampler = build_boundary_distribution(args)
    prior_test_sampler = build_prior_test_sampler(args)
    
    def cost(x, y):
        return torch.mean(((x-y).view(x.size(0), -1))**2, dim=1)

    # save configurations
    jsonstr = json.dumps(args.__dict__, indent=4)
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        f.write(jsonstr)

    # Start training
    start = datetime.now()
    iter = 0
    eval_noise = prior_test_sampler.sample().to(device)
    eval_z = torch.randn(batch_size, nz, device=device)

    for phase in range(args.num_phase):
        if phase == 0:
            def netG_old(x, z):
                return Q(x)
            num_iterations = args.init_num_iterations
            h = args.init_h
        else:
            _, netG_old = get_model(args)
            netG_old = netG_old.to(device)
            netG_old = nn.DataParallel(netG_old)
            netG_old.load_state_dict(netG.state_dict())
            num_iterations = args.num_iterations
            h = args.h
            
        
        for i in range(num_iterations):
            iter += 1

            #### Update potential ####
            for p in netD.parameters():  
                p.requires_grad = True

            netD.zero_grad()

            real_data = data_sampler.sample().to(device)
            real_data.requires_grad = True
            noise = prior_sampler.sample().to(device)

            # Real D loss
            D_real = netD(real_data + (1/(32*(phase+1)))*torch.randn_like(real_data))
            errD_real = F.softplus(-D_real) 
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)
            
            # R1 regularization
            if args.reg_name == 'r1':
                grad_real = torch.autograd.grad(outputs=D_real.sum(), inputs=real_data, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.lmbda / 2 * grad_penalty
                grad_penalty.backward()

            # fake D loss
            latent_z = torch.randn(batch_size, nz, device=device)
            fake_data = netG(noise, latent_z)
            D_fake = netD(fake_data)
            
            errD_fake = F.softplus(D_fake)
            errD_fake = errD_fake.mean()
            errD_fake.backward()
            
            errD = errD_real + errD_fake

            if args.clip > 0:
                nn.utils.clip_grad_norm_(netD.parameters(), args.clip)
            
            optimizerD.step()

            #### Update Generator ####
            for p in netD.parameters():
                p.requires_grad = False
            
            netG.zero_grad()

            # Generator loss
            noise = prior_sampler.sample().to(device)
            latent_z = torch.randn(batch_size, nz, device=device)
            with torch.no_grad():
                fake_data_old = netG_old(noise, latent_z)
            latent_z = torch.randn(batch_size, nz, device=device)
            fake_data = netG(noise, latent_z)
            D_fake = netD(fake_data)

            c = cost(fake_data, fake_data_old)
            
            err = 0.5 * c / h + F.softplus(-D_fake) # GAN!
            err = err.mean()
            err.backward()
            optimizerG.step()
        
            #### Update Schedulers
            if args.lr_scheduler:
                schedulerG.step()
                schedulerD.step()


            #### Visualizations and Save ####
            ## save losses
            if (iter + 1) % args.print_every == 0:
                with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                    f.write(f'Iteration {iter + 1:07d} : G Loss {err.item():.4f}, D Loss {errD.item():.4f}, Elapsed {datetime.now() - start}')
                    f.write('\n')
            
            # save content
            if (iter + 1) % args.save_content_every == 0:
                print('Saving content.')
                if args.lr_scheduler:
                    content = {'iteration': iter + 1, 'args': args,
                                'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                else:
                    content = {'iteration': iter + 1, 'args': args,
                                'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                'netD_dict': netD.state_dict(), 'optimizerD': optimizerD.state_dict()}
                
                torch.save(content, os.path.join(exp_path, 'content.pth'))
            
            # save checkpoint
            if (iter + 1) % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(iter + 1)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                
                torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_{}.pth'.format(iter + 1)))

            # save generated images
            if (iter + 1) % args.save_image_every == 0:
                if args.model_name == 'toy':
                    NUM_SAMPLES = 5000
                    REAL_SAMPLES = [data_sampler.sample() for i in range(NUM_SAMPLES//batch_size)]
                    REAL_SAMPLES = np.concatenate(REAL_SAMPLES)

                    # fake data samples
                    FAKE_SAMPLES = []
                    NOISES = []

                    with torch.no_grad():
                        for _ in range(NUM_SAMPLES//batch_size):
                            noise = prior_test_sampler.sample().to(device)
                            latent_z = torch.randn(batch_size, nz, device=device)
                            FAKE_SAMPLES.append(netG(noise, latent_z).cpu().numpy())
                            NOISES.append(noise.cpu().numpy())
                    FAKE_SAMPLES = np.concatenate(FAKE_SAMPLES)
                    NOISES = np.concatenate(NOISES)                

                    np.savez(os.path.join(exp_path, 'iter_{}.npz'.format(iter + 1)),{'real': REAL_SAMPLES, 'fake': FAKE_SAMPLES, 'noise': NOISES})

                else:
                    with torch.no_grad():
                        images = netG(eval_noise, eval_z)
                        # images_old = netG_old(noise, eval_z)
                    images = (0.5*(images+1)).detach().cpu()
                    # images_old = (0.5*(images_old+1)).detach().cpu()
                    # torchvision.utils.save_image(images_old, os.path.join(exp_path, '{}_old.png'.format(iter + 1)))
                    torchvision.utils.save_image(images, os.path.join(exp_path, '{}_target.png'.format(iter + 1)))
                    
                    
                    if args.dataset == 'emnist2mnist':
                        noise = (0.5*(noise+1)).detach().cpu()
                        torchvision.utils.save_image(noise, os.path.join(exp_path, '{}_prior.png'.format(iter + 1)))

            # calculate fid
            if (iter + 1) % args.fid_every == 0 and args.fid:
                # use ema model
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                
                iters_needed = 50000 // batch_size
                
                for i in range(iters_needed):
                    with torch.no_grad():
                        noise = prior_test_sampler.sample().to(device)
                        latent_z = torch.randn(batch_size, nz, device=device)
                        fake_sample = netG(noise, latent_z)
                        fake_sample = (fake_sample + 1.) / 2.
                        
                        for j, x in enumerate(fake_sample):
                            index = i * args.batch_size + j 
                            torchvision.utils.save_image(x, os.path.join(exp_path,'generated_samples/{}.jpg'.format(index)))
                        
                        print('generating batch ', i, end='\r')
            
                paths = [FID_img_path, real_img_dir]
            
                kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
                fid = calculate_fid_given_paths(paths=paths, **kwargs)
                print(fid)
                with open(os.path.join(exp_path, 'log.txt'), 'a') as f:
                    f.write(f'Iteration {iter + 1:04d} FID : {fid}')
                    f.write('\n')
                
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                
                # if fid > fid_prev:
                #     assert 0
                # else:
                #     fid_prev = fid
                fid_prev = fid

if __name__ == '__main__':
    parser = argparse.ArgumentParser('UOTM parameters')
    
    # Experiment description
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--exp', default='WGF_temp', help='name of the experiment')
    parser.add_argument('--resume', action='store_true',default=False, help='Resume training or not')
    parser.add_argument('--dataset', default='cifar10', choices=['checkerboard', 
                                                                 '8gaussian', 
                                                                 '25gaussian', 
                                                                 'spiral', 
                                                                 'mnist', 
                                                                 'cifar10', 
                                                                 'celeba64', 
                                                                 'lsun', 
                                                                 'celeba_256',
                                                                 'moon2spiral',
                                                                 'emnist2mnist'], help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32, help='size of image (or data)')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
    # Network configurations
    parser.add_argument('--model_name', default='ncsnpp', choices=['ncsnpp', 'ddpm', 'drunet', 'otm', 'toy'], help='Choose default model')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
    parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
    parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
    parser.add_argument('--nz', type=int, default=100, help='latent dimension')
    parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
    
    # Training/Optimizer configurations
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--lr_g', type=float, default=2.0e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.0e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--clip', type=float, default=0, help='Clip the gradient if the clip value is positive (>0)')
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help='Use lr scheduler or not. We use cosine scheduler if the argument is activated.')
    parser.add_argument('--eta_min', type=float, default=5e-5, help='eta_min of lr_scheduler')
    
    # Loss configurations
    parser.add_argument('--cost', type=str, default='w2', choices=['w1', 'w2'], help='Wasserstein1(w1) and Wasserstein2(w2) Cost')
    parser.add_argument('--reg_name', type=str, default='r1', choices=['none', 'gp', 'r1'], help='Set regularization, GP/R1')
    parser.add_argument('--lmbda', type=float, default=0.2, help='coef for regularization')  
    
    # (ADD) phase and iters
    parser.add_argument('--init_num_iterations', type=int, default=10000)
    parser.add_argument('--num_iterations', type=int, default=2000, help='the number of iterations')
    parser.add_argument('--init_h', type=float, default=0.1)
    parser.add_argument('--h', type=float, default=0.1, help='proportion of the cost c')
    parser.add_argument('--num_phase', type=int, default=50)
    parser.add_argument('--init_noise', type=float, default=0.03125, help='Add noise to real data for warmup. Noise scale is scheduled every phase. Linear schedule, init_noise->1/256')
    parser.add_argument('--uniform_quantization', action='store_true', default=False)

    # Visualize/Save configurations
    parser.add_argument('--print_every', type=int, default=100, help='print current loss for every x iterations')
    parser.add_argument('--save_content_every', type=int, default=2000, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10000, help='save ckpt every x epochs')
    parser.add_argument('--save_image_every', type=int, default=2000, help='save images every x epochs')
    parser.add_argument('--fid_every', type=int, default=10000, help='monitor FID every x epochs')
    parser.add_argument('--fid', action='store_false', default=True, help="Calculate FID")
    args = parser.parse_args()

    if args.model_name == 'toy': args.fid = False
    train(args)
