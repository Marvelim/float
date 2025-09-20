import os, argparse, json

class BaseOptions():
	def parse(self):
		parser = argparse.ArgumentParser()
		self.parser = self.initialize(parser)
		self.opt = self.parser.parse_args()
		return self.opt

	def initialize(self, parser):
		parser.add_argument('--pretrained_dir', type=str, default='/home/mli374/float/checkpoints')
		parser.add_argument('--seed', default=15, type=int)
		parser.add_argument('--fix_noise_seed', action='store_true')

		# video
		parser.add_argument('--input_size', type=int, default=512, help='input image size')
		parser.add_argument('--input_nc', type=int, default=3, help='input image channel')        
		parser.add_argument('--fps', type=float, default=25.)

		# audio
		parser.add_argument('--sampling_rate', type=int, default=16000)
		parser.add_argument('--audio_marcing', type=int, default=2, help='number of adjacent frames. For value v, t -> [t-v, ..., t, ..., t+v]')        
		parser.add_argument('--wav2vec_sec', default=2, type=float, help='window length L (seconds), 50 frames')
		parser.add_argument('--wav2vec_model_path', default='/home/mli374/float/checkpoints/wav2vec2-base-960h')
		parser.add_argument('--audio2emotion_path', default='/home/mli374/float/checkpoints/wav2vec-english-speech-emotion-recognition')
		parser.add_argument('--attention_window', default=2, type=int, help='attention window size, e.g., if 1, attend frames of t-1, t, t+1 for frame t')

		parser.add_argument('--only_last_features', action='store_true')
		parser.add_argument('--average_emotion', action='store_true', help='averaging emotion or not.')

		# dropout
		parser.add_argument('--audio_dropout_prob', default=0.1, type=float)
		parser.add_argument('--ref_dropout_prob', default=0.1, type=float)
		parser.add_argument('--emotion_dropout_prob', default=0.1, type=float)

		# model Hyper Parameters
		parser.add_argument('--style_dim', type=int, default=512, help='w latent dimension')
		parser.add_argument('--dim_a', type=int, default=512, help='audio dimension')
		parser.add_argument('--dim_w', type=int, default=512, help='face dimension')
		parser.add_argument('--dim_h', type=int, default=1024, help='hidden dimension')
		parser.add_argument('--dim_m', type=int, default=20, help='dimension of orthogonal basis')
		parser.add_argument('--dim_e', type=int, default=7, help='emotion dimension')

		# option for FMT
		parser.add_argument('--fmt_depth', default=8, type=int)
		parser.add_argument('--num_heads', default=8, type=int)
		parser.add_argument('--mlp_ratio', default=4.0, type=float)
		parser.add_argument('--no_learned_pe', action='store_true')
		parser.add_argument('--num_prev_frames', type=int, default=10)
		parser.add_argument('--max_grad_norm', default=1, type=float, help='max grad norm for training transformers')
		
		parser.add_argument('--ode_atol', default=1e-5, type=float)
		parser.add_argument('--ode_rtol', default=1e-5, type=float)
		parser.add_argument('--nfe', default=10, type=int,
							help='Number of Function Evaluateions (NFEs) for ODE solver')
		parser.add_argument('--torchdiffeq_ode_method', default='euler',
							help='ODE solver')
		parser.add_argument('--a_cfg_scale', default=2.0, type=float,
							help='audio classifier-free guidance (vector field) scale')
		parser.add_argument('--e_cfg_scale', default=1.0, type=float,
							help='emotion classifier-free guidance (vector field) scale')        
		parser.add_argument('--r_cfg_scale', default=1.0, type=float,
							help='reference classifier-free guidance (vector field) scale')        

		# option for Diffusion (ablation)
		parser.add_argument('--n_diff_steps', type=int, default=500, help='number of diffusion steps')
		parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
		parser.add_argument('--diffusion_mode', type=str, default='sample', choices=['sample', 'noise'])

		parser.add_argument('--data_root', type=str, default='/home/mli374/float/datasets', help='数据根目录')
		parser.add_argument('--steps', type=int, default=200000, help='训练步数')
		parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
		parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
		parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')
		parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help='混合精度')
		parser.add_argument('--log_step', type=int, default=50)
		parser.add_argument('--sample_step', type=int, default=2000)
		parser.add_argument('--save_step', type=int, default=10000)
		parser.add_argument('--prev_frame_dropout_prob', type=float, default=0.5)
		
		# 设备相关参数
		parser.add_argument('--rank', type=str, default='cuda', help='训练设备')
		
		# 训练相关参数
		parser.add_argument('--batch_step', type=int, default=20, help='每个批次的子步骤数')
		
		# wandb 相关参数
		parser.add_argument('--use_wandb', action='store_true', help='是否使用 wandb 进行实验跟踪')
		parser.add_argument('--wandb_project', type=str, default='float-training', help='wandb 项目名称')
		parser.add_argument('--wandb_entity', type=str, default=None, help='wandb 实体名称')
		parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb 运行名称')
		parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='wandb 标签')
		
		return parser


	def print_options(self):
		"""Print and save options

		It will print both current options and default values(if different).
		It will save options into a text file / [checkpoints_dir] / opt.txt
		"""
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(self.opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)


def save_options(opt, save_path):
	with open(save_path, 'wt') as f:
		json.dump(vars(opt), f, indent=4)


def load_options(opt, load_path):
	with open(load_path, 'rt') as f:
		_update = json.loads(f)
	opt.update(_update)
	return opt

