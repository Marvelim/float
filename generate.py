"""
	Inference Stage 2
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions

# 导入工具函数
from utils.checkpoint_utils import load_weight
from utils.video_utils import save_generated_video
from utils.data_utils import get_audio_preprocessor, load_audio
from utils.image_utils import ImageProcessor


class DataProcessor:
	def __init__(self, opt):
		self.opt = opt
		self.fps = opt.fps
		self.sampling_rate = opt.sampling_rate
		self.input_size = opt.input_size

		# 使用统一的图像处理器
		self.image_processor = ImageProcessor(
			input_size=opt.input_size,
			use_face_alignment=True,
			device='auto'
		)

		# wav2vec2 audio preprocessor
		self.wav2vec_preprocessor = get_audio_preprocessor(opt)

		# image transform
		self.transform = A.Compose([
				A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
				A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
				A_pytorch.ToTensorV2(),
			])

	@torch.no_grad()
	def process_img(self, img:np.ndarray) -> np.ndarray:
		# 使用统一的图像处理器
		return self.image_processor.process_image(img)

	def default_img_loader(self, path) -> np.ndarray:
		img = cv2.imread(path)
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def default_aud_loader(self, path: str) -> torch.Tensor:
		# 使用 utils 中的 load_audio 函数
		audio_tensor = load_audio(path, self.opt)
		# 确保形状是 (1, sequence_length)
		if audio_tensor.dim() == 1:
			audio_tensor = audio_tensor.unsqueeze(0)
		return audio_tensor


	def preprocess(self, ref_path:str, audio_path:str, no_crop:bool) -> dict:
		print(f"ref_path: {ref_path}")
		print(f"audio_path: {audio_path}")
		print(f"no_crop: {no_crop}")
		s = self.default_img_loader(ref_path)
		if not no_crop:
			s = self.process_img(s)
		print("process_img done")
		s = self.transform(image=s)['image'].unsqueeze(0)
		a = self.default_aud_loader(audio_path)
		print("default_aud_loader done")
		print("preprocess done")
		return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
	def __init__(self, opt):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = opt.rank

		# Load Model
		self.load_model()
		self.load_weight(opt.ckpt_path, rank=self.rank)
		self.G.to(self.rank)
		self.G.eval()

		# Load Data Processor
		self.data_processor = DataProcessor(opt)

	def load_model(self) -> None:
		self.G = FLOAT(self.opt)

	def load_weight(self, checkpoint_path: str, rank: int) -> None:
		# 使用 utils 中的 load_weight 函数
		load_weight(self.G, checkpoint_path, torch.device(rank))

	def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
		# 使用 utils 中的 save_generated_video 函数
		save_generated_video(vid_target_recon, video_path, audio_path, self.opt.fps)
		return video_path

	@torch.no_grad()
	def run_inference(
		self,
		res_video_path: str,
		ref_path: str,
		audio_path: str,
		a_cfg_scale: float	= 2.0,
		r_cfg_scale: float	= 1.0,
		e_cfg_scale: float	= 1.0,
		emo: str 			= 'S2E',
		nfe: int			= 10,
		no_crop: bool 		= False,
		seed: int			= 25,
		verbose: bool 		= True
	) -> str:

		data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)
		if verbose: print(f"> [Done] Preprocess.")

		# inference
		d_hat = self.G.inference(
			data 		= data,
			a_cfg_scale = a_cfg_scale,
			r_cfg_scale = r_cfg_scale,
			e_cfg_scale = e_cfg_scale,
			emo 		= emo,
			nfe			= nfe,
			seed		= seed
			)['d_hat']

		res_video_path = self.save_video(d_hat, res_video_path, audio_path)
		if verbose: print(f"> [Done] result saved at {res_video_path}")
		return res_video_path


class InferenceOptions(BaseOptions):
	def __init__(self):
		super().__init__()

	def initialize(self, parser):
		super().initialize(parser)
		parser.add_argument("--ref_path",
				default=None, type=str,help='ref')
		parser.add_argument('--aud_path',
				default=None, type=str, help='audio')
		parser.add_argument('--emo',
				default=None, type=str, help='emotion', choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
		parser.add_argument('--no_crop',
				action = 'store_true', help = 'not using crop')
		parser.add_argument('--res_video_path',
				default=None, type=str, help='res video path')
		parser.add_argument('--ckpt_path',
				default="./checkpoints/float.pth", type=str, help='checkpoint path')
		parser.add_argument('--res_dir',
				default="./results", type=str, help='result dir')
		return parser


if __name__ == '__main__':
	opt = InferenceOptions().parse()
	opt.rank, opt.ngpus  = 0,1
	agent = InferenceAgent(opt)
	os.makedirs(opt.res_dir, exist_ok = True)

	# -------------- input -------------
	ref_path 		= opt.ref_path
	aud_path 		= opt.aud_path
	# ----------------------------------

	if opt.res_video_path is None:
		video_name = os.path.splitext(os.path.basename(ref_path))[0]
		audio_name = os.path.splitext(os.path.basename(aud_path))[0]
		call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
		res_video_path = os.path.join(opt.res_dir, "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s.mp4" \
									% (call_time, video_name, audio_name, opt.nfe, opt.seed, opt.a_cfg_scale, opt.e_cfg_scale, opt.emo))
	else:
		res_video_path = opt.res_video_path

	agent.run_inference(
		res_video_path,
		ref_path,
		aud_path,
		a_cfg_scale = opt.a_cfg_scale,
		r_cfg_scale = opt.r_cfg_scale,
		e_cfg_scale = opt.e_cfg_scale,
		emo 		= opt.emo,
		nfe			= opt.nfe,
		no_crop 	= opt.no_crop,
		seed 		= opt.seed
		)

