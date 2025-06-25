from cog import Path, Input, BasePredictor
import os
import time
import torch
import random
import subprocess
from PIL import Image
from torchvision.transforms.functional import to_tensor
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.utils.img_util import create_collage
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/OmniGen2/OmniGen2/model.tar"

def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Set environment variable for flash attention
        os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = "TRUE"
        
        # Download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.pipeline = OmniGen2Pipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=self.dtype
        )
        self.pipeline = self.pipeline.to(self.device)

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt describing the desired image edit",
            default="Make the person smile"
        ),
        image: Path = Input(
            description="Input image to edit", default=None
        ),
        image_2: Path = Input(
            description="Optional second input image for multi-image operations",
            default=None
        ),
        image_3: Path = Input(
            description="Optional third input image for multi-image operations", 
            default=None
        ),
        negative_prompt: str = Input(
            description="Negative prompt to guide what should not be in the image",
            default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
            ge=256,
            le=1024
        ),
        height: int = Input(
            description="Height of output image", 
            default=1024,
            ge=256,
            le=1024
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            default=50,
            ge=20,
            le=100
        ),
        text_guidance_scale: float = Input(
            description="Guidance scale for text prompt",
            default=5.0,
            ge=1.0,
            le=8.0
        ),
        image_guidance_scale: float = Input(
            description="Guidance scale for input image. Higher values increase consistency with input image",
            default=2.0,
            ge=1.0,
            le=3.0
        ),
        cfg_range_start: float = Input(
            description="CFG range start", 
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        cfg_range_end: float = Input(
            description="CFG range end",
            default=1.0, 
            ge=0.0,
            le=1.0
        ),
        scheduler: str = Input(
            description="Scheduler to use",
            choices=["euler", "dpmsolver"],
            default="euler"
        ),
        max_input_image_side_length: int = Input(
            description="Maximum input image side length",
            default=2048,
            ge=256,
            le=2048
        ),
        max_pixels: int = Input(
            description="Maximum number of pixels in output",
            default=1048576,  # 1024*1024
            ge=65536,        # 256*256
            le=2359296       # 1536*1536
        ),
        seed: int = Input(
            description="Random seed. Set to -1 for random seed",
            default=-1
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # Handle seed
        if seed == -1:
            seed = random.randint(0, 2**16 - 1)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Load and prepare input images
        input_images = []
        if image:
            input_images.append(Image.open(image).convert("RGB"))
        if image_2:
            input_images.append(Image.open(image_2).convert("RGB"))
        if image_3:
            input_images.append(Image.open(image_3).convert("RGB"))
            
        if len(input_images) == 0:
            input_images = None
        
        # Set scheduler
        if scheduler == 'euler':
            self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
        elif scheduler == 'dpmsolver':
            self.pipeline.scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint", 
                solver_order=2,
                prediction_type="flow_prediction",
            )
        
        # Ensure cfg_range_end >= cfg_range_start
        cfg_range_end = max(cfg_range_start, cfg_range_end)
        
        # Generate image
        results = self.pipeline(
            prompt=prompt,
            input_images=input_images,
            width=width,
            height=height,
            max_input_image_side_length=max_input_image_side_length,
            max_pixels=max_pixels,
            num_inference_steps=num_inference_steps,
            max_sequence_length=1024,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        
        # Handle multiple images output
        if len(results.images) == 1:
            output_image = results.images[0]
        else:
            # Create collage for multiple images
            vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
            output_image = create_collage(vis_images)
        
        # Save output
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        
        return Path(output_path)
