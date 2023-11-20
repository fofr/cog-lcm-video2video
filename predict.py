import cv2 as cv
import os
import torch
import subprocess
import glob
import tarfile
import numpy as np
import time
from typing import Optional, List
from diffusers import ControlNetModel, AutoPipelineForImage2Image
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet
from cog import BasePredictor, Input, Path
from PIL import Image

MODEL_CACHE_URL = (
    "https://weights.replicate.delivery/default/fofr-lcm/lcm-sd15-ds7-canny-qr.tar"
)
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def create_pipeline(
        self,
        pipeline_class,
        controlnet: Optional[ControlNetModel] = None,
    ):
        kwargs = {
            "cache_dir": MODEL_CACHE,
            "local_files_only": True,
            "safety_checker": None,
        }

        if controlnet:
            kwargs["controlnet"] = controlnet
            kwargs["scheduler"] = None

        pipe = pipeline_class.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", **kwargs)
        pipe.to(torch_device="cuda", torch_dtype=torch.float16)
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_CACHE_URL, MODEL_CACHE)

        torch_device = "cuda"
        torch_dtype = torch.float16

        self.img2img_pipe = self.create_pipeline(
            AutoPipelineForImage2Image,
        )

        self.controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            cache_dir="model_cache",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(torch_device)

        self.controlnet_monster = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            cache_dir="model_cache",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(torch_device)

    def extract_frames(self, video, fps, extract_all_frames):
        os.makedirs("/tmp", exist_ok=True)

        if not extract_all_frames:
            command = f'ffmpeg -i "{video}" -vf fps={fps} /tmp/out%03d.png'
        else:
            command = f'ffmpeg -i "{video}" /tmp/out%03d.png'

        subprocess.run(command, shell=True, check=True)
        frame_files = sorted(os.listdir("/tmp"))
        frame_files = [
            file for file in frame_files if file.endswith(".png") and "out" in file
        ]

        print(f"Extracted {len(frame_files)} frames from video")
        return [f"/tmp/{frame_file}" for frame_file in frame_files]

    def width_height(self, frame_paths):
        img = Image.open(frame_paths[0])
        width, height = img.size
        return width, height

    def resize_frames(self, frame_paths, max_width):
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            width, height = img.size
            if width > max_width:
                height = int(height * max_width / width)
                width = max_width
                img = img.resize((width, height))
                img.save(frame_path)

    def images_to_video(self, image_folder_path, output_video_path, fps, prefix="out"):
        # Forming the ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(fps),  # Set the framerate for the input files
            "-pattern_type",
            "glob",  # Enable pattern matching for filenames
            "-i",
            f"{image_folder_path}/{prefix}*.jpg",  # Input files pattern
            "-c:v",
            "libx264",  # Set the codec for video
            "-pix_fmt",
            "yuv420p",  # Set the pixel format
            "-crf",
            "17",  # Set the constant rate factor for quality
            output_video_path,  # Output file
        ]

        # Run the ffmpeg command
        subprocess.run(cmd)

    def tar_frames(self, frame_paths, tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            for frame in frame_paths:
                tar.add(frame)

    def control_image(self, image, canny_low_threshold, canny_high_threshold):
        image = np.array(image)
        canny = cv.Canny(image, canny_low_threshold, canny_high_threshold)
        return Image.fromarray(canny)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for video2video",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        video: Path = Input(description="Video to split into frames"),
        fps: int = Input(
            description="Number of images per second of video, when not exporting all frames",
            default=8,
            ge=1,
        ),
        extract_all_frames: bool = Input(
            description="Get every frame of the video. Ignores fps. Slow for large videos.",
            default=False,
        ),
        max_width: int = Input(
            description="Maximum width of the video. Maintains aspect ratio.",
            default=512,
            ge=1,
        ),
        prompt_strength: float = Input(
            description="1.0 corresponds to full destruction of information in video frame",
            ge=0.0,
            le=1.0,
            default=0.2,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps per frame. Recommend 1 to 8 steps.",
            ge=1,
            le=50,
            default=4,
        ),
        controlnet: str = Input(
            description="Controlnet to use",
            choices=["none", "canny", "illusion"],
            default="none",
        ),
        controlnet_conditioning_scale: float = Input(
            description="Controlnet conditioning scale",
            ge=0.1,
            le=4.0,
            default=2.0,
        ),
        control_guidance_start: float = Input(
            description="Controlnet start",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        control_guidance_end: float = Input(
            description="Controlnet end",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        canny_low_threshold: float = Input(
            description="Canny low threshold",
            ge=1,
            le=255,
            default=100,
        ),
        canny_high_threshold: float = Input(
            description="Canny high threshold",
            ge=1,
            le=255,
            default=200,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=8.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        return_frames: bool = Input(
            description="Return a tar file with all the frames alongside the video",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        prediction_start = time.time()

        # Removing all temporary frames
        tmp_frames = glob.glob("/tmp/out*.*")
        tmp_control_images = glob.glob("/tmp/control*.*")
        for frame in tmp_frames + tmp_control_images:
            os.remove(frame)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")

        # Extract frames from video
        extract_frames_start = time.time()
        print(f"Extracting frames from video: {video}")
        frame_paths = self.extract_frames(video, fps, extract_all_frames)
        print(f"Extracting frames took: {time.time() - extract_frames_start:.2f}s")

        # Resize frames
        resize_frames_start = time.time()
        print(f"Resizing frames to max width: {max_width}")
        self.resize_frames(frame_paths, max_width)
        print(f"Resizing frames took: {time.time() - resize_frames_start:.2f}s")

        width, height = self.width_height(frame_paths)

        img2img_args = {
            "num_inference_steps": num_inference_steps,
            "prompt": prompt,
            "strength": prompt_strength,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "lcm_origin_steps": 50,
            "output_type": "pil",
        }

        controlnet_args = {
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        if controlnet == "none":
            print("Running img2img pipeline on each frame")
        else:
            print("Running controlnet on each frame")
            self.img2img_controlnet_pipe = self.create_pipeline(
                LatentConsistencyModelPipeline_controlnet,
                controlnet=self.controlnet_canny
                if controlnet == "canny"
                else self.controlnet_monster,
            )

        generating_frames_start = time.time()
        for index, frame_path in enumerate(frame_paths):
            frame = Image.open(frame_path)
            img2img_args["image"] = frame

            if controlnet != "none":
                if controlnet == "canny":
                    control_image = self.control_image(
                        frame, canny_low_threshold, canny_high_threshold
                    )
                else:
                    control_image = frame

                img2img_args["control_image"] = control_image
                result = self.img2img_controlnet_pipe(
                    **img2img_args, **controlnet_args, generator=torch.manual_seed(seed)
                ).images
            else:
                result = self.img2img_pipe(
                    **img2img_args, generator=torch.manual_seed(seed)
                ).images

            frame_path_jpg = frame_path.replace(".png", ".jpg")
            print(f"Saving frame {index+1} of {len(frame_paths)}: {frame_path_jpg}")
            result[0].save(frame_path_jpg)

            if controlnet == "canny":
                control_image.save(frame_path_jpg.replace("out", "control"))

        print(f"Generating frames took: {time.time() - generating_frames_start:.2f}s")

        print("Creating video from frames")
        video_start_time = time.time()
        video_path = "/tmp/output_video.mp4"
        self.images_to_video("/tmp", video_path, fps)
        print(f"Video creation took: {time.time() - video_start_time:.2f}s")

        paths = [Path(video_path)]

        if controlnet == "canny":
            control_video_path = "/tmp/control_video.mp4"
            self.images_to_video("/tmp", control_video_path, fps, prefix="control")
            paths.append(Path(control_video_path))

        # Tar and return all the frames if return_frames is True
        if return_frames:
            print("Tarring and returning all frames")
            tar_path = "/tmp/frames.tar.gz"
            self.tar_frames(frame_paths, tar_path)
            paths.append(Path(tar_path))

        print(f"Prediction took: {time.time() - prediction_start:.2f}s")
        return paths
