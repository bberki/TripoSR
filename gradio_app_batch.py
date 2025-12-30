import logging
import os
import tempfile
import time
import zipfile
import io

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial
from pathlib import Path

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

# Import Batch processor
from batch_processor import BatchProcessor

import argparse


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


# =============================================================================
# SINGLE IMAGE PROCESSING FUNCTIONS (EXISTING)
# =============================================================================

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj, mesh_name_glb = generate(preprocessed, 256, ["obj", "glb"])
    return preprocessed, mesh_name_obj, mesh_name_glb


# =============================================================================
# BATCH PROCESSING FUNCTIONS (NEW)
# =============================================================================

def process_batch_gradio(
    files,
    do_remove_background,
    foreground_ratio,
    mc_resolution,
    output_format
):
    """
    Batch processing from Gradio interface
    
    Args:
        files: Files from Gradio File component
        do_remove_background: Remove background flag
        foreground_ratio: Foreground ratio
        mc_resolution: Marching cubes resolution
        output_format: Output format
        
    Returns:
        (zip_file, result_text): ZIP file and result text
    """
    if files is None or len(files) == 0:
        raise gr.Error("Please upload at least one image!")
    
    # File count check
    if len(files) > 50:
        raise gr.Error(f"You can upload a maximum of 50 images! ({len(files)} uploaded)")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "inputs"
        input_dir.mkdir()
        
        # Save files to temp directory
        image_paths = []
        for idx, file in enumerate(files):
            # Gradio File format: contains file.name
            file_name = Path(file.name).name if hasattr(file, 'name') else f"input_{idx}.png"
            file_path = input_dir / file_name
            
            # Copy file
            if hasattr(file, 'name'):
                # File path
                import shutil
                shutil.copy(file.name, file_path)
            else:
                # Bytes
                file_path.write_bytes(file)
            
            image_paths.append(str(file_path))
        
        # Create Batch processor
        processor = BatchProcessor(
            model=model,
            output_dir=temp_path / "outputs",
            device=device
        )
        
        # Process batch
        try:
            results = processor.process_batch(
                image_paths=image_paths,
                output_format=output_format,
                do_remove_background=do_remove_background,
                foreground_ratio=foreground_ratio,
                mc_resolution=mc_resolution,
                save_processed_images=False
            )
        except Exception as e:
            raise gr.Error(f"Batch processing error: {str(e)}")
        
        # Prepare results as ZIP
        zip_buffer = io.BytesIO()
        
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                batch_dir = Path(results['batch_dir'])
                
                # Add all successful mesh files to zip
                for result in results['results']:
                    if result['status'] == 'success':
                        mesh_path = Path(result['mesh_path'])
                        if mesh_path.exists():
                            # Organized structure inside ZIP
                            folder_name = mesh_path.parent.name
                            arcname = f"{folder_name}/{mesh_path.name}"
                            zip_file.write(mesh_path, arcname=arcname)
                            
                            # Add metadata if exists
                            metadata_path = mesh_path.parent / "metadata.json"
                            if metadata_path.exists():
                                zip_file.write(metadata_path, 
                                               arcname=f"{folder_name}/metadata.json")
                
                # Add batch report
                report_path = batch_dir / "batch_report.json"
                if report_path.exists():
                    zip_file.write(report_path, arcname="batch_report.json")
                
                # Add summary report
                summary_path = batch_dir / "SUMMARY.txt"
                if summary_path.exists():
                    zip_file.write(summary_path, arcname="SUMMARY.txt")
            
            zip_buffer.seek(0)
            
        except Exception as e:
            raise gr.Error(f"ZIP creation error: {str(e)}")
        
        # Create result message
        result_lines = [
            "="*60,
            "BATCH PROCESSING COMPLETED",
            "="*60,
            f"Total        : {results['total']} images",
            f"Successful   : {results['successful']} ({results['success_rate']})",
            f"Failed       : {results['failed']}",
            f"Total Time   : {results['total_time_sec']:.1f} seconds",
            f"Average      : {results['avg_time_per_image_sec']:.1f} seconds/image",
            "="*60,
            "",
            "Download the ZIP file and use the meshes inside!",
        ]
        
        # List failed ones
        if results['failed'] > 0:
            result_lines.append("\nFailed Processes:")
            for result in results['results']:
                if result['status'] != 'success':
                    result_lines.append(f" - {result['filename']}: {result.get('error', 'Unknown')}")
        
        result_text = "\n".join(result_lines)
        
        return zip_buffer.getvalue(), result_text


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="TripoSR - Single & Batch") as interface:
    gr.Markdown(
        """
    # TripoSR Demo - Single & Batch Processing
    
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image.
    
    **New Feature:** You can now process multiple images in batches!
    """
    )
    import torch
    
    def get_gpu_info():
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f" **GPU Aktif:** {gpu_name} ({vram:.1f} GB VRAM)"
        else:
            return " **CPU Modu:** Sistem sadece işlemci kullanıyor (Yavaş olabilir)"

    with gr.Row():
        gr.Markdown(get_gpu_info())
    
    with gr.Tabs():
        # =================================================================
        # TAB 1: SINGLE IMAGE PROCESSING (EXISTING)
        # =================================================================
        with gr.TabItem("Single Image", id=0):
            gr.Markdown("""
            ### 3D Model Generation from Single Image
            Upload an image and generate a 3D model.
            """)
            
            with gr.Row(variant="panel"):
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(
                            label="Input Image",
                            image_mode="RGBA",
                            sources="upload",
                            type="pil",
                            elem_id="content_image",
                        )
                        processed_image = gr.Image(
                            label="Processed Image", 
                            interactive=False
                        )
                    with gr.Row():
                        with gr.Group():
                            do_remove_background = gr.Checkbox(
                                label="Remove Background", value=True
                            )
                            foreground_ratio = gr.Slider(
                                label="Foreground Ratio",
                                minimum=0.5,
                                maximum=1.0,
                                value=0.85,
                                step=0.05,
                            )
                            mc_resolution = gr.Slider(
                                label="Marching Cubes Resolution",
                                minimum=32,
                                maximum=320,
                                value=256,
                                step=32
                            )
                    with gr.Row():
                        submit = gr.Button(
                            "Generate 3D Model", 
                            elem_id="generate", 
                            variant="primary"
                        )
                        
                with gr.Column():
                    with gr.Tab("OBJ"):
                        output_model_obj = gr.Model3D(
                            label="Output Model (OBJ Format)",
                            interactive=False,
                        )
                        gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
                    with gr.Tab("GLB"):
                        output_model_glb = gr.Model3D(
                            label="Output Model (GLB Format)",
                            interactive=False,
                        )
                        gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
            
            # Examples
            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        "examples/hamburger.png",
                        "examples/poly_fox.png",
                        "examples/robot.png",
                        "examples/teapot.png",
                        "examples/tiger_girl.png",
                    ],
                    inputs=[input_image],
                    outputs=[processed_image, output_model_obj, output_model_glb],
                    cache_examples=False,
                    fn=partial(run_example),
                    label="Examples",
                    examples_per_page=20,
                )
            
            # Event handlers
            submit.click(
                fn=check_input_image, 
                inputs=[input_image]
            ).success(
                fn=preprocess,
                inputs=[input_image, do_remove_background, foreground_ratio],
                outputs=[processed_image],
            ).success(
                fn=generate,
                inputs=[processed_image, mc_resolution],
                outputs=[output_model_obj, output_model_glb],
            )
        
        # =================================================================
        # TAB 2: BATCH PROCESSING (NEW)
        # =================================================================
        with gr.TabItem("Batch Processing", id=1):
            gr.Markdown("""
            ### Batch Processing
            
            Process multiple images at once and download as ZIP.
            
            **How to Use:**
            1. Upload multiple images (PNG, JPG, JPEG, BMP, WEBP)
            2. Configure processing settings
            3. Click "Process Batch" button
            4. Wait for the process to complete
            5. Download the ZIP file
            
            **Limits:**
            - Maximum 50 images
            - A separate folder is created for each image
            - Failed processes are reported
            
            **Tips:**
            - High resolution = Better quality + Slower processing
            - Background removal takes more time
            - Use lower resolution if GPU memory is limited
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload
                    batch_files = gr.File(
                        label="Upload Multiple Images (Max 50)",
                        file_count="multiple",
                        file_types=["image"],
                        height=200
                    )
                    
                    gr.Markdown("---")
                    
                    # Settings
                    with gr.Group():
                        gr.Markdown("### Processing Settings")
                        
                        batch_remove_bg = gr.Checkbox(
                            label="Remove Background",
                            value=True,
                            info="Automatically remove background (U2-Net model)"
                        )
                        
                        batch_foreground_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                            info="Object ratio in the image"
                        )
                        
                        batch_mc_resolution = gr.Slider(
                            label="Marching Cubes Resolution",
                            minimum=32,
                            maximum=320,
                            value=256,
                            step=32,
                            info="Higher = More detailed + Slower processing"
                        )
                        
                        batch_format = gr.Radio(
                            label="Output Format",
                            choices=["obj", "glb"],
                            value="obj",
                            info="Mesh file format"
                        )
                    
                    gr.Markdown("---")
                    
                    # Process button
                    batch_submit = gr.Button(
                        "Process Batch",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown("""
                    **Attention:**
                    - Estimated time: (Number of images x 5-10 seconds)
                    - Do not close the page during processing!
                    """)
                
                with gr.Column(scale=1):
                    # Results
                    gr.Markdown("### Processing Results")
                    
                    batch_result_text = gr.Textbox(
                        label="Status",
                        lines=15,
                        interactive=False,
                        placeholder="Results will appear here when processing starts..."
                    )
                    
                    batch_output_file = gr.File(
                        label="Download Results (ZIP)",
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    ### ZIP Content
                    ```
                    results.zip
                    ├── 001_image1/
                    │   ├── mesh.obj
                    │   └── metadata.json
                    ├── 002_image2/
                    │   ├── mesh.obj
                    │   └── metadata.json
                    ├── batch_report.json
                    └── SUMMARY.txt
                    ```
                    """)
            
            # Event handler
            batch_submit.click(
                fn=process_batch_gradio,
                inputs=[
                    batch_files,
                    batch_remove_bg,
                    batch_foreground_ratio,
                    batch_mc_resolution,
                    batch_format
                ],
                outputs=[batch_output_file, batch_result_text]
            )


# =============================================================================
# APP LAUNCH
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )