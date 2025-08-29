import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
from PIL import Image, ImageDraw, ImageFont

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json # <<< MODIFICATION: Import for loading GT data
from detectron2.utils.visualizer import Visualizer, VisImage # <<< MODIFICATION: Import the standard visualizer for GT

from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# constants
WINDOW_NAME = "COCO detections"


def draw_predictions_with_custom_colors(img, predictions, metadata):
    """
    Draw predictions with enforced custom colors for each class using Pillow for better text rendering.
    """
    # Convert OpenCV BGR image to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load a nice font (adjust path and size as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 12)  # Try Arial first
    except:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)  # Fallback font
        # If neither works, it will default to PIL's built-in font
    
    if "instances" in predictions:
        instances = predictions["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get the color for this specific class
            color = metadata.thing_colors[cls]
            class_name = metadata.thing_classes[cls]
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            
            # Prepare text label
            label = f"{class_name}: {score:.2f}"
            
            # Get text size using textbbox
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle(
                [(x1, y1 - text_height - 5), (x1 + text_width, y1)],
                fill=color
            )
            
            # Draw label text
            draw.text(
                (x1, y1 - text_height - 3),
                label,
                fill="white",
                font=font
            )
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_ground_truth_with_custom_colors(img, annotations, metadata):
    """
    Draw ground truth annotations with enforced custom colors for each class using Pillow.
    """
    # Convert OpenCV BGR image to RGB for PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    
    # Create mapping from category_id to class index
    class_mapping = {
        0: 0,  # dent
        1: 1,  # scratch
        2: 2,  # crack
        3: 3,  # glass shatter
        4: 4,  # lamp broken
        5: 5   # tire flat
    }
    
    for ann in annotations:
        bbox = ann["bbox"]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        # Map category_id to correct class index
        category_id = ann["category_id"]
        if category_id not in class_mapping:
            continue  # skip unknown categories
            
        cls = class_mapping[category_id]
        
        # Get the color and class name
        color = metadata.thing_colors[cls]
        class_name = metadata.thing_classes[cls]
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        
        # Prepare text label
        label = f"{class_name}"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle(
            [(x1, y1 - text_height - 10), (x1 + text_width, y1)],
            fill=color
        )
        
        # Draw label text
        draw.text(
            (x1, y1 - text_height - 5),
            label,
            fill="white",
            font=font
        )
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ==========================================================
# ============== REGISTER THE CarDD DATASET ================
# This function ensures that the demo script knows about your custom dataset.
# (No changes needed in this function)
# ==========================================================
def register_cardd_dataset():
    """
    Registers the CarDD dataset and its metadata so the visualizer can find the class names.
    """
    # 1. Define the paths to your dataset
    DATASET_ROOT = "datasets/carDD"
    ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TEST_JSON = os.path.join(ANNOTATIONS_ROOT, "test.json")
    TEST_PATH = os.path.join(DATASET_ROOT, "test")
    
    # 2. Define the class names in the correct order
    CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
    
    # Define the colors for each class
    CLASS_COLORS = [
        (255, 192, 203),  # pink for dent
        (88, 180, 255),      # blue for scratch
        (0, 200, 0),      # green for crack
        (128, 0, 128),    # purple for glass shatter
        (180, 180, 0),    # yellow for lamp broken
        (255, 0, 0)       # red for tire flat
    ]


    # 3. Register the dataset instance (so Detectron2 can load it)
    register_coco_instances("carDD_test", {}, TEST_JSON, TEST_PATH)

    # 4. (This is the critical fix) Explicitly set the "thing_classes" and "thing_colors" metadata
    # for your dataset. This is what the visualizer looks for.
    metadata = MetadataCatalog.get("carDD_test")
    metadata.thing_classes = CLASS_NAMES
    metadata.thing_colors = CLASS_COLORS
# ==========================================================
# ==========================================================


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    # Call the registration function right at the start
    register_cardd_dataset()
    
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # The VisualizationDemo will now automatically find the correct metadata
    demo = VisualizationDemo(cfg)

    # <<< MODIFICATION: Load Ground Truth annotations to create visualizations >>>
    gt_data_map = {}
    if args.input and args.output:
        logger.info("Loading ground truth annotations for visualization...")
        # Define paths to your test annotations
        DATASET_ROOT = "datasets/carDD"
        ANNOTATIONS_ROOT = os.path.join(DATASET_ROOT, "annotations")
        TEST_JSON = os.path.join(ANNOTATIONS_ROOT, "test.json")
        TEST_PATH = os.path.join(DATASET_ROOT, "test")

        # Load the COCO-formatted JSON file
        dataset_dicts = load_coco_json(TEST_JSON, TEST_PATH, dataset_name="carDD_test")

        # Create a dictionary that maps a filename to its annotation data
        gt_data_map = {os.path.basename(d["file_name"]): d for d in dataset_dicts}
        logger.info(f"Loaded {len(gt_data_map)} ground truth annotations.")
    # <<< END MODIFICATION >>>


    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            
            predictions, visualized_output = demo.run_on_image(img)
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                
                # Get metadata for drawing
                metadata = MetadataCatalog.get("carDD_test")

                # Use the new Pillow-based function for prediction visualization
                visualized_output_array = draw_predictions_with_custom_colors(img, predictions, metadata)
                cv2.imwrite(out_filename, visualized_output_array)

                # <<< MODIFICATION: Create and save the Ground Truth visualization with custom colors >>>
                basename = os.path.basename(path)
                if basename in gt_data_map:
                    # Get the ground truth annotations for this specific image
                    gt_dict = gt_data_map[basename]
                    
                    # Create custom ground truth visualization with enforced colors
                    if "annotations" in gt_dict:
                        visualized_gt_array = draw_ground_truth_with_custom_colors(img, gt_dict["annotations"], metadata)
                    else:
                        # No annotations, just use the original image
                        visualized_gt_array = img
                    
                    # Construct the output path for the ground truth image
                    name, ext = os.path.splitext(out_filename)
                    gt_out_filename = f"{name}_gt{ext}"
                    
                    # Save the ground truth visualization
                    cv2.imwrite(gt_out_filename, visualized_gt_array)
                else:
                    logger.warning(f"Could not find ground truth for {basename}. Skipping GT visualization.")
                # <<< END MODIFICATION >>>

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        metadata = MetadataCatalog.get("carDD_test")
        
        # Custom webcam processing with enforced colors
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get predictions
            predictions, _ = demo.run_on_image(frame)
            
            # Create custom visualization
            vis_frame = draw_predictions_with_custom_colors(frame, predictions, metadata)
            
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cap.release()
        cv2.destroyAllWindows()
        
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        
        assert os.path.isfile(args.video_input)
        metadata = MetadataCatalog.get("carDD_test")
        
        # Custom video processing with enforced colors
        for _ in tqdm.tqdm(range(num_frames), total=num_frames):
            ret, frame = video.read()
            if not ret:
                break
                
            # Get predictions
            predictions, _ = demo.run_on_image(frame)
            
            # Create custom visualization
            vis_frame = draw_predictions_with_custom_colors(frame, predictions, metadata)
            
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
