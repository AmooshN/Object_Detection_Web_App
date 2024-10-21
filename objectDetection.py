from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from PIL import Image 
import cv2
import detectron2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import numpy as np
import tqdm

class Detector:


    def __init__(self, model_type="faster_rcnn"):
        self.cfg = get_cfg()
        
        # Variable to hold the configuration file name
        config_file = ""
        
        # Configure model based on the selected type
        if model_type == 'faster_rcnn':
            config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # Updated to available model
        elif model_type == 'mask_rcnn':
            config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # Updated to available model
        elif model_type == 'panoptic_fpn':
            config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"  # Updated to available model
        
        # Load configuration
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)

        output = viz.draw_instance_predictions(predictions['instances'].to('cpu'))
        filename = 'result.jpg'
        cv2.imwrite(filename, output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onVideo(self, videoPath):
        video = cv2.VideoCapture(videoPath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("output.mp4", fourcc, fps=float(frames_per_second), frameSize=(width, height), isColor=True)

        v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

        def runOnVideo(video, maxFrames):
            readFrames = 0
            while True:
                hasFrame, frame = video.read()
                if not hasFrame:
                    break
                outputs = self.predictor(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                visualization = v.draw_instance_predictions(frame, outputs['instances'].to('cpu'))
                visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
                yield visualization

                readFrames += 1
                if readFrames > maxFrames:
                    break

        num_frames = 200  # Limit the number of frames to process
        for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
            video_writer.write(visualization)
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()
