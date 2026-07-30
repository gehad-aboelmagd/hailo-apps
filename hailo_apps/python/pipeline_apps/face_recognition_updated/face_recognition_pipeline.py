# region imports
# Standard library imports
import os
import shutil
import json
import sys
import time
import threading
import queue
import uuid
import setproctitle
from pathlib import Path
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
from PIL import Image

# Local application-specific imports
import hailo
from hailo import HailoTracker
from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record
from hailo_apps.python.core.common.core import (
    get_pipeline_parser,
    get_resource_path,
    handle_list_models_flag,
    configure_multi_model_hef_path,
    resolve_hef_paths,
)
from hailo_apps.python.core.common.buffer_utils import get_numpy_from_buffer_efficient, get_caps_from_pad
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp
from hailo_apps.python.core.common.defines import (
    RESOURCES_SO_DIR_NAME, 
    FACE_RECOGNITION_PIPELINE,
    FACE_RECOGNITION_APP_TITLE,
    FACE_DETECTION_POSTPROCESS_SO_FILENAME, 
    FACE_RECOGNITION_POSTPROCESS_SO_FILENAME, 
    FACE_ALIGN_POSTPROCESS_SO_FILENAME, 
    FACE_CROP_POSTPROCESS_SO_FILENAME,
    RESOURCES_VIDEOS_DIR_NAME,
    FACE_RECOGNITION_VIDEO_NAME,
    FACE_RECON_TRAIN_DIR_NAME,
    FACE_RECON_SAMPLES_DIR_NAME,
    RESOURCES_JSON_DIR_NAME,
    FACE_DETECTION_JSON_NAME,
    DEFAULT_LOCAL_RESOURCES_PATH,
    FACE_RECON_DATABASE_DIR_NAME,
    FACE_RECON_LOCAL_SAMPLES_DIR_NAME,
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    SCRFD_10G_POSTPROCESS_FUNCTION,
    SCRFD_2_5G_POSTPROCESS_FUNCTION,
    HAILO8_ARCH,
    HAILO10H_ARCH,
    HAILO8L_ARCH
)
from hailo_apps.python.core.gstreamer.gstreamer_helper_pipelines import QUEUE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE, CROPPER_PIPELINE
from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)
# endregion imports

class GStreamerFaceRecognitionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_pipeline_parser()
        parser.add_argument("--mode", default='run', help="The mode of the application: run, train, delete")
        
        # Configure --hef-path for multi-model support (face detection + face recognition)
        configure_multi_model_hef_path(parser)
        
        # Handle --list-models flag before full initialization
        handle_list_models_flag(parser, FACE_RECOGNITION_PIPELINE)
        
        super().__init__(parser, user_data)
        setproctitle.setproctitle(FACE_RECOGNITION_APP_TITLE)

        # Criteria for when a candidate frame is good enough to try recognize a person from it (e.g., skip the first few frames since in them person only entered the frame and usually is blurry)
        json_file_path = os.path.join(os.path.dirname(__file__), "face_recon_algo_params.json")
        with open(json_file_path, "r+") as json_file:
            self.algo_params = json.load(json_file)
        # 1. How many frames to skip between detection attempts: avoid porocessing first frames since usually they are blurry since person just entered the frame, see self.track_id_frame_count
        self.skip_frames = self.algo_params['skip_frames']
        # 2. Confidence threshold for face classification: if the confidence is below this value, the face will not be recognized
        self.lance_db_vector_search_classificaiton_confidence_threshold = self.algo_params['lance_db_vector_search_classificaiton_confidence_threshold']
        # Both for face detection & recognition networks (not tunable from the UI)
        self.batch_size = self.algo_params['batch_size']

        # Initialize directories
        current_dir = Path(__file__).parent
        self.train_images_dir = current_dir / FACE_RECON_TRAIN_DIR_NAME
        self.samples_dir = current_dir / FACE_RECON_SAMPLES_DIR_NAME
        self.database_dir = current_dir / FACE_RECON_DATABASE_DIR_NAME
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Initialize the database and table
        self.db_handler = DatabaseHandler(db_name='persons.db', 
                                          table_name='persons', 
                                          schema=Record, 
                                          threshold=self.lance_db_vector_search_classificaiton_confidence_threshold,
                                          database_dir=self.database_dir,
                                          samples_dir=self.samples_dir)

        # Architecture is already handled by GStreamerApp parent class
        # Use self.arch which is set by parent
        
        # The app always operates on a LIVE camera feed - in both run (recognition) and
        # train (enrollment) modes. Any pre-recorded/bundled example video source is ignored.
        # NOTE: 'usb' matches CAMERA_KEYWORDS in hailo_apps.python.core.common.defines.
        # Both video_source AND source_type must be set here: super().__init__() (called
        # above) already resolved self.source_type from --input *before* this line runs,
        # so overriding video_source alone leaves source_type as "file" and
        # get_source_pipeline() falls back to a filesrc pointed at the literal string
        # "usb" (that's the "No such file 'usb'" error). Use 'rpi' instead of 'usb' if
        # you're on the Raspberry Pi CSI camera rather than a USB webcam.
        self.video_source = '/dev/video0'
        self.source_type = 'usb'

        # Runtime mode - distinct from self.options_menu.mode (the CLI default). The app
        # always STARTS in 'run' mode; it is switched to 'train' automatically at runtime
        # whenever an unrecognized ("Unknown") face is detected. See run().
        self.mode = 'run'

        # region live-training state
        self.current_person_name = None  # name currently being enrolled, set by enter_train_mode()
        self.processed_names = set()  # ((key-name, val-global_id)) - avoid re-creating a DB record for a name that already exists in this session
        self.captured_samples_count = 0  # number of samples captured for the person currently being enrolled
        self.train_frame_counter = 0  # used to throttle how often we save a sample during training
        self.train_capture_every_n_frames = self.algo_params.get('train_capture_every_n_frames', 5)  # only save 1 out of every N frames to avoid near-duplicate samples
        self.train_directions = self.algo_params.get('train_directions', ["center", "left", "right", "up", "down"])  # head orientations the user is asked to show
        self.train_seconds_per_direction = self.algo_params.get('train_seconds_per_direction', 3)  # how long the pipeline stays PLAYING per direction
        self.min_required_train_samples = self.algo_params.get('min_required_train_samples', 5)  # minimum total samples captured for a training run to be considered "OK"
        # endregion

        # Resolve HEF paths for multi-model app (face detection + face recognition)
        # Uses --hef-path arguments if provided, otherwise uses defaults
        models = resolve_hef_paths(
            hef_paths=self.options_menu.hef_path,  # List from action='append' or None
            app_name=FACE_RECOGNITION_PIPELINE,
            arch=self.arch,
        )
        self.hef_path_detection = models[0].path
        self.hef_path_recognition = models[1].path
    
        if self.arch in (HAILO8_ARCH, HAILO10H_ARCH):
            self.detection_func = SCRFD_10G_POSTPROCESS_FUNCTION
        elif self.arch == HAILO8L_ARCH:
            self.detection_func = SCRFD_2_5G_POSTPROCESS_FUNCTION
        else:
            hailo_logger.error("Unsupported Hailo architecture: %s", self.arch)
            print(
                f"ERROR: Unsupported Hailo architecture: {self.arch}. "
                "Supported architectures are: hailo8, hailo8l, hailo10h.",
                file=sys.stderr
            )
            sys.exit(1)
        
        self.recognition_func = "filter"
        self.cropper_func = "face_recognition"

        # Set the post-processing shared object file
        self.post_process_so_scrfd = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, arch=self.arch, model=FACE_DETECTION_POSTPROCESS_SO_FILENAME)
        self.post_process_so_face_recognition = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, arch=self.arch, model=FACE_RECOGNITION_POSTPROCESS_SO_FILENAME)
        self.post_process_so_face_align = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, arch=self.arch, model=FACE_ALIGN_POSTPROCESS_SO_FILENAME)
        self.post_process_so_cropper = get_resource_path(pipeline_name=None, resource_type=RESOURCES_SO_DIR_NAME, arch=self.arch, model=FACE_CROP_POSTPROCESS_SO_FILENAME)
        
        # Callbacks: bindings between the C++ & Python code
        self.app_callback = app_callback
        self.vector_db_callback_name = "vector_db_callback"
        self.train_vector_db_callback_name = "train_vector_db_callback"
        self.create_pipeline()  # initialize self.pipeline - always starts in 'run' mode
        self.connect_vector_db_callback()
        self.track_id_frame_count = {}  # Dictionary to track frame counts for each track ID - avoid porocessing first frames since usually they are blurry since person just entered the frame 
        self.tracker = HailoTracker.get_instance()  # tracker object

        # region worker queue threads for saving images
        # Create a queue to hold the tasks
        self.task_queue = queue.Queue()

        def worker():
            while True:  # while pipeline playing
                task = self.task_queue.get()
                if task is None:  # Exit signal
                    break

                # Check the task type and process accordingly
                if task['type'] == 'save_image':
                    frame, image_path = task['frame'], task['image_path']
                    self.save_image_file(frame, image_path)
                elif task['type'] == 'send_notification':
                    user_data.send_notification(
                        name=task['name'],
                        global_id=task['global_id'],
                        confidence=task['confidence'],
                        frame=task['frame']
                    )
                self.task_queue.task_done()

        # Start worker threads
        self.num_worker_threads = 1
        self.threads = []
        for i in range(self.num_worker_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            self.threads.append(t)
        # endregion
        
    def get_pipeline_string(self):
        source_pipeline = self.get_source_pipeline()
        detection_pipeline = INFERENCE_PIPELINE(hef_path=self.hef_path_detection, post_process_so=self.post_process_so_scrfd, post_function_name=self.detection_func, batch_size=self.batch_size, config_json=get_resource_path(pipeline_name=None, resource_type=RESOURCES_JSON_DIR_NAME, arch=self.arch, model=FACE_DETECTION_JSON_NAME))
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=-1, kalman_dist_thr=0.7, iou_thr=0.8, init_iou_thr=0.9, keep_new_frames=2, keep_tracked_frames=6, keep_lost_frames=8, keep_past_metadata=True, name='hailo_face_tracker')
        mobile_facenet_pipeline = INFERENCE_PIPELINE(hef_path=self.hef_path_recognition, post_process_so=self.post_process_so_face_recognition, post_function_name=self.recognition_func, batch_size=self.batch_size, config_json=None, name='face_recognition_inference')
        cropper_pipeline = CROPPER_PIPELINE(inner_pipeline=(f'hailofilter so-path={self.post_process_so_face_align} '
                                                            f'name=face_align_hailofilter use-gst-buffer=true qos=false ! '
                                                            f'{QUEUE(name="detector_pos_face_align_q")} ! '
                                                            f'{mobile_facenet_pipeline}'),
                                            so_path=self.post_process_so_cropper, function_name=self.cropper_func, internal_offset=True)
        # 'identity name' - a GStreamer element that does nothing, but allows a probe to be added to it.
        # The video source itself (source_pipeline, built above via self.get_source_pipeline()) is always
        # the LIVE camera feed, in both 'run' and 'train' mode - only the callback element differs, since
        # that's what decides whether incoming faces are matched against the DB (run) or enrolled (train).
        if self.mode == 'train':
            vector_db_callback_pipeline = USER_CALLBACK_PIPELINE(name=self.train_vector_db_callback_name)
        else:
            vector_db_callback_pipeline = USER_CALLBACK_PIPELINE(name=self.vector_db_callback_name)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        return (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{cropper_pipeline} ! '
            f'{vector_db_callback_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
    
    def run(self):
        """
        Top-level control loop. The app ALWAYS starts in 'run' (recognition) mode on the
        live camera feed. The business logic in the app's per-frame callback (app_callback,
        in face_recognition.py) inspects the inference results and sets
        `self.user_data.unknown_face_detected = True` whenever a face could not be matched
        against the database. As soon as that happens here we:
          1. stop the recognition pipeline,
          2. prompt the user for their name,
          3. run a short live enrollment capture across several head orientations,
          4. report an OK / NOT OK training status,
          5. resume recognition mode.
        Loops forever until interrupted (Ctrl+C).
        """
        try:
            while True:
                self.run_recognition()
                if not self.user_data.unknown_face_detected:
                    # run_recognition() returned for another reason (e.g. pipeline/bus error) - stop.
                    break
                person_name = self._prompt_for_name()
                status_ok = self.enter_train_mode(person_name)
                self.user_data.last_train_status = 'OK' if status_ok else 'NOT OK'
                print(f"[TRAIN] Enrollment for '{person_name}' finished with status: {self.user_data.last_train_status}")
        except KeyboardInterrupt:
            print("Interrupted by user - shutting down.")
        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)

    def run_recognition(self):
        """
        Runs the live recognition pipeline (mode='run') until either:
          - an unknown face is detected (self.user_data.unknown_face_detected becomes True), or
          - a pipeline error/EOS is posted on the bus.
        """
        self.mode = 'run'
        self.user_data.unknown_face_detected = False
        self.create_pipeline()
        self.connect_vector_db_callback()
        bus = self.pipeline.get_bus()
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Running in RECOGNITION mode (live camera). Waiting for faces...")
        try:
            while not self.user_data.unknown_face_detected:
                message = bus.timed_pop_filtered(200 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    if message.type == Gst.MessageType.ERROR:
                        err, debug = message.parse_error()
                        hailo_logger.error(f"GStreamer error: {err} ({debug})")
                    break
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _prompt_for_name(self):
        name = input("\nUnrecognized face detected. Please enter your name to enroll: ").strip()
        while not name:
            name = input("Name cannot be empty. Please enter your name: ").strip()
        return name

    def enter_train_mode(self, person_name):
        """
        Live enrollment flow for `person_name`: asks the user to look in a series of
        directions (self.train_directions) and captures face samples/embeddings from the
        live camera feed for each direction, storing them in the vector database.

        Returns:
            bool: True ("OK") if at least self.min_required_train_samples samples were
                  captured and stored, False ("NOT OK") otherwise.
        """
        self.mode = 'train'
        self.current_person_name = person_name
        self.captured_samples_count = 0

        for direction in self.train_directions:
            print(f">>> Please look {direction.upper()} and hold still...")
            time.sleep(1.5)  # give the person a moment to move into position
            self.train_frame_counter = 0
            self.create_pipeline()
            self.connect_train_vector_db_callback()
            try:
                self.pipeline.set_state(Gst.State.PLAYING)
                time.sleep(self.train_seconds_per_direction)
            except Exception as e:
                hailo_logger.error(f"Error capturing direction '{direction}' for '{person_name}': {e}")
            finally:
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)

        self.mode = 'run'
        status_ok = self.captured_samples_count >= self.min_required_train_samples
        print(f"Captured {self.captured_samples_count} sample(s) for '{person_name}'.")
        return status_ok

    def connect_vector_db_callback(self):
        identity = self.pipeline.get_by_name(self.vector_db_callback_name)
        if identity:
            identity_pad = identity.get_static_pad("src")  # src is the output of an element
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.vector_db_callback, self.user_data)  # trigger - when the pad gets buffer
    
    def connect_train_vector_db_callback(self):
        identity = self.pipeline.get_by_name(self.train_vector_db_callback_name)
        if identity:
            identity_pad = identity.get_static_pad("src")  # src is the output of an element
            identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.train_vector_db_callback, self.user_data)  # trigger - when the pad gets buffer

    def save_image_file(self, frame, image_path):
        image = Image.fromarray(frame)
        image.save(image_path, format="JPEG", quality=85)  # Save as a compressed JPEG with quality 85
    
    def crop_frame(self, frame, bbox, width, height):
        # Retrieve the bounding box of the detection to save only the cropped area - useful in case there are more than 1 person in the frame
        # Add extra padding 0.15 to each side of the bounding box
        # Clamp the relative coordinates to the range [0, 1]
        x_min = max(0, min(bbox.xmin()-0.15, 1))
        y_min = max(0, min(bbox.ymin()-0.15, 1))
        x_max = max(0, min(bbox.xmax()+0.15, 1))
        y_max = max(0, min(bbox.ymax()+0.15, 1))

        # Scale the relative coordinates to absolute pixel values
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        x_max = int(x_max * width)
        y_max = int(y_max * height)

        # Crop the frame to the detection area
        return frame[y_min:y_max, x_min:x_max]

    def add_task(self, task_type, **kwargs):
        """
        Add a task to the queue.

        Args:
            task_type (str): The type of task (e.g., 'save_image').
            kwargs: Additional arguments for the task.
        """
        task = {'type': task_type, **kwargs}
        self.task_queue.put(task)

    def get_processed_names_by_name(self, key):
        """
        Retrieve a value from the processed_names set by its key.

        Args:
            key (str): The key to search for.

        Returns:
            Any: The value associated with the key, or None if the key is not found.
        """
        for k, v in self.processed_names:
            if k == key:
                return v
        return None
    
    def is_name_processed(self, key):
        """
        Check if a key exists in the processed_names set.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        for k, _ in self.processed_names:
            if k == key:
                return True
        return False

    def vector_db_callback(self, pad, info, user_data):
        tracker_name = self.tracker.get_trackers_list()[0]  # we have a single tracker
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        format, width, height = get_caps_from_pad(pad)
        roi = hailo.get_roi_from_buffer(buffer)
        
        # for each face detection
        for detection in (d for d in roi.get_objects_typed(hailo.HAILO_DETECTION) if d.get_label() == 'face'):
            track_id = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)[0].get_id() if detection.get_objects_typed(hailo.HAILO_UNIQUE_ID) else None
            
            # still in the skip frames period -skip
            if self.track_id_frame_count.get(track_id, 0) < self.skip_frames:
                self.track_id_frame_count[track_id] = self.track_id_frame_count.get(track_id, 0) + 1
                continue
            
            # after self.skip_frames  
            frame = get_numpy_from_buffer_efficient(buffer, format, width, height)
            embedding = detection.get_objects_typed(hailo.HAILO_MATRIX)  # face recognition embedding
            if len(embedding) == 0:
                continue  # if cropper pipeline element decided to pass the detection - it will arrive to this stage of the pipeline without face embedding
            if len(embedding) > 1:
                print(f"Warning: Multiple embeddings found for track ID {track_id}. Using the first one.") 
                detection.remove_object(embedding[0])
                continue
            # exactly single embedding is expected, so we can safely remove it from the detection
            embedding_vector = np.array(embedding[0].get_data())
            person = self.db_handler.search_record(embedding=embedding_vector)  # most time consuming operation - search the database for the person with the closest embedding
            new_confidence = (1-person['_distance'])
            classification = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            if not classification or classification[0].get_confidence() < new_confidence:
                if classification:
                    detection.remove_object(classification[0])
                new_classification = hailo.HailoClassification(type='face_recon', label=person['label'], confidence=new_confidence)
                detection.add_object(new_classification)
                self.tracker.remove_classifications_from_track(tracker_name, track_id, 'face_recon')
                self.tracker.add_object_to_track(tracker_name, track_id, new_classification)
            
            # anyway re-process for "double-check" after self.skip_frames X 3
            self.track_id_frame_count[track_id] = -3 * self.skip_frames  

        return Gst.PadProbeReturn.OK
    
    def train_vector_db_callback(self, pad, info, user_data):
        """
        Live-enrollment callback. Runs while the pipeline is PLAYING during a single
        direction of enter_train_mode(). Captures one sample per invocation for
        self.current_person_name, throttled to roughly 1 out of every
        self.train_capture_every_n_frames frames so consecutive samples aren't
        near-duplicates of each other.
        """
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        self.train_frame_counter += 1
        if self.train_frame_counter % self.train_capture_every_n_frames != 0:
            return Gst.PadProbeReturn.OK

        format, width, height = get_caps_from_pad(pad)
        frame = get_numpy_from_buffer_efficient(buffer, format, width, height)
        roi = hailo.get_roi_from_buffer(buffer)

        for detection in (d for d in roi.get_objects_typed(hailo.HAILO_DETECTION) if d.get_label() == "face"):
            embedding = detection.get_objects_typed(hailo.HAILO_MATRIX)
            if len(embedding) != 1:  # might be new person, or another sample of an existing person
                continue  # if the cropper pipeline element decided to pass the detection, it arrives here without a face embedding
            detection.remove_object(embedding[0])  # avoid the tracker re-using this same embedding object elsewhere
            cropped_frame = self.crop_frame(frame, detection.get_bbox(), width, height)
            embedding_vector = np.array(embedding[0].get_data())
            image_path = os.path.join(self.samples_dir, f"{uuid.uuid4()}.jpeg")
            self.add_task('save_image', frame=cropped_frame, image_path=image_path)

            name = self.current_person_name
            if self.is_name_processed(name):
                self.db_handler.insert_new_sample(record=self.db_handler.get_record_by_id(self.get_processed_names_by_name(name)), embedding=embedding_vector, sample=image_path, timestamp=int(time.time()))
            else:
                person = self.db_handler.create_record(embedding=embedding_vector, sample=image_path, timestamp=int(time.time()), label=name)
                self.processed_names.add((name, person['global_id']))

            self.captured_samples_count += 1
            print(f"  captured sample #{self.captured_samples_count} for '{name}'")
            return Gst.PadProbeReturn.OK  # one sample per callback invocation is enough
        return Gst.PadProbeReturn.OK