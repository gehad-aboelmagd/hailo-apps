# region imports
# Standard library imports
import datetime
from datetime import datetime
import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.face_recognition_updated.face_recognition_pipeline import GStreamerFaceRecognitionApp

hailo_logger = get_logger(__name__)
# endregion imports


class user_callbacks_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.latest_track_id = -1

        # Business-logic / mode-decision state -----------------------------------------
        # The last successfully recognized person's name (set whenever inference matches
        # a known face).
        self.recognized_name = None
        # Set to True by app_callback() whenever an "Unknown" face is seen. The pipeline's
        # run() loop (in face_recognition_pipeline.py) polls this flag: as soon as it's
        # True it stops the recognition pipeline and starts the live enrollment (train)
        # flow for that person.
        self.unknown_face_detected = False
        # Result of the most recently completed enrollment ("OK" / "NOT OK" / None if no
        # training has run yet), set by GStreamerFaceRecognitionApp.run().
        self.last_train_status = None
        # --------------------------------------------------------------------------------


def app_callback(element, buffer, user_data):
    """
    Business logic for the recognition stage. For each face detection produced by the
    pipeline's vector_db_callback (which already ran the DB lookup and attached a
    classification), decide what happens next:
      - recognized face  -> record/report the person's name (user_data.recognized_name).
      - "Unknown" face   -> flag it (user_data.unknown_face_detected) so the pipeline's
                            run() loop switches to live enrollment (train) mode.
    """
    # Note: Frame counting is handled automatically by the framework wrapper
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        label = detection.get_label()
        detection_confidence = detection.get_confidence()
        if label != "face":
            continue

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) > 0:
            track_id = track[0].get_id()

        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        if len(classifications) == 0:
            continue

        for classification in classifications:
            # Only act once per track ID (avoid re-triggering every frame for the same face)
            if track_id <= user_data.latest_track_id:
                continue
            user_data.latest_track_id = track_id
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if classification.get_label() == 'Unknown':
                print(f'[{timestamp}]: Unknown face detected (track {track_id}, confidence {detection_confidence:.1f}) - starting enrollment...')
                user_data.unknown_face_detected = True  # signals pipeline.run() to switch to TRAIN mode
            else:
                name = classification.get_label()
                confidence = classification.get_confidence()
                user_data.recognized_name = name  # <-- "returns" the recognized person's name
                print(f'[{timestamp}]: Person recognized: {name} (Confidence: {confidence:.1f})')
    return


def main():
    hailo_logger.info("Starting Face Recognition App.")
    user_data = user_callbacks_class()
    pipeline = GStreamerFaceRecognitionApp(app_callback, user_data)

    # The app always starts in RUN (recognition) mode on the live camera feed.
    # From there, pipeline.run() (see face_recognition_pipeline.py) drives the app-level
    # decision automatically based on inference results each frame:
    #   - a recognized face keeps the app in run mode and reports the person's name
    #     (user_data.recognized_name);
    #   - an unrecognized ("Unknown") face switches the app into a live enrollment
    #     (train) flow: it prompts for the person's name, captures samples while they
    #     look in different directions, stores them in the database, and reports an
    #     OK / NOT OK training status (user_data.last_train_status) before resuming
    #     run mode.
    pipeline.options_menu.mode = 'run'
    pipeline.run()


if __name__ == "__main__":
    main()
