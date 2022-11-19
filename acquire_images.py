"""Interface a camera to acquire live images and convert them to absorption spectra

"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

configure_path = None

import numpy as np
# install the Linux SDK for scientific cameras: https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame

import threading
import queue

from typing import List

EXPOSURE_TIME = 10 # us
POLL_TIMEOUT = 0 # ms
HEIGHT = 540 # pixels
DB_GAIN = 6 # dB


class CaptureThread(threading.Thread):

    def __init__(self, camera: TLCamera) -> None:
        """Constructor

        Args:
            camera (TLCamera): camera instance
        """

        super(CaptureThread, self).__init__()
        self._camera = camera

        self._camera.exposure_time_us = EXPOSURE_TIME
        self._camera.image_poll_timeout_ms = POLL_TIMEOUT
        self._camera.roi = (0, 540-HEIGHT//2, 1440, 540+HEIGHT//2)
        self._camera.gain = camera.convert_decibels_to_gain(DB_GAIN)

        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self) -> queue.Queue:
        """Get image queue

        Returns:
            queue.Queue: queue containing the two latest images
        """
        return self._image_queue

    def stop(self):
        """Set threading stop event

        """
        self._stop_event.set()

    def _get_spectrum(self, frame: Frame) -> np.ndarray:
        """Get spectrum

        Args:
            frame (Frame): camera frame

        Returns:
            np.ndarray: normalized spectrum of shape [num_width_pixels]
        """
        spectrum = np.copy(frame.image_buffer).mean(axis=0) # average on height
        spectrum /= np.linalg.norm(spectrum) # normalization
        return spectrum

    def run(self):
        """Thread run loop

        """
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    spectrum = self._get_spectrum(frame)
                    self._image_queue.put_nowait(spectrum)
            except queue.Full:
                pass


def update_plot(frame) -> List:
    """Update plot lines

    Returns:
        List: list of matplotlib artists
    """
    spectrum = q.get_nowait()
    for line in lines:
        line.set_ydata(spectrum)
    return lines


dummy_spectrum = np.zeros(1440)

fig, ax = plt.subplots()
lines = ax.plot(dummy_spectrum)
ax.axis((0, len(dummy_spectrum), -1, 1))
ax.set_xlabel('Pixels (-)')
ax.set_ylabel('Intensité (a.u.)')

fig.tight_layout(pad=0)

ani = FuncAnimation(fig, update_plot, blit=True)

with TLCameraSDK() as sdk:

    cameras = sdk.discover_available_cameras()

    with sdk.open_camera(cameras[0]) as camera:
        
        old_roi = camera.roi
        image_acquisition_thread = CaptureThread(camera)
        q = image_acquisition_thread.get_output_queue()

        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.arm(2)
        camera.issue_software_trigger()

        image_acquisition_thread.start()

        plt.show()

        image_acquisition_thread.stop()
        image_acquisition_thread.join()

        camera.disarm()
        camera.roi = old_roi
