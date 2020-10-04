import queue
from threading import Thread
import os
import datetime as dt

import numpy as np
import zarr
import ephem

from picamera import PiCamera
from picamera.array import PiYUVArray


class YUVStorage(PiYUVArray):
    """Storage for image data."""

    def __init__(self, camera, queue, start_time):
        """Initialize storage."""
        self._queue = queue
        self._time = dt.datetime.utcnow()
        self._start_time = start_time
        super(YUVStorage, self).__init__(camera)

    def flush(self):
        """Flush the stream by putting the data to the queue."""
        if dt.datetime.utcnow() > self._start_time:
            self.queue.put((self._time, self.array.copy()))
        self.truncate(0)


class StorageCreator(object):
    """Create storage items until *end_time* is reached."""

    def __init__(self, camera, queue, start_time, end_time):
        """Initialize storage creator."""
        self._camera = camera
        self._queue = queue
        self._start_time = start_time
        self._end_time = end_time

    def get_storage(self):
        """Get a storage object."""
        while True:
            yield YUVStorage(self._camera, self._queue, self._start_time)
            if dt.datetime.utcnow() < self._end_time:
                break


class Stacks(object):
    """Class for managing image stacking."""

    def __init__(self, config, queue, exposure_time):
        """Initialize stacks."""
        self._config = config
        self._queue = queue
        self._exposure_time = exposure_time
        self._sum = None
        self._count = None
        self._max = None
        self._max_time_reference = None
        self._image_times = []
        self._stack_until = None
        self._loop = False

    def _init_stacks(self, image_time, data):
        self._stack_until = image_time + self._config.get('stack_length', )
        self._sum = data.astype(np.uint32)
        self._count = 1
        self._max = data
        self._max_time_reference = np.zeros(data.shape[:2], dtype=np.uint16)

    def _update_sum(self, data):
        self._sum += data

    def _update_count(self):
        self._count += 1

    def _update_max_and_time_reference(self, data):
        y_idxs, x_idxs = np.where(data[:, :, 0] > self._max[:, :, 0])
        for i in range(3):
            self._max[y_idxs, x_idxs, i] = data[y_idxs, x_idxs, i]
        self._max_time_reference[y_idxs, x_idxs] = len(self._image_times)

    def run(self):
        """Run stacking."""
        self._loop = True
        while self._loop:
            try:
                itm = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            image_time, data = itm
            if self._sum is None:
                self._init_stacks(image_time, data)
            else:
                self._update_sum(data)
                self._update_count()
                self._update_max_and_time_reference(data)
            self._image_times.append(np.datetime64(image_time))
            if image_time >= self._stack_until:
                self.save()

    def stop(self):
        """Stop the stacking thread."""
        self._loop = False
        self.save()

    def save(self):
        """Save the stacks."""
        if self._max is None:
            return

        file_path = self._get_file_path()
        with zarr.open(file_path, "w") as fid:
            fid["image_times"] = np.array(self._image_times)
            fid["max"] = self._max
            fid["max_time_reference"] = self._max_time_reference
            fid["sum"] = self._sum
            fid["count"] = self._count
            fid["exposure_time"] = self._exposure_time
        self._clear()

    def _get_file_path(self):
        save_dir = self._config.get("save_dir", ".")
        time_fmt = self._config.get("time_format", "%Y%m%d_%H%M%S.%f")
        fname = self._config["camera_name"] + "_" + dt.datetime.strftime(self._image_times[0], time_fmt) + ".zarr"
        return os.path.join(save_dir, fname)

    def _clear(self):
        self._sum = None
        self._count = None
        self._max = None
        self._max_time_reference = None
        self._image_times = []


class SkyCam(object):
    """Sky Camera."""

    def __init__(self, config):
        """Initialize the camera."""
        self._config = config
        self._camera = None
        self._queue = queue.Queue()
        self._create_camera()
        start_time = dt.datetime.utcnow() + dt.timedelta(seconds=self._config.get('start_delay', 0))
        end_time = self._get_end_time()
        self._storage = StorageCreator(self._config, self._queue, start_time, end_time)
        self._stacks = Stacks(self._config, self._queue, 1. / self._camera.framerate)
        self._stack_thread = Thread(target=self._stacks.run, daemon=True)
        self._stack_thread.start()

    def _create_camera(self):
        """Create the camera instance."""
        self._camera = PiCamera(
            resolution=self._config.get('resolution'),
            framerate=1./self._config['exposure_time'],
            )
        self._camera.iso = self._config.get('iso', 0)
        self._camera.awb_mode = self._config.get('awb_mode', 'auto')

    def _get_end_time(self):
        """Get end time of the imaging period."""
        return next_sun_above_horizon(
            self._config['latitude'],
            self._config['longitude'],
            horizon=self._config.get('sun_elevation_limit', 0.0)
        )

    def run(self):
        """Run the camera."""
        self._camera.capture_sequence(
            self._storage.get_storage(),
            format='yuv',
            use_video_port=True
        )
        self._stacks.stop()
        self._stack_thread.join()
        self._camera.close()


def sun_elevation_now(lat, lon):
    """Calculate Sun elevation now at (*lat*, *lon*)."""
    place = _get_observer(lat, lon)
    sun = ephem.Sun()
    sun.compute(place)

    return np.degrees(sun.alt)


def _get_observer(lat, lon, horizon=0.0):
    place = ephem.Observer()
    place.lat = '%f' % lat
    place.lon = '%f' % lon
    place.horizon = '%f' % horizon

    return place


def next_sun_above_horizon(lat, lon, horizon=0.0):
    """Calculate the time of the next Sunrise."""
    place = _get_observer(lat, lon, horizon=horizon)
    return place.next_rising(ephem.Sun()).datetime()
