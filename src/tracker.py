import cv2

(major, minor) = cv2.__version__.split(".")[:2]

class Tracker(object):
    OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create(),
		"kcf": cv2.TrackerKCF_create(),
		#"mosse": cv2.TrackerMOSSE_create
        }

    def __init__(self, track_model='kcf'):
        '''
            Generate an object tracker based on OpenCV module.

            Arguments:
                -track_model: A string containing the name of tracker to be used.
                            This can be one of the following: 
                            csrt, kcf, mosse.
        
        '''
        if int(major) == 3 and int(minor) < 3:
            self.tracker = cv2.Tracker_create(track_model.upper())
        else:
            if track_model == 'kcf':
                self.tracker = cv2.TrackerKCF_create()
                """
                    elif track_model == 'MOSSE':
                    self.tracker = cv2.TrackerMOSSE_create() 
                """
            elif track_model == "csrt":
                self.tracker = cv2.TrackerCSRT_create()

    def start_tracking(self, image, init_box):
        '''
            Initializer the tracker with a known bounding box that
            surround the target.
        '''
        self.tracker.init(image, init_box)
        pass

    def tracking(self, image):
        (ret, bbox) = self.tracker.update(image)
        return (ret, bbox)
    
