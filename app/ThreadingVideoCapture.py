import threading
import queue
import cv2

class ThreadingVideoCapture:
    def __init__(self, objectId, queueSize=3):
        print("Opening Camera")
        self.video = cv2.VideoCapture(objectId)
        self.q = queue.Queue(maxsize=queueSize)
        self.stopped = False
    
    def start(self):
        thread = threading.Thread(target=self.update, args=())
        thread.start()

        return self
    
    def update(self):
        previousFrame = None
        previousDiff = 0
        delta = 0
        skippedFrames = 0
        queuedFrames = 0

        try:
            while True:
                if self.stopped:
                    return

                ret, frame = self.video.read()

                if not ret:
                    self.stop()
                    return

                if previousFrame is None:
                    previousFrame = frame
                    continue
                
                difference = cv2.subtract(frame, previousFrame)
                b, g, r = cv2.split(difference)
                diff = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)
                delta = abs(diff - previousDiff)

                if delta > 8000:
                    while not self.q.empty():
                        self.q.get()
                    
                    self.q.put(frame)
                    queuedFrames = queuedFrames + 1

                    previousFrame = frame
                    previousDiff = diff
                else:
                    skippedFrames = skippedFrames + 1
                

        except Exception as e:
            print("Error: %s" % str(e))

    def read(self):
        return self.q.get(block=True)
    
    def more(self):
        return self.q.size() > 0
    
    def stop(self):
        self.stopped = True
    
    def release(self):
        self.stopped = True
        self.video.release()
        print("release")
